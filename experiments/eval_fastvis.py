# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Staleness A/B: does the fast path recover what a stale slow path loses?

For every held-out window the slow inputs (all cameras + state) come from `d` frames before
t - what the action thread sees in the async pipeline - while the ground-truth actions, and
the fast inputs of the fastvis model, come from t. Sweeping d from 0 (synchronous, the upper
bound) through the deployment range shows how much accuracy staleness costs the baseline and
how much of it the fast path wins back.

    python eval_fastvis.py --baseline ./runs/baseline/checkpoint.pt \\
                           --fastvis ./runs/fastvis/checkpoint.pt
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _local_smolvla import use_local_smolvla

use_local_smolvla()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from eval_ab import EVAL_EPISODES, load_variant, select_windows  # noqa: E402
from fast_vision import (  # noqa: E402
    FAST_IMAGE,
    FAST_STATE,
    FastVisionConfig,
    attach_fast_vision,
    stale_frame_offsets,
    staleness_delta_timestamps,
)
from train_ab import CAMERAS, FPS, REPO, build_policy, default_device  # noqa: E402


def load_fastvis(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt["fast_vision"])
    cfg["image_hw"] = tuple(cfg["image_hw"])
    cfg["pretrained_backbone"] = False  # the trained trunk weights come from the checkpoint
    policy = attach_fast_vision(build_policy(device), FastVisionConfig(**cfg))
    missing, unexpected = policy.load_state_dict(ckpt["state_dict"], strict=False)
    real_missing = [k for k in missing if "lm_head" not in k]
    assert not real_missing and not unexpected, (real_missing[:3], unexpected[:3])
    return policy.eval(), cfg["camera_key"]


def evaluate(policy, ds, windows, offsets, d, lang, stats, device, fast_camera=None):
    """Open-loop MAE (raw units) with the slow inputs `d` frames stale."""
    frame = 0 if d == 0 else 1 + offsets.index(d)
    a_mean = torch.tensor(np.asarray(stats["action"]["mean"]), dtype=torch.float32, device=device)
    a_std = torch.tensor(np.asarray(stats["action"]["std"]) + 1e-8, dtype=torch.float32, device=device)
    s_mean = torch.tensor(np.asarray(stats[OBS_STATE]["mean"]), dtype=torch.float32, device=device)
    s_std = torch.tensor(np.asarray(stats[OBS_STATE]["std"]) + 1e-8, dtype=torch.float32, device=device)

    errs = []
    for idx in windows:
        s = ds[idx]
        batch = {
            OBS_STATE: ((s[OBS_STATE][frame].to(device) - s_mean) / s_std).unsqueeze(0),
            OBS_LANGUAGE_TOKENS: lang[0],
            OBS_LANGUAGE_ATTENTION_MASK: lang[1],
        }
        for cam in CAMERAS:
            batch[cam] = s[cam][frame].to(device).unsqueeze(0)
        if fast_camera is not None:
            batch[FAST_IMAGE] = s[fast_camera][0].to(device).unsqueeze(0)
            batch[FAST_STATE] = ((s[OBS_STATE][0].to(device) - s_mean) / s_std).unsqueeze(0)
        policy.reset()
        with torch.no_grad():
            pred = policy.predict_action_chunk(batch)[0].float() * a_std + a_mean
        valid = ~s["action_is_pad"].to(device)
        errs.append((pred - s["action"].to(device)).abs()[valid].reshape(-1, 6).cpu().numpy())
    return float(np.concatenate(errs).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="./runs/baseline/checkpoint.pt")
    parser.add_argument("--fastvis", default="./runs/fastvis/checkpoint.pt")
    parser.add_argument("--vision-period-ms", type=float, default=260.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--out", default="./runs/eval_fastvis.json")
    args = parser.parse_args()

    offsets = stale_frame_offsets(args.vision_period_ms / 1e3, FPS)
    fastvis, fast_camera = load_fastvis(args.fastvis, args.device)
    delta = {"action": [i / FPS for i in range(50)]}
    delta.update(staleness_delta_timestamps(FPS, offsets, CAMERAS, fast_camera))
    ds = LeRobotDataset(REPO, delta_timestamps=delta, episodes=EVAL_EPISODES)
    windows = select_windows(ds)
    print(f"eval: {len(windows)} held-out windows, stale offsets {offsets} frames")

    task = ds[0]["task"]
    task = task if task.endswith("\n") else task + "\n"
    tok = fastvis.model.vlm_with_expert.processor.tokenizer
    enc = tok(task, max_length=fastvis.config.tokenizer_max_length, truncation=True, return_tensors="pt")
    lang = (enc["input_ids"].to(args.device), enc["attention_mask"].bool().to(args.device))

    baseline = load_variant("baseline", args.baseline, args.device)
    results = {"offsets_frames": offsets, "num_steps": args.num_steps, "rows": []}
    print(f"{'stale':>12} | {'baseline':>9} | {'fastvis':>9}")
    for d in [0, *offsets]:
        row = {"stale_frames": d, "stale_ms": round(d * 1e3 / FPS)}
        for name, policy, cam in (("baseline", baseline, None), ("fastvis", fastvis, fast_camera)):
            policy.config.num_steps = args.num_steps
            row[name] = evaluate(policy, ds, windows, offsets, d, lang, ds.meta.stats, args.device, cam)
        results["rows"].append(row)
        print(f"{d:3d} fr {row['stale_ms']:4d}ms | {row['baseline']:8.3f}° | {row['fastvis']:8.3f}°", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
