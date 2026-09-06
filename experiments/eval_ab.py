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

"""Held-out evaluation for the baseline-vs-DCT A/B.

For every window of the held-out episodes (45-49), predicts a 50-step action chunk from
the frame's observations and reports open-loop MAE against the ground-truth actions in
raw units (degrees / gripper units), plus chunk-inference latency.

    python eval_ab.py --baseline ./runs/baseline/checkpoint.pt --dct ./runs/dct/checkpoint.pt
"""

import argparse
import json
import sys
import time
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

from dct_flow import convert_policy_to_dct  # noqa: E402
from train_ab import CAMERAS, FPS, REPO, build_policy  # noqa: E402

EVAL_EPISODES = list(range(45, 50))
DIM_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def select_windows(ds: LeRobotDataset, stride: int = 25) -> list[int]:
    """Local dataset indices whose 50-step action window stays inside the episode."""
    eps = ds.hf_dataset["episode_index"]
    frames = ds.hf_dataset["frame_index"]
    eps = np.asarray([int(e) for e in eps])
    frames = np.asarray([int(f) for f in frames])
    ep_len = {int(e): int((eps == e).sum()) for e in np.unique(eps)}
    return [
        i
        for i in range(len(eps))
        if frames[i] % stride == 0 and frames[i] + 50 <= ep_len[int(eps[i])]
    ]


def load_variant(variant: str, ckpt_path: str, device: str):
    policy = build_policy(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if variant == "dct":
        stats = np.load(Path(ckpt_path).parent / "coeff_stats.npz")
        convert_policy_to_dct(
            policy, int(stats["k"]), stats["mean"], stats["std"],
            energy_weighted=bool(stats["energy_weighted"]) if "energy_weighted" in stats else False,
        )
    missing, unexpected = policy.load_state_dict(ckpt["state_dict"], strict=False)
    # coeff_loss_weight is a training-only buffer added after the first runs; checkpoints
    # saved without it are fine (the freshly initialized buffer is used).
    real_missing = [k for k in missing if "lm_head" not in k and "coeff_loss_weight" not in k]
    assert not real_missing and not unexpected, (real_missing[:3], unexpected[:3])
    policy.eval()
    return policy


def evaluate(policy, ds, windows, lang_tokens, lang_masks, stats, device, num_steps):
    policy.config.num_steps = num_steps
    a_mean = torch.tensor(np.asarray(stats["action"]["mean"]), dtype=torch.float32, device=device)
    a_std = torch.tensor(np.asarray(stats["action"]["std"]) + 1e-8, dtype=torch.float32, device=device)
    s_mean = torch.tensor(np.asarray(stats[OBS_STATE]["mean"]), dtype=torch.float32, device=device)
    s_std = torch.tensor(np.asarray(stats[OBS_STATE]["std"]) + 1e-8, dtype=torch.float32, device=device)

    abs_errs = []
    for idx in windows:
        s = ds[idx]
        batch = {
            OBS_STATE: ((s[OBS_STATE].to(device) - s_mean) / s_std).unsqueeze(0),
            OBS_LANGUAGE_TOKENS: lang_tokens,
            OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
        }
        for cam in CAMERAS:
            batch[cam] = s[cam].to(device).unsqueeze(0)
        with torch.no_grad():
            pred_norm = policy.predict_action_chunk(batch)[0]  # (50, 6) normalized
        pred = pred_norm.float() * a_std + a_mean
        gt = s["action"].to(device)  # (50, 6) raw
        valid = ~s["action_is_pad"].to(device)
        abs_errs.append((pred - gt).abs()[valid].reshape(-1, 6).cpu().numpy())

    err = np.concatenate(abs_errs, axis=0)  # (N*50, 6)
    return {
        "mae_per_dim": err.mean(axis=0).tolist(),
        "mae_overall": float(err.mean()),
        "p95_per_dim": np.quantile(err, 0.95, axis=0).tolist(),
    }


def measure_latency(policy, ds, windows, lang_tokens, lang_masks, stats, device, num_steps, reps=10):
    policy.config.num_steps = num_steps
    s = ds[windows[0]]
    s_mean = torch.tensor(np.asarray(stats[OBS_STATE]["mean"]), dtype=torch.float32, device=device)
    s_std = torch.tensor(np.asarray(stats[OBS_STATE]["std"]) + 1e-8, dtype=torch.float32, device=device)
    batch = {
        OBS_STATE: ((s[OBS_STATE].to(device) - s_mean) / s_std).unsqueeze(0),
        OBS_LANGUAGE_TOKENS: lang_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: lang_masks,
    }
    for cam in CAMERAS:
        batch[cam] = s[cam].to(device).unsqueeze(0)
    with torch.no_grad():
        for _ in range(3):
            policy.predict_action_chunk(batch)
        if device == "mps":
            torch.mps.synchronize()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            policy.predict_action_chunk(batch)
            if device == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="./runs/baseline/checkpoint.pt")
    parser.add_argument("--dct", default="./runs/dct/checkpoint.pt")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default="./runs/eval_results.json")
    args = parser.parse_args()

    delta = {"action": [i / FPS for i in range(50)]}
    ds = LeRobotDataset(REPO, delta_timestamps=delta, episodes=EVAL_EPISODES)
    windows = select_windows(ds)
    print(f"eval: {len(windows)} held-out windows from episodes {EVAL_EPISODES}")

    results = {}
    for variant, ckpt in [("baseline", args.baseline), ("dct", args.dct)]:
        policy = load_variant(variant, ckpt, args.device)
        task = ds[0]["task"]
        if not task.endswith("\n"):
            task += "\n"
        tok = policy.model.vlm_with_expert.processor.tokenizer
        enc = tok(task, max_length=policy.config.tokenizer_max_length, truncation=True,
                  return_tensors="pt")
        lang_tokens = enc["input_ids"].to(args.device)
        lang_masks = enc["attention_mask"].bool().to(args.device)

        results[variant] = {}
        for ns in (10, 4):
            r = evaluate(policy, ds, windows, lang_tokens, lang_masks, ds.meta.stats,
                         args.device, num_steps=ns)
            results[variant][f"steps{ns}"] = r
            print(f"{variant} @ {ns} steps: MAE {r['mae_overall']:.3f} | per-dim "
                  + ", ".join(f"{n}={v:.2f}" for n, v in zip(DIM_NAMES, r["mae_per_dim"])),
                  flush=True)
        lat = measure_latency(policy, ds, windows, lang_tokens, lang_masks, ds.meta.stats,
                              args.device, num_steps=4)
        results[variant]["latency_ms_steps4"] = lat * 1e3
        print(f"{variant}: chunk latency @4 steps = {lat * 1e3:.1f} ms", flush=True)

        del policy
        if args.device == "mps":
            torch.mps.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
