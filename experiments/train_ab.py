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

"""A/B finetune: stock SmolVLA (50 action tokens) vs DCT-16 (16 coefficient tokens).

Both variants start from lerobot/smolvla_base and train with identical data order,
steps, batch size, lr and seed on svla_so100_pickplace (train episodes 0-44). The only
difference is the denoising target space. VLM/vision stay frozen (train_expert_only).

    python train_ab.py --variant baseline --steps 1000
    python train_ab.py --variant dct --k 16 --steps 1000
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
from torch.utils.data import DataLoader  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

from dct_flow import compute_coeff_stats, convert_policy_to_dct  # noqa: E402

REPO = "lerobot/svla_so100_pickplace"
CAMERAS = ["observation.images.top", "observation.images.wrist"]
FPS = 30
TRAIN_EPISODES = list(range(45))  # 45 train / 5 held-out eval
ACTION_DIM = 6


def build_policy(device: str) -> SmolVLAPolicy:
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(ACTION_DIM,))}
    for cam in CAMERAS:
        input_features[cam] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    config = SmolVLAConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
        load_vlm_weights=True,  # pretrained SmolVLM2 for VLM/vision
    )
    policy = SmolVLAPolicy(config)

    # Overlay the full smolvla_base policy weights (expert + projections + VLM finetune).
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ckpt = hf_hub_download("lerobot/smolvla_base", "model.safetensors")
    sd = load_file(ckpt)
    missing, unexpected = policy.load_state_dict(sd, strict=False)
    # Expected: unexpected lm_head keys (we strip the head); nothing else should be missing.
    real_missing = [k for k in missing if "lm_head" not in k]
    print(f"init from smolvla_base: {len(sd)} keys, missing={len(real_missing)}, "
          f"unexpected={len(unexpected)} (lm_head etc.)")
    assert not real_missing, real_missing[:5]

    policy.to(device)
    return policy


def normalized_action_chunks(ds: LeRobotDataset, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """All (T=50)-step windows from the training episodes, MEAN_STD-normalized (for
    coefficient statistics)."""
    import pyarrow.parquet as pq

    path = ds.root / "data/chunk-000/file-000.parquet"
    tbl = pq.read_table(path, columns=["action", "episode_index"])
    actions = np.stack(tbl["action"].to_pylist()).astype(np.float64)
    episodes = np.asarray(tbl["episode_index"])
    chunks = []
    for ep in TRAIN_EPISODES:
        a = actions[episodes == ep]
        for s in range(0, len(a) - 50 + 1, 25):
            chunks.append(a[s : s + 50])
    z = (np.stack(chunks) - mean) / std
    return z


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["baseline", "dct"], required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--energy-weighted",
        action="store_true",
        help="Weight the DCT flow loss by coefficient energy (std^2) - equivalent to "
        "action-space MSE of the kept frequencies",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out or f"./runs/{args.variant}")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    delta = {"action": [i / FPS for i in range(50)]}
    ds = LeRobotDataset(REPO, delta_timestamps=delta, episodes=TRAIN_EPISODES)
    a_mean = np.asarray(ds.meta.stats["action"]["mean"])
    a_std = np.asarray(ds.meta.stats["action"]["std"]) + 1e-8
    s_mean = np.asarray(ds.meta.stats[OBS_STATE]["mean"])
    s_std = np.asarray(ds.meta.stats[OBS_STATE]["std"]) + 1e-8
    print(f"dataset: {len(ds)} frames, {len(TRAIN_EPISODES)} episodes")

    policy = build_policy(args.device)
    policy.train()

    if args.variant == "dct":
        chunks_z = normalized_action_chunks(ds, a_mean, a_std)
        coeff_mean, coeff_std = compute_coeff_stats(
            chunks_z, args.k, policy.config.max_action_dim
        )
        convert_policy_to_dct(
            policy, args.k, coeff_mean, coeff_std,
            energy_weighted=args.energy_weighted, action_dim=ACTION_DIM,
        )
        np.savez(out_dir / "coeff_stats.npz", mean=coeff_mean, std=coeff_std, k=args.k,
                 energy_weighted=args.energy_weighted)
        print(f"DCT variant: K={args.k}, energy_weighted={args.energy_weighted}, "
              f"coeff std range "
              f"[{coeff_std[:, :ACTION_DIM].min():.3f}, {coeff_std[:, :ACTION_DIM].max():.3f}]")

    # Language: single task for the whole dataset; tokenize once (task must end with \n).
    task = ds[0]["task"]
    if not task.endswith("\n"):
        task += "\n"
    tok = policy.model.vlm_with_expert.processor.tokenizer
    enc = tok(task, max_length=policy.config.tokenizer_max_length, truncation=True,
              return_tensors="pt")
    lang_tokens = enc["input_ids"].to(args.device)
    lang_masks = enc["attention_mask"].bool().to(args.device)

    a_mean_t = torch.tensor(a_mean, dtype=torch.float32, device=args.device)
    a_std_t = torch.tensor(a_std, dtype=torch.float32, device=args.device)
    s_mean_t = torch.tensor(s_mean, dtype=torch.float32, device=args.device)
    s_std_t = torch.tensor(s_std, dtype=torch.float32, device=args.device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable params: {n_train / 1e6:.1f}M")
    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-10)

    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2,
                        drop_last=True, persistent_workers=True)

    losses, step, t0 = [], 0, time.perf_counter()
    log_path = out_dir / "train_log.jsonl"
    with open(log_path, "w") as logf:
        while step < args.steps:
            for batch in loader:
                if step >= args.steps:
                    break
                lr = args.lr * min(1.0, (step + 1) / args.warmup)
                for g in optim.param_groups:
                    g["lr"] = lr

                bsize = batch["action"].shape[0]
                model_batch = {
                    OBS_STATE: (batch[OBS_STATE].to(args.device) - s_mean_t) / s_std_t,
                    "action": (batch["action"].to(args.device) - a_mean_t) / a_std_t,
                    OBS_LANGUAGE_TOKENS: lang_tokens.expand(bsize, -1),
                    OBS_LANGUAGE_ATTENTION_MASK: lang_masks.expand(bsize, -1),
                }
                for cam in CAMERAS:
                    model_batch[cam] = batch[cam].to(args.device)

                with torch.autocast(device_type=args.device, dtype=torch.bfloat16):
                    loss, _ = policy.forward(model_batch)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 10.0)
                optim.step()

                losses.append(loss.item())
                step += 1
                if step % 20 == 0 or step == 1:
                    dt_step = (time.perf_counter() - t0) / step
                    rec = {"step": step, "loss": float(np.mean(losses[-20:])),
                           "sec_per_step": round(dt_step, 2)}
                    print(json.dumps(rec), flush=True)
                    logf.write(json.dumps(rec) + "\n")
                    logf.flush()

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "variant": args.variant,
            "k": args.k if args.variant == "dct" else None,
            "steps": args.steps,
            "losses": losses,
        },
        out_dir / "checkpoint.pt",
    )
    print(f"done: final loss (last 50) = {np.mean(losses[-50:]):.4f}, saved to {out_dir}")


if __name__ == "__main__":
    main()
