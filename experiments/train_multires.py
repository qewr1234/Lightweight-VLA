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

"""Fine-tune a checkpoint so one set of weights works at several input resolutions.

A SmolVLA checkpoint is best at the resolution it was trained with: a smaller input changes
the image features and, without posmap, the position of every token. Here each training step
draws one resolution from a list. Below the checkpoint's own resolution the image tokens are
put back on its position grid (posmap), so the model only has to learn the coarser features.
The vision encoder and the VLM stay frozen and only the action expert is trained, which keeps
the run inside an 8 GB GPU.

    # one model for 512, 384 and 320 px
    python experiments/train_multires.py --resolutions 512 384 320 --steps 2000 --out runs/multires

    # control with the same data order and steps, at the training resolution only
    python experiments/train_multires.py --resolutions 512 --steps 2000 --out runs/control512

    # both are ordinary checkpoints for eval_libero.py
    python experiments/eval_libero.py --policy-path runs/multires \\
        --settings 512:off:0 384:on:0 320:on:0

    # how much memory and time a step takes, before committing to a long run
    python experiments/train_multires.py --resolutions 512 384 320 --steps 20 --no-save
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_libero as E  # noqa: E402  (calls use_local_smolvla())

import numpy as np  # noqa: E402
import torch  # noqa: E402


def freeze_all_but_expert(policy) -> list[torch.nn.Parameter]:
    """Freeze the vision encoder and the VLM; returns the parameters left to train."""
    vlm_with_expert = policy.model.vlm_with_expert
    vlm_with_expert.train_expert_only = True
    vlm_with_expert.freeze_vision_encoder = True
    vlm_with_expert.set_requires_grad()
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True
    return [p for p in policy.parameters() if p.requires_grad]


def resolution_schedule(resolutions: list[int], steps: int, seed: int) -> list[int]:
    """Resolution of every step. Each run of len(resolutions) steps holds each one once, in a
    random order, so the resolutions get equal shares at any stopping point."""
    rng = random.Random(seed)
    schedule: list[int] = []
    while len(schedule) < steps:
        block = list(resolutions)
        rng.shuffle(block)
        schedule += block
    return schedule[:steps]


def set_resolution(policy, resolution: int, train_resolution: int) -> None:
    E.apply_setting(policy, {"resolution": resolution, "posmap": resolution != train_resolution}, train_resolution)


def learning_rate(step: int, steps: int, peak: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to a tenth of the peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress)))


def compute_loss(policy, batch: dict, device: str, noise=None, time_=None) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        loss, _ = policy.forward(dict(batch), noise=noise, time=time_)
    return loss


def train_step(policy, batch: dict, optimizer, trainable, device: str, *, accum_index: int = 0,
               accum: int = 1) -> float:
    """One backward pass; the optimizer steps on every `accum`-th call."""
    loss = compute_loss(policy, batch, device)
    (loss / accum).backward()
    if (accum_index + 1) % accum == 0:
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


def make_dataset(args, chunk_size: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    fps = LeRobotDatasetMetadata(args.dataset).fps
    delta = {"action": [i / fps for i in range(chunk_size)]}
    return LeRobotDataset(args.dataset, delta_timestamps=delta, video_backend=args.video_backend)


def save_checkpoint(policy, preprocessor, postprocessor, out_dir: Path, train_resolution: int) -> None:
    """A directory eval_libero.py --policy-path (and SmolVLAPolicy.from_pretrained) can read."""
    resolution, posmap = policy.config.resize_imgs_with_padding, policy.config.posmap_ref_resolution
    # Saved as it was loaded: the training resolution, posmap off. The evaluator turns
    # posmap on per setting, with this resolution as the reference grid.
    policy.config.resize_imgs_with_padding = (train_resolution, train_resolution)
    policy.config.posmap_ref_resolution = None
    try:
        policy.save_pretrained(out_dir)
        preprocessor.save_pretrained(out_dir)
        postprocessor.save_pretrained(out_dir)
    finally:
        policy.config.resize_imgs_with_padding, policy.config.posmap_ref_resolution = resolution, posmap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--policy-path", default="lerobot/smolvla_libero", help="checkpoint to start from")
    src.add_argument("--random", action="store_true", help="random weights: wiring check only")
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--video-backend", default=None, help="torchcodec (default) or pyav")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[512, 384, 320],
                        help="input resolutions to train on (multiples of 64, none above the checkpoint's)")
    parser.add_argument("--steps", type=int, default=2000, help="optimizer steps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=1,
                        help="backward passes per optimizer step: a larger effective batch in the same memory")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="peak learning rate; low because the checkpoint is already trained")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0, help="also save every N optimizer steps")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--device", default=E.default_device())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="./runs/multires")
    args = parser.parse_args()
    if min(args.steps, args.batch, args.accum, args.warmup) < 1:
        parser.error("--steps, --batch, --accum and --warmup must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_args = SimpleNamespace(random=args.random, policy_path=None if args.random else args.policy_path,
                                rename_map=None, num_steps=None, n_action_steps=None, dtype="float32",
                                device=args.device)
    policy, preprocessor, postprocessor = E.load_policy(load_args)
    resize = policy.config.resize_imgs_with_padding
    if not resize or resize[0] != resize[1]:
        raise SystemExit("this checkpoint has no fixed square input resolution to train around")
    train_resolution = resize[0]
    for r in args.resolutions:
        if r <= 0 or r % 64 or r > train_resolution:
            raise SystemExit(f"resolution {r}: must be a multiple of 64 and at most {train_resolution} "
                             f"(the checkpoint's own)")
    if any(r != train_resolution for r in args.resolutions) and policy.model.add_image_special_tokens:
        raise SystemExit("posmap needs a checkpoint with add_image_special_tokens=false")

    trainable = freeze_all_but_expert(policy)
    policy.train()
    n_train, n_all = sum(p.numel() for p in trainable), sum(p.numel() for p in policy.parameters())
    print(f"start: {'random weights' if args.random else args.policy_path} | checkpoint resolution "
          f"{train_resolution} | training {n_train / 1e6:.1f}M of {n_all / 1e6:.1f}M parameters")
    print(f"resolutions {args.resolutions} | {args.steps} steps x batch {args.batch} x accum {args.accum} "
          f"| lr {args.lr} | device {args.device}")

    dataset = make_dataset(args, policy.config.chunk_size)
    print(f"dataset: {args.dataset}, {len(dataset)} frames")
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=args.workers,
        persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-10)
    optimizer.zero_grad(set_to_none=True)
    schedule = resolution_schedule(args.resolutions, args.steps * args.accum, args.seed)

    recent = {r: [] for r in args.resolutions}
    log_path = out_dir / "train_log.jsonl"
    passes, t0 = 0, time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with open(log_path, "w") as log:
        while passes < len(schedule):
            for batch in loader:
                if passes >= len(schedule):
                    break
                step = passes // args.accum
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup)
                resolution = schedule[passes]
                set_resolution(policy, resolution, train_resolution)
                loss = train_step(policy, preprocessor(batch), optimizer, trainable, args.device,
                                  accum_index=passes, accum=args.accum)
                if not np.isfinite(loss):
                    raise SystemExit(f"loss became {loss} at step {step + 1}; lower --lr")
                recent[resolution] = (recent[resolution] + [loss])[-50:]
                passes += 1
                if passes % args.accum:
                    continue
                step += 1
                if step == 1 or step % 20 == 0 or step == args.steps:
                    record = {"step": step,
                              "loss": {str(r): round(float(np.mean(v)), 4) for r, v in recent.items() if v},
                              "sec_per_step": round((time.perf_counter() - t0) / step, 2)}
                    if args.device == "cuda":
                        record["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                if args.save_every and step % args.save_every == 0 and step < args.steps and not args.no_save:
                    save_checkpoint(policy, preprocessor, postprocessor, out_dir, train_resolution)
                    print(f"saved {out_dir} at step {step}", flush=True)

    summary = {"start": "random" if args.random else args.policy_path, "dataset": args.dataset,
               "resolutions": args.resolutions, "train_resolution": train_resolution, "steps": args.steps,
               "batch": args.batch, "accum": args.accum, "lr": args.lr, "seed": args.seed,
               "trainable_params": n_train, "wall_s": round(time.perf_counter() - t0, 1),
               "final_loss": {str(r): round(float(np.mean(v)), 4) for r, v in recent.items() if v}}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    if args.no_save:
        print(f"done (--no-save): {summary['final_loss']}")
        return
    save_checkpoint(policy, preprocessor, postprocessor, out_dir, train_resolution)
    print(f"done: {summary['final_loss']} | saved {out_dir}")


if __name__ == "__main__":
    main()
