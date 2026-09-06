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

"""Produce a deployment-optimized SmolVLA checkpoint.

Loads a pretrained SmolVLA policy with this folder's lightweight code (LM head stripped,
SDPA attention, expert-KV-projection caching) and bakes lighter inference settings
(num_steps, image resolution) into the saved config. Checkpoint settings are KEPT unless
you pass --num-steps / --resolution explicitly.

The saved config.json is stock-lerobot compatible (fork-only fields are stripped), and
companion files sitting next to a local source checkpoint (pre/post-processor configs,
normalization stats, ...) are copied through.

Weights are saved as-is: smolvla_base is already bf16 apart from the projections, and
safetensors restores dtypes per the *receiving* module, so a bf16 save bought ~6MB of disk
and zero runtime memory. Run in bf16 by casting after loading instead
(`policy.model.to(torch.bfloat16)`, which is what benchmark.py --dtype bfloat16 does).

Example (from this folder, inside an env where lerobot is installed):
    python optimize_for_inference.py \
        --policy-path lerobot/smolvla_base \
        --output-dir ./smolvla_optimized --num-steps 4
"""

import argparse
import shutil
from pathlib import Path

from _local_smolvla import use_local_smolvla

use_local_smolvla()  # must run before any lerobot import

import torch  # noqa: E402

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def summarize(policy: SmolVLAPolicy) -> None:
    model = policy.model
    vlm_with_expert = model.vlm_with_expert
    rows = {
        "vision encoder": count_params(vlm_with_expert.get_vlm_model().vision_model),
        "VLM text layers": count_params(vlm_with_expert.get_vlm_model().text_model),
        "action expert": count_params(vlm_with_expert.lm_expert),
        "total": count_params(policy),
    }
    bytes_total = sum(p.numel() * p.element_size() for p in policy.parameters())
    for name, n in rows.items():
        print(f"  {name:>16}: {n / 1e6:8.1f}M params")
    print(f"  {'weight memory':>16}: {bytes_total / 1e6:8.1f} MB")
    print(
        f"  {'inference cfg':>16}: num_steps={policy.config.num_steps}, "
        f"resize={policy.config.resize_imgs_with_padding}"
    )


def copy_companion_files(source_dir: Path, output_dir: Path) -> None:
    """Copy processor configs / stats etc. that live next to a local checkpoint."""
    ours = {"config.json", "model.safetensors"}
    for item in sorted(source_dir.iterdir()):
        if item.name in ours or item.name.startswith("."):
            continue
        if item.is_dir():
            shutil.copytree(item, output_dir / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, output_dir / item.name)
        print(f"  copied companion file: {item.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument("--output-dir", default="./smolvla_optimized")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Flow-matching steps to bake into the saved config (default: keep the "
        "checkpoint's value; 4 is a good latency/accuracy tradeoff)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Image resize to bake into the saved config, multiple of 64 (default: keep "
        "the checkpoint's value). Finetune at the new resolution for best accuracy.",
    )
    args = parser.parse_args()
    if args.resolution is not None and args.resolution % 64 != 0:
        # Validate here: assigning to an existing config bypasses __post_init__, so a bad
        # value would save fine but make the checkpoint unloadable via from_pretrained.
        parser.error(
            f"--resolution must be a multiple of 64 (SigLIP patch 16 x pixel-shuffle 4), "
            f"got {args.resolution}"
        )

    print(f"Loading {args.policy_path} with lightweight SmolVLA code ...")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.eval()

    print("Before optimization (LM head already stripped at construction):")
    summarize(policy)

    changed = []
    if args.num_steps is not None and args.num_steps != policy.config.num_steps:
        changed.append(f"num_steps: {policy.config.num_steps} -> {args.num_steps}")
        policy.config.num_steps = args.num_steps
    if args.resolution is not None:
        new_res = (args.resolution, args.resolution)
        if new_res != tuple(policy.config.resize_imgs_with_padding or ()):
            changed.append(
                f"resize_imgs_with_padding: {policy.config.resize_imgs_with_padding} -> {new_res}"
            )
            policy.config.resize_imgs_with_padding = new_res
    if changed:
        print("Baking inference settings into the saved config:")
        for c in changed:
            print(f"  {c}")
    else:
        print("Keeping the checkpoint's inference settings unchanged.")

    print("After optimization:")
    summarize(policy)

    output_dir = Path(args.output_dir)
    policy.save_pretrained(output_dir)

    source = Path(args.policy_path)
    if source.is_dir():
        copy_companion_files(source, output_dir)
    else:
        print(
            "  (hub source: no companion files to copy - lerobot/smolvla_base ships only "
            "config.json + model.safetensors)"
        )

    print(f"Saved optimized checkpoint to {output_dir}")
    print(
        "Validate on your task before trusting it: fewer denoising steps and a lower "
        "resolution trade accuracy for speed - and the resolution is by far the more "
        "expensive of the two. Measure task success rate, not just action MSE."
    )


if __name__ == "__main__":
    main()
