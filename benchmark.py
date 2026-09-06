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

"""Benchmark chunk-inference latency of the lightweight SmolVLA.

Measures `predict_action_chunk` latency with dummy observations and reports whether a
target control rate (default 30 Hz) is sustainable given the action chunk size.

Smoke test with random weights (only downloads the small SmolVLM2 config/tokenizer):
    python benchmark.py --random --device cpu --iters 3

Jetson (Orin) example with the pretrained checkpoint:
    python benchmark.py --policy-path lerobot/smolvla_base --device cuda \
        --dtype bfloat16 --num-steps 4 --resolution 384 --compile
"""

import argparse
import time

from _local_smolvla import use_local_smolvla

use_local_smolvla()  # must run before any lerobot import

import torch  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def build_policy(args) -> SmolVLAPolicy:
    if args.random:
        input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(args.state_dim,)),
        }
        for i in range(args.cameras):
            input_features[f"observation.images.cam{i}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            )
        output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(args.state_dim,))}
        config = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            load_vlm_weights=False,
        )
        policy = SmolVLAPolicy(config)
    else:
        policy = SmolVLAPolicy.from_pretrained(args.policy_path)

    # Only override what was explicitly requested; otherwise measure the checkpoint's own
    # settings (for --random, the lightweight defaults 4/384 already apply).
    if args.num_steps is not None:
        policy.config.num_steps = args.num_steps
    if args.resolution is not None:
        policy.config.resize_imgs_with_padding = (args.resolution, args.resolution)
    policy.config.attention_implementation = args.attention
    policy.model.vlm_with_expert.attention_implementation = args.attention
    policy.model.vlm_with_expert.cache_expert_kv_projections = not args.no_kv_projection_cache
    return policy


def build_batch(policy: SmolVLAPolicy, args) -> dict:
    device = args.device
    bsize = 1
    state_dim = policy.config.input_features[OBS_STATE].shape[0]
    # Mimic the real preprocessor's padding=max_length behavior: pad language tokens to
    # tokenizer_max_length with mask=False so the padded-row attention path (the one real
    # deployments exercise) is part of the measurement.
    total_len = max(args.lang_tokens, policy.config.tokenizer_max_length)
    lang_mask = torch.zeros(bsize, total_len, dtype=torch.bool, device=device)
    lang_mask[:, : args.lang_tokens] = True
    batch = {
        OBS_STATE: torch.rand(bsize, state_dim, device=device),
        OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (bsize, total_len), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: lang_mask,
    }
    for key, feature in policy.config.input_features.items():
        if feature.type == FeatureType.VISUAL:
            batch[key] = torch.rand(bsize, *feature.shape, device=device)
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument("--random", action="store_true", help="Random weights (no 1GB download)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--num-steps", type=int, default=None, help="Override denoising steps (default: keep config)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override image resize, multiple of 64 (default: keep config)",
    )
    parser.add_argument("--attention", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--no-kv-projection-cache", action="store_true")
    parser.add_argument("--compile", action="store_true", help="torch.compile (CUDA recommended)")
    parser.add_argument("--cameras", type=int, default=2, help="Camera count in --random mode")
    parser.add_argument("--state-dim", type=int, default=6, help="State/action dim in --random mode")
    parser.add_argument("--lang-tokens", type=int, default=20)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.resolution is not None and args.resolution % 64 != 0:
        # Assigning to an existing config bypasses __post_init__ validation; catch it here
        # instead of failing deep inside the vision connector's pixel-shuffle reshape.
        parser.error(
            f"--resolution must be a multiple of 64 (SigLIP patch 16 x pixel-shuffle 4), "
            f"got {args.resolution}"
        )

    policy = build_policy(args)
    policy.eval()
    policy.to(args.device)
    if args.dtype == "bfloat16":
        policy.model = policy.model.to(torch.bfloat16)
    eager_sample = policy.model.sample_actions
    if args.compile:
        torch.set_float32_matmul_precision("high")
        # reduce-overhead (CUDA graphs, no autotuning) suits this batch-1, launch-bound
        # workload better than max-autotune.
        policy.model.sample_actions = torch.compile(eager_sample, mode="reduce-overhead")

    batch = build_batch(policy, args)

    # Inductor fails at first call, not at compile() time, and some backends (e.g. MPS)
    # fail hard - measure eager rather than losing the whole run.
    try:
        for _ in range(args.warmup):
            policy.predict_action_chunk(dict(batch))
    except Exception as e:
        if not args.compile:
            raise
        print(f"!! torch.compile failed ({type(e).__name__}: {e}) - measuring eager instead")
        policy.model.sample_actions, args.compile = eager_sample, False
        for _ in range(args.warmup):
            policy.predict_action_chunk(dict(batch))
    sync(args.device)

    print(
        f"device={args.device} dtype={args.dtype} attention={args.attention} "
        f"num_steps={policy.config.num_steps} resolution={policy.config.resize_imgs_with_padding} "
        f"kv_proj_cache={policy.model.vlm_with_expert.cache_expert_kv_projections} "
        f"compile={args.compile}"
    )

    latencies = []
    for _ in range(args.iters):
        start = time.perf_counter()
        policy.predict_action_chunk(dict(batch))
        sync(args.device)
        latencies.append(time.perf_counter() - start)

    latencies.sort()
    median = latencies[len(latencies) // 2]
    n_action_steps = policy.config.n_action_steps

    print(f"chunk latency: median {median * 1e3:.1f} ms (min {latencies[0] * 1e3:.1f} ms)")
    print(f"actions per chunk: {n_action_steps} (open-loop replay rate {n_action_steps / median:.1f} Hz)")
    print(
        "Closed-loop control rate is NOT latency-derived: the shipped synchronous loop stalls "
        "one full chunk latency at every chunk boundary. Measure the real rate with lerobot "
        "async inference (its budget is chunk_size_threshold * n_action_steps / target_hz)."
    )


if __name__ == "__main__":
    main()
