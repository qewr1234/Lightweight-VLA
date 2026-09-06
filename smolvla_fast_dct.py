#!/usr/bin/env python
"""Architecture skeleton for SmolVLA-FAST DCT (paper configuration).

This builds the paper's *architecture* from the pieces already in this repo:

    VLM layers 16 -> 8          config: num_vlm_layers=8      (slices text_model.layers[:8])
    chunk size 50 -> 10         config: chunk_size=n_action_steps=10
    DCT frequency domain        experiments/dct_flow.py::convert_policy_to_dct(k=10)

Config verified against the paper's Table 1: with `strip_lm_head=False` this builds
322.3M total / 50.8M trainable, vs the paper's 322.3M / 50.6M. (The default here is
275.0M because this repo drops the unused 47.3M LM head, which SmolVLA never calls.)

WHAT THIS IS NOT
----------------
Slicing to 8 layers throws away half of a pretrained transformer, so the model produced
here outputs noise until it is retrained (the paper: 30k steps, batch 128, 80 episodes of
SO-101 cube grasping). This file gives you the shape, not the policy.

Also absent, because none of it lives in this repo: the 2-thread async pipeline with the
shared prefix KV cache, SO-101 motor I/O / bus_lock, and the camera threads. Those are the
paper's actual contribution and they run on the Jetson.

Coefficient statistics: real training needs per-(frequency, dim) mean/std from the dataset.
`experiments/train_ab.py` already computes them via `compute_coeff_stats`; pass them in as
`coeff_stats`. Without them this falls back to mean 0 / std 1, which is fine for a shape
smoke test and wrong for training.

Usage:
    python smolvla_fast_dct.py            # build + forward a dummy batch, print shapes
    python smolvla_fast_dct.py --no-dct   # same, time domain (the ablation arm)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))

from _local_smolvla import use_local_smolvla  # noqa: E402

use_local_smolvla()

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

# Paper configuration (Table 1).
NUM_VLM_LAYERS = 8
CHUNK_SIZE = 10
RESOLUTION = 512
CAMERAS = 2
ACTION_DIM = 6  # SO-101 6-DoF


def build_config(*, cameras=CAMERAS, action_dim=ACTION_DIM, load_vlm_weights=False):
    """SmolVLAConfig with the paper's architecture values."""
    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(action_dim,)),
        **{
            f"observation.images.cam{i}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
            for i in range(cameras)
        },
    }
    return SmolVLAConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))},
        num_vlm_layers=NUM_VLM_LAYERS,
        chunk_size=CHUNK_SIZE,
        n_action_steps=CHUNK_SIZE,
        resize_imgs_with_padding=(RESOLUTION, RESOLUTION),
        load_vlm_weights=load_vlm_weights,
    )


def build_policy(*, dct=True, coeff_stats=None, action_dim=ACTION_DIM, **kwargs):
    """Paper-shaped policy. `coeff_stats` is the (mean, std) pair from compute_coeff_stats."""
    policy = SmolVLAPolicy(build_config(action_dim=action_dim, **kwargs))
    if not dct:
        return policy

    from dct_flow import convert_policy_to_dct

    k = CHUNK_SIZE  # no truncation: an orthonormal rotation, same token count
    if coeff_stats is None:
        print("!! coeff_stats=None -> mean 0 / std 1. Shape check only, NOT trainable.")
        mean = np.zeros((k, policy.config.max_action_dim))
        std = np.ones((k, policy.config.max_action_dim))
    else:
        mean, std = coeff_stats
    return convert_policy_to_dct(policy, k, mean, std, action_dim=action_dim)


def dummy_batch(policy, device="cpu"):
    cfg = policy.config
    n_lang = cfg.tokenizer_max_length
    batch = {
        OBS_STATE: torch.rand(1, cfg.input_features[OBS_STATE].shape[0], device=device),
        OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, n_lang), device=device),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, n_lang, dtype=torch.bool, device=device),
    }
    for key, feat in cfg.input_features.items():
        if feat.type == FeatureType.VISUAL:
            batch[key] = torch.rand(1, *feat.shape, device=device)
    return batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--no-dct", action="store_true", help="time domain (ablation arm)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--load-vlm-weights", action="store_true", help="download SmolVLM2 backbone")
    args = p.parse_args()

    policy = build_policy(dct=not args.no_dct, load_vlm_weights=args.load_vlm_weights)
    policy.eval().to(args.device)

    n_params = sum(p_.numel() for p_ in policy.parameters())
    n_train = sum(p_.numel() for p_ in policy.parameters() if p_.requires_grad)
    print(
        f"domain={'DCT' if not args.no_dct else 'time'} "
        f"vlm_layers={policy.model.vlm_with_expert.num_vlm_layers} "
        f"chunk={policy.config.chunk_size} resolution={policy.config.resize_imgs_with_padding}"
    )
    print(f"params: {n_params / 1e6:.1f}M total, {n_train / 1e6:.1f}M trainable")

    batch = dummy_batch(policy, args.device)
    t0 = time.perf_counter()
    chunk = policy.predict_action_chunk(dict(batch))
    dt = time.perf_counter() - t0
    print(f"chunk: {tuple(chunk.shape)} in {dt * 1e3:.0f} ms (untrained weights)")

    # The only thing worth asserting here: the shape contract survives both arms.
    assert chunk.shape == (1, CHUNK_SIZE, ACTION_DIM), chunk.shape
    assert torch.isfinite(chunk).all(), "non-finite actions"
    print("OK - shape contract holds. Model is UNTRAINED; retrain before any claim.")


if __name__ == "__main__":
    main()
