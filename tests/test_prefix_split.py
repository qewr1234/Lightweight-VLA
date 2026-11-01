"""encode_prefix / sample_from_prefix: the split the async vision/action pipeline runs on.

Paper configuration (8 VLM layers, chunk 10, 2 cameras @512), random weights, CPU, fixed
seeds. Every check is exact (max|diff| == 0): the split must not change a single bit.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from _local_smolvla import use_local_smolvla

use_local_smolvla()

import torch

from smolvla_fast_dct import build_policy, dummy_batch

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def noise_for(policy, seed):
    g = torch.Generator().manual_seed(seed)
    cfg = policy.config
    return torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)


def inputs(policy, batch):
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    return images, img_masks, batch["observation.language.tokens"], batch["observation.language.attention_mask"], state


def maxdiff(a, b):
    return (a - b).abs().max().item()


print("=" * 70)
print("Building paper-config policy (8L, chunk 10, 2 cams @512) ...")
torch.manual_seed(0)
policy = build_policy(dct=False).eval()
model = policy.model
batch_a = dummy_batch(policy)
batch_b = dict(batch_a)
g = torch.Generator().manual_seed(9)
for key in ("observation.images.cam0", "observation.images.cam1"):
    batch_b[key] = torch.rand(batch_a[key].shape, generator=g)

with torch.no_grad():
    full_a = model.sample_actions(*inputs(policy, batch_a), noise=noise_for(policy, 2))
    full_b = model.sample_actions(*inputs(policy, batch_b), noise=noise_for(policy, 2))

    prefix_a = model.encode_prefix(*inputs(policy, batch_a))
    split_a = model.sample_from_prefix(prefix_a, noise=noise_for(policy, 2))
    check("encode_prefix + sample_from_prefix == sample_actions", maxdiff(full_a, split_a) == 0,
          f"max|diff| = {maxdiff(full_a, split_a):.2e}")

    # Action thread: several chunks from one prefix (the expert KV-projection cache, filled
    # on the first chunk, is reused by the later ones).
    again_a = model.sample_from_prefix(prefix_a, noise=noise_for(policy, 2))
    check("one prefix, repeated chunks: identical", maxdiff(split_a, again_a) == 0,
          f"max|diff| = {maxdiff(split_a, again_a):.2e}")
    other_noise = model.sample_from_prefix(prefix_a, noise=noise_for(policy, 3))
    full_other = model.sample_actions(*inputs(policy, batch_a), noise=noise_for(policy, 3))
    check("one prefix, new noise == full call with that noise", maxdiff(other_noise, full_other) == 0,
          f"max|diff| = {maxdiff(other_noise, full_other):.2e}")

    # Vision thread publishes a new prefix: nothing from the old one may leak into it.
    prefix_b = model.encode_prefix(*inputs(policy, batch_b))
    split_b = model.sample_from_prefix(prefix_b, noise=noise_for(policy, 2))
    check("new prefix uses its own KV cache (no leak from the old one)", maxdiff(full_b, split_b) == 0,
          f"max|diff| = {maxdiff(full_b, split_b):.2e}, differs from prefix A by {maxdiff(split_a, split_b):.2e}")
    check("old prefix still valid after a new one is built",
          maxdiff(model.sample_from_prefix(prefix_a, noise=noise_for(policy, 2)), full_a) == 0)

    # posmap path: prefix_offsets is not None and must reach the denoising loop.
    policy.config.posmap_ref_resolution = 640
    full_p = model.sample_actions(*inputs(policy, batch_a), noise=noise_for(policy, 2))
    prefix_p = model.encode_prefix(*inputs(policy, batch_a))
    split_p = model.sample_from_prefix(prefix_p, noise=noise_for(policy, 2))
    check("posmap on: prefix carries offsets, split == full",
          prefix_p.prefix_offsets is not None and maxdiff(full_p, split_p) == 0,
          f"max|diff| = {maxdiff(full_p, split_p):.2e}, vs posmap off {maxdiff(full_p, full_a):.2e}")
    policy.config.posmap_ref_resolution = None

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
