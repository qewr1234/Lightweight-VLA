"""Fast vision path: equivalence, sensitivity, training and staleness checks.

Paper configuration (8 VLM layers, chunk 10, 2 cameras @512), random weights, CPU, fixed
seeds. The ResNet trunk is built without ImageNet weights so no download is needed.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from _local_smolvla import use_local_smolvla

use_local_smolvla()

import torch

from fast_vision import (
    FAST_IMAGE,
    FAST_STATE,
    FastVisionConfig,
    apply_staleness,
    attach_fast_vision,
    stale_frame_offsets,
    staleness_delta_timestamps,
)
from lerobot.utils.constants import OBS_STATE
from smolvla_fast_dct import build_policy, dummy_batch

FAST_CAM = "observation.images.cam0"  # fast path reuses one of the slow cameras here
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def noise_for(policy, seed):
    g = torch.Generator().manual_seed(seed)
    cfg = policy.config
    return torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=g)


def chunk(policy, batch, seed):
    policy.reset()
    with torch.no_grad():
        return policy.predict_action_chunk(dict(batch), noise=noise_for(policy, seed))


def maxdiff(a, b):
    return (a - b).abs().max().item()


print("=" * 70)
print("Building paper-config policy (8L, chunk 10, 2 cams @512) ...")
torch.manual_seed(0)
policy = build_policy(dct=False).eval()
batch = dummy_batch(policy)
base_params = sum(p.numel() for p in policy.parameters())

# 1) The split API reproduces the monolithic call exactly.
reference = chunk(policy, batch, seed=2)
with torch.no_grad():
    images, img_masks = policy.prepare_images(batch)
    state = policy.prepare_state(batch)
    prefix = policy.model.encode_prefix(
        images, img_masks, batch["observation.language.tokens"],
        batch["observation.language.attention_mask"], state,
    )
    split = policy.model.sample_from_prefix(prefix, noise=noise_for(policy, 2))[:, :, :6]
    again = policy.model.sample_from_prefix(prefix, noise=noise_for(policy, 2))[:, :, :6]
check("encode_prefix + sample_from_prefix == sample_actions", maxdiff(reference, split) == 0,
      f"max|diff| = {maxdiff(reference, split):.2e}")
check("prefix cache reusable across chunks", maxdiff(split, again) == 0,
      f"max|diff| = {maxdiff(split, again):.2e}")

# 2) Attaching the fast path changes nothing until the gates are trained.
cfg = FastVisionConfig(camera_key=FAST_CAM, pretrained_backbone=False)
attach_fast_vision(policy, cfg)
policy.eval()
added = sum(p.numel() for p in policy.parameters()) - base_params
attached = chunk(policy, batch, seed=2)
check("attached, gates = 0: bit-identical to the original", maxdiff(reference, attached) == 0,
      f"max|diff| = {maxdiff(reference, attached):.2e}")
print(f"       fast path adds {added / 1e6:.2f}M params "
      f"({policy.model.fast_encoder.num_tokens} image tokens + 1 state token)")

# 3) Once the gates open, actions depend on the fast image (and only through it).
with torch.no_grad():
    policy.model.fast_adapter.attn_gate.fill_(0.5)
    policy.model.fast_adapter.ff_gate.fill_(0.5)
g = torch.Generator().manual_seed(7)
other = dict(batch)
other[FAST_IMAGE] = torch.rand(batch[FAST_CAM].shape, generator=g)
open_same = chunk(policy, batch, seed=2)
open_other = chunk(policy, other, seed=2)
check("gates open: output moves away from the original", maxdiff(reference, open_same) > 1e-4,
      f"max|diff| = {maxdiff(reference, open_same):.2e}")
check("gates open: a different fast image changes the chunk", maxdiff(open_same, open_other) > 1e-4,
      f"max|diff| = {maxdiff(open_same, open_other):.2e}")

# 4) Async halves: slow cache built once, fast path refreshed per tick.
prefix = policy.encode_slow(batch)
tick_a = policy.act_fast(prefix, batch, noise=noise_for(policy, 2))
tick_b = policy.act_fast(prefix, other, noise=noise_for(policy, 2))
check("act_fast(encode_slow) == predict_action_chunk", maxdiff(tick_a, open_same) == 0,
      f"max|diff| = {maxdiff(tick_a, open_same):.2e}")
check("same slow cache, new fast frame -> new chunk", maxdiff(tick_a, tick_b) > 1e-4,
      f"max|diff| = {maxdiff(tick_a, tick_b):.2e}")
print(f"       chunk shape {tuple(tick_a.shape)}")

# 5) Training: loss finite, gradients reach the gates (and, once open, the trunk).
with torch.no_grad():
    policy.model.fast_adapter.attn_gate.zero_()
    policy.model.fast_adapter.ff_gate.zero_()
policy.train()
train_batch = dict(batch)
train_batch["action"] = torch.randn(1, policy.config.chunk_size, 6)
loss, _ = policy.forward(train_batch)
loss.backward()
gate_grad = policy.model.fast_adapter.attn_gate.grad
check("training forward from gates = 0: loss finite, gate gets gradient",
      torch.isfinite(loss) and gate_grad is not None and gate_grad.abs().item() > 0,
      f"loss = {loss.item():.4f}, |d gate| = {0 if gate_grad is None else gate_grad.abs().item():.2e}")
policy.zero_grad(set_to_none=True)
with torch.no_grad():
    policy.model.fast_adapter.attn_gate.fill_(0.5)
loss, _ = policy.forward(train_batch)
loss.backward()
trunk_grad = policy.model.fast_encoder.proj.weight.grad
check("gates open: gradient reaches the fast encoder",
      trunk_grad is not None and trunk_grad.abs().sum().item() > 0,
      f"|d proj| = {0 if trunk_grad is None else trunk_grad.abs().sum().item():.2e}")
policy.zero_grad(set_to_none=True)
policy.eval()

# 6) Staleness simulation picks one past frame for the slow path, keeps the fast path fresh.
offsets = stale_frame_offsets(vision_period_s=0.26, fps=30)
check("deployment age 0.26-0.52 s @30 fps -> offsets 8..16 frames", offsets[0] == 8 and offsets[-1] == 16,
      f"offsets = {offsets}")
slow_keys = ["observation.images.cam0", "observation.images.cam1"]
delta = staleness_delta_timestamps(30, offsets, slow_keys, fast_key=FAST_CAM)
n = len(delta[OBS_STATE])
B = 4
# Synthetic stacked batch where every frame's value encodes its own index.
frames = torch.arange(n, dtype=torch.float32).view(1, n, 1, 1, 1).expand(B, n, 3, 8, 8).clone()
stacked = {k: frames.clone() for k in slow_keys}
stacked[OBS_STATE] = torch.arange(n, dtype=torch.float32).view(1, n, 1).expand(B, n, 6).clone()
out = apply_staleness(stacked, slow_keys, FAST_CAM, generator=torch.Generator().manual_seed(0))
pick = out["fast_vision.stale_index"]
stale_ok = all(
    (out[k][i] == pick[i]).all() and (out[OBS_STATE][i] == pick[i]).all() for k in slow_keys for i in range(B)
)
fresh_ok = (out[FAST_IMAGE] == 0).all() and (out[FAST_STATE] == 0).all()
check("staleness: slow cams + state share one past frame per sample", stale_ok and (pick >= 1).all(),
      f"picked indices = {pick.tolist()}")
check("staleness: fast image and fast state stay at time t", fresh_ok)
check("staleness: delta_timestamps covers t and every offset",
      delta[OBS_STATE][0] == 0 and len(delta[OBS_STATE]) == len(offsets) + 1)

# 7) Cost of the fast path per chunk vs. the slow path (CPU, informational).
def median_ms(fn, reps=5):
    fn()
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return sorted(xs)[len(xs) // 2] * 1e3

with torch.no_grad():
    slow_ms = median_ms(lambda: policy.encode_slow(batch), reps=3)
    fast_ms = median_ms(lambda: policy.model.encode_fast(batch[FAST_CAM], batch[OBS_STATE]))
    act_ms = median_ms(lambda: policy.act_fast(prefix, batch), reps=3)
check("fast encoder is far cheaper than the slow path (informational)", fast_ms < slow_ms / 10,
      f"slow {slow_ms:.0f} ms | fast encode {fast_ms:.1f} ms | act_fast {act_ms:.0f} ms per chunk (CPU)")

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
