"""Equivalence + speed check for the batched multi-camera vision path."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _local_smolvla import use_local_smolvla

use_local_smolvla()

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.constants import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
)

STATE_DIM = 6
CAMERAS = 3  # stress with 3 cameras
LANG_LEN = 12


def make_policy(seed=0):
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
    for i in range(CAMERAS):
        input_features[f"observation.images.cam{i}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        )
    torch.manual_seed(seed)
    policy = SmolVLAPolicy(SmolVLAConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,))},
        load_vlm_weights=False,
    ))
    policy.eval()
    return policy


def make_batch(seed=1, masked=True):
    g = torch.Generator().manual_seed(seed)
    batch = {
        OBS_STATE: torch.rand(1, STATE_DIM, generator=g),
        OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, LANG_LEN), generator=g),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, LANG_LEN, dtype=torch.bool),
    }
    for i in range(CAMERAS):
        batch[f"observation.images.cam{i}"] = torch.rand(1, 3, 480, 640, generator=g)
    if masked:
        batch["observation.images.cam2_padding_mask"] = torch.zeros(1, dtype=torch.bool)
    return batch


policy = make_policy()
batch = make_batch()
g = torch.Generator().manual_seed(2)
noise = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=g)


def run(batched: bool):
    policy.config.batch_vision_encoder = batched
    policy.reset()
    with torch.no_grad():
        return policy.predict_action_chunk(dict(batch), noise=noise.clone())


# The invariant that actually defines "batching is correct": running the cameras as one
# batch through the vision encoder must return the same features as running them one by
# one. Checked here directly, where nothing amplifies - tight relative tolerance.
embed = policy.model.vlm_with_expert.embed_image
cam_imgs = [torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(i)) for i in range(3)]
with torch.no_grad():
    emb_seq = torch.cat([embed(img) for img in cam_imgs], dim=0)
    emb_bat = embed(torch.cat(cam_imgs, dim=0))
emb_scale = emb_seq.abs().max().item()
emb_rel = (emb_seq - emb_bat).abs().max().item() / emb_scale
ok_emb = emb_rel < 1e-5
print(f"[{'PASS' if ok_emb else 'FAIL'}] vision encoder: batched == sequential "
      f"rel = {emb_rel:.2e} (scale {emb_scale:.2f})")

# End to end, the same float noise is amplified by 16 untrained VLM layers and num_steps
# of Euler integration, so it lands around 1e-3 relative on CPU and near 0 on a backend
# whose batched and unbatched GEMMs reduce in the same order (MPS). A semantic break in
# the batching would show up here as an O(1) difference, not as a few 1e-3.
out_seq = run(False)
out_bat = run(True)
out_scale = out_seq.abs().max().item()
rel = (out_seq - out_bat).abs().max().item() / out_scale
ok = rel < 1e-2
print(f"[{'PASS' if ok else 'FAIL'}] batched vs sequential actions (3 cams, 1 masked) "
      f"rel = {rel:.2e} (scale {out_scale:.2f})")
ok = ok and ok_emb

# Speed in bf16 (deployment-like). MPS if present, else CUDA, else CPU;
# override with VLA_TEST_DEVICE=cpu|cuda|mps.
def pick_device():
    forced = os.environ.get("VLA_TEST_DEVICE")
    if forced:
        return forced
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = pick_device()
synchronize = {
    "mps": lambda: torch.mps.synchronize(),
    "cuda": lambda: torch.cuda.synchronize(),
}.get(device, lambda: None)
policy.to(device)
policy.model = policy.model.to(torch.bfloat16)
batch_d = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
noise_d = noise.to(device)


def bench(batched: bool, iters=8):
    policy.config.batch_vision_encoder = batched
    with torch.no_grad():
        for _ in range(3):
            policy.reset(); policy.predict_action_chunk(dict(batch_d), noise=noise_d.clone())
        synchronize()
        ts = []
        for _ in range(iters):
            policy.reset()
            t0 = time.perf_counter()
            policy.predict_action_chunk(dict(batch_d), noise=noise_d.clone())
            synchronize()
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


t_seq = bench(False)
t_bat = bench(True)
print(f"{device} bf16, 3 cams: sequential {t_seq * 1e3:.1f} ms -> batched {t_bat * 1e3:.1f} ms "
      f"({t_seq / t_bat:.2f}x)")
sys.exit(0 if ok else 1)
