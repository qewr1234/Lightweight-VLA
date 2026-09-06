"""Equivalence + smoke tests for the lightweight SmolVLA copy.

Runs against the venv's installed lerobot, with the modified modules injected from
this repository via _local_smolvla. Random weights (load_vlm_weights=False),
CPU, fixed seeds.
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from _local_smolvla import use_local_smolvla

use_local_smolvla()

import draccus
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
CAMERAS = 2
LANG_LEN = 12


def make_config(**overrides):
    input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
    for i in range(CAMERAS):
        input_features[f"observation.images.cam{i}"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 480, 640)
        )
    output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,))}
    kwargs = dict(
        input_features=input_features,
        output_features=output_features,
        load_vlm_weights=False,
    )
    kwargs.update(overrides)
    return SmolVLAConfig(**kwargs)


def make_policy(seed=0, **overrides):
    torch.manual_seed(seed)
    policy = SmolVLAPolicy(make_config(**overrides))
    policy.eval()
    return policy


def make_batch(seed=1, masked_camera=False):
    g = torch.Generator().manual_seed(seed)
    batch = {
        OBS_STATE: torch.rand(1, STATE_DIM, generator=g),
        OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, LANG_LEN), generator=g),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, LANG_LEN, dtype=torch.bool),
    }
    for i in range(CAMERAS):
        batch[f"observation.images.cam{i}"] = torch.rand(1, 3, 480, 640, generator=g)
    if masked_camera:
        batch["observation.images.cam1_padding_mask"] = torch.zeros(1, dtype=torch.bool)
    return batch


def fixed_noise(policy, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(
        1, policy.config.chunk_size, policy.config.max_action_dim, generator=g
    )


def run(policy, batch, noise):
    policy.reset()
    with torch.no_grad():
        return policy.predict_action_chunk(dict(batch), noise=noise.clone())


def set_attention(policy, impl):
    policy.config.attention_implementation = impl
    policy.model.vlm_with_expert.attention_implementation = impl


def set_kv_cache(policy, enabled):
    policy.config.cache_expert_kv_projections = enabled
    policy.model.vlm_with_expert.cache_expert_kv_projections = enabled


results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


print("=" * 70)
print("Building policy (random weights, fp32, CPU) ...")
policy = make_policy(seed=0)
batch = make_batch()
noise = fixed_noise(policy)

# --- Test 1a: attention kernels agree on identical inputs -------------------
vwe = policy.model.vlm_with_expert
g = torch.Generator().manual_seed(3)
B, L, H, KV, D = 1, 100, vwe.num_attention_heads, vwe.num_key_value_heads, 64
q = torch.randn(B, L, H, D, generator=g)
k = torch.randn(B, L, KV, D, generator=g)
v = torch.randn(B, L, KV, D, generator=g)
mask = torch.rand(B, L, L, generator=g) > 0.3
mask |= torch.eye(L, dtype=torch.bool)[None]  # no fully-masked rows
a_eager = vwe.eager_attention_forward(mask, B, D, q, k, v)
a_sdpa = vwe.sdpa_attention_forward(mask, B, D, q, k, v)
kdiff = (a_eager - a_sdpa).abs().max().item()
check("attention kernel: eager vs sdpa max|diff|", kdiff < 1e-5, f"= {kdiff:.2e}")

# --- Test 1b: end-to-end, both within FP noise of an fp64 reference ---------
set_kv_cache(policy, False)
set_attention(policy, "eager")
out_eager = run(policy, batch, noise)
set_attention(policy, "sdpa")
out_sdpa = run(policy, batch, noise)
diff = (out_eager - out_sdpa).abs().max().item()

policy64 = make_policy(seed=0)  # same weights (same seed), cast to fp64
policy64.model = policy64.model.to(torch.float64)
set_attention(policy64, "eager")
set_kv_cache(policy64, False)
ref64 = run(policy64, batch, noise.double()).float()
err_eager = (out_eager - ref64).abs().max().item()
err_sdpa = (out_sdpa - ref64).abs().max().item()
ratio = err_sdpa / max(err_eager, 1e-12)
check(
    "end-to-end: sdpa no farther from fp64 truth than eager (x3)",
    ratio < 3.0,
    f"|eager-ref64| = {err_eager:.2e}, |sdpa-ref64| = {err_sdpa:.2e}, eager-vs-sdpa = {diff:.2e}",
)

# --- Test 2: KV projection cache on/off (exact) -----------------------------
set_attention(policy, "sdpa")
set_kv_cache(policy, False)
out_nocache = run(policy, batch, noise)
set_kv_cache(policy, True)
out_cache = run(policy, batch, noise)
diff = (out_nocache - out_cache).abs().max().item()
check("kv-projection cache on vs off max|diff|", diff < 1e-6, f"= {diff:.2e}")

# --- Test 3: masked camera -> no NaN, eager/sdpa agree ----------------------
masked_batch = make_batch(masked_camera=True)
set_kv_cache(policy, True)
set_attention(policy, "sdpa")
out_masked_sdpa = run(policy, masked_batch, noise)
set_attention(policy, "eager")
set_kv_cache(policy, False)
out_masked_eager = run(policy, masked_batch, noise)
check("masked camera: sdpa output finite", torch.isfinite(out_masked_sdpa).all().item())
set_attention(policy64, "eager")
ref64_masked = run(policy64, masked_batch, noise.double()).float()
err_e = (out_masked_eager - ref64_masked).abs().max().item()
err_s = (out_masked_sdpa - ref64_masked).abs().max().item()
check(
    "masked camera: sdpa within FP noise of fp64 truth (x3 of eager)",
    err_s / max(err_e, 1e-12) < 3.0,
    f"|eager-ref64| = {err_e:.2e}, |sdpa-ref64| = {err_s:.2e}",
)

# --- Test 4: lm_head strip is lossless and saves params ---------------------
policy_full = make_policy(seed=0, strip_lm_head=False)
n_full = sum(p.numel() for p in policy_full.parameters())
n_stripped = sum(p.numel() for p in policy.parameters())
saved = n_full - n_stripped
check("lm_head strip saves ~47M params", 40e6 < saved < 55e6, f"saved = {saved / 1e6:.1f}M")
set_attention(policy_full, "sdpa")
set_kv_cache(policy_full, True)
out_full = run(policy_full, batch, noise)
set_attention(policy, "sdpa")
set_kv_cache(policy, True)
out_stripped = run(policy, batch, noise)
diff = (out_full - out_stripped).abs().max().item()
check("lm_head strip: identical actions", diff == 0.0, f"max|diff| = {diff:.2e}")

# --- Test 5: training forward path unbroken (loss finite, cache inactive) ---
train_batch = dict(batch)
train_batch["action"] = torch.rand(1, policy.config.chunk_size, STATE_DIM)
policy.train()
loss, loss_dict = policy.forward(train_batch)
policy.eval()
check("training forward loss finite", torch.isfinite(loss).item(), f"loss = {loss.item():.4f}")

# --- Test 6: full bf16 inference runs and stays close -----------------------
policy_bf16 = make_policy(seed=0)
set_attention(policy_bf16, "sdpa")
set_kv_cache(policy_bf16, True)
policy_bf16.model = policy_bf16.model.to(torch.bfloat16)
out_bf16 = run(policy_bf16, batch, noise)
check("bf16 inference finite", torch.isfinite(out_bf16).all().item())
rel = (out_bf16.float() - out_stripped).abs().max().item()
scale = out_stripped.abs().max().item()
check("bf16 vs fp32 sane (informational)", rel < 0.2 * max(scale, 1.0), f"max|diff| = {rel:.4f} (out scale ~{scale:.2f})")

# --- Test 6b: bf16 TRAINING forward runs (review fix: fp32 hardcode removed) -
train_batch_bf16 = dict(batch)
train_batch_bf16["action"] = torch.rand(1, policy_bf16.config.chunk_size, STATE_DIM)
policy_bf16.train()
loss_bf16, _ = policy_bf16.forward(train_batch_bf16)
policy_bf16.eval()
check("bf16 training forward runs, loss finite", torch.isfinite(loss_bf16).item(), f"loss = {loss_bf16.item():.4f}")

# --- Test 8: save -> config.json has no fork-only keys, stock-parseable ------
import json as _json
import subprocess
import tempfile

with tempfile.TemporaryDirectory() as td:
    policy.save_pretrained(td)
    with open(f"{td}/config.json") as f:
        cfg_data = _json.load(f)
    fork_keys = set(SmolVLAConfig._FORK_ONLY_FIELDS)
    leaked = fork_keys & set(cfg_data)
    check("saved config.json has no fork-only keys", not leaked, f"leaked = {leaked or 'none'}")

    # Parse the saved config with the STOCK SmolVLAConfig class in a clean subprocess.
    # The stock class is loaded by file path to dodge lerobot.policies.__init__, which is
    # broken in this venv by an unrelated GROOT/transformers-5.x issue; the class body
    # itself (the thing draccus validates fields against) is byte-identical stock code.
    stock_test = f'''
import importlib.util, sys, types
from pathlib import Path
import draccus
import lerobot
# Path-only stub so importing rtc/smolvla submodules skips the package __init__ that is
# broken in this venv by an unrelated GROOT/transformers-5.x issue. The SmolVLAConfig
# class being parsed below is byte-identical stock code.
stub = types.ModuleType("lerobot.policies")
stub.__path__ = [str(Path(lerobot.__file__).resolve().parent / "policies")]
sys.modules["lerobot.policies"] = stub
stock_cfg = Path(lerobot.__file__).resolve().parent / "policies" / "smolvla" / "configuration_smolvla.py"
spec = importlib.util.spec_from_file_location(
    "lerobot.policies.smolvla.configuration_smolvla", str(stock_cfg),
)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
# Mimic stock PreTrainedConfig.from_pretrained (configs/policies.py): it pops the
# "type" discriminator before parsing with the concrete subclass.
import json, tempfile
with open("{td}/config.json") as f:
    data = json.load(f)
data.pop("type")
with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tf:
    json.dump(data, tf)
    tmp = tf.name
with draccus.config_type("json"):
    cfg = draccus.parse(mod.SmolVLAConfig, tmp, args=[])
print("STOCK_PARSE_OK", cfg.num_steps, cfg.resize_imgs_with_padding)
'''
    r = subprocess.run(
        [sys.executable, "-c", stock_test],
        capture_output=True, text=True, timeout=120,
    )
    check(
        "stock lerobot parses saved config.json",
        "STOCK_PARSE_OK" in r.stdout,
        (r.stdout.strip().splitlines() or [r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "?"])[-1][:80],
    )

    # Roundtrip with the fork: reload and confirm fork defaults re-apply.
    reloaded = SmolVLAPolicy.from_pretrained(td)
    check(
        "fork reload: defaults re-apply (sdpa/cache/strip)",
        reloaded.config.attention_implementation == "sdpa"
        and reloaded.config.cache_expert_kv_projections
        and reloaded.config.strip_lm_head
        and reloaded.model.vlm_with_expert.vlm.lm_head is None,
    )
    del reloaded

# --- Test 9: stub compatibility (review fix: async_inference imports) --------
stub_test = f'''
import sys
sys.path.insert(0, r"{REPO_ROOT}")
from _local_smolvla import use_local_smolvla
use_local_smolvla()
from lerobot.policies import SmolVLAConfig, ACTConfig, DiffusionConfig, PI0Config, PI05Config, VQBeTConfig
import lerobot.policies.smolvla.modeling_smolvla as m
import lerobot
assert lerobot.policies.smolvla.modeling_smolvla is m
assert SmolVLAConfig.__module__ == "lerobot.policies.smolvla.configuration_smolvla"
assert m.__file__.startswith(r"{REPO_ROOT}"), m.__file__
print("STUB_OK")
'''
r = subprocess.run(
    [sys.executable, "-c", stub_test],
    capture_output=True, text=True, timeout=180,
)
check(
    "stub: from lerobot.policies import <configs> + attr chain",
    "STUB_OK" in r.stdout,
    (r.stderr.strip().splitlines()[-1][:100] if r.stderr.strip() and "STUB_OK" not in r.stdout else ""),
)

# --- Test 7: latency, baseline vs lightweight (fp32 CPU) --------------------
def bench(policy_b, batch_b, steps, res, attention, cache, iters=3):
    policy_b.config.num_steps = steps
    policy_b.config.resize_imgs_with_padding = (res, res)
    set_attention(policy_b, attention)
    set_kv_cache(policy_b, cache)
    run(policy_b, batch_b, fixed_noise(policy_b))  # warmup
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run(policy_b, batch_b, fixed_noise(policy_b))
        times.append(time.perf_counter() - t0)
    return min(times)


t_base = bench(policy, batch, steps=10, res=512, attention="eager", cache=False)
t_opt = bench(policy, batch, steps=4, res=384, attention="sdpa", cache=True)
check(
    "lightweight faster than baseline",
    t_opt < t_base,
    f"baseline {t_base * 1e3:.0f} ms -> optimized {t_opt * 1e3:.0f} ms ({t_base / t_opt:.2f}x, CPU fp32)",
)

# --- Test 10: draccus dump is fork-clean (covers train_config.json too) -----
# _save_pretrained was replaced by an encoder hook so that TrainPipelineConfig's nested
# dump - i.e. train_config.json, which lerobot-train --resume reads back - is stripped too.
import io

with draccus.config_type("json"):
    buf = io.StringIO()
    draccus.dump(make_config(), buf)
    dumped = _json.loads(buf.getvalue())
leaked_dump = set(SmolVLAConfig._FORK_ONLY_FIELDS) & set(dumped)
check(
    "draccus.dump has no fork-only keys, keeps `type`",
    not leaked_dump and dumped.get("type") == "smolvla",
    f"leaked = {leaked_dump or 'none'}, type = {dumped.get('type')}",
)

# --- Test 11: cameras of differing native size don't break batched vision ----
# torch.cat needs identical H,W; with resize_imgs_with_padding=None they can differ.
hetero_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
for _i, _res in enumerate([(256, 256), (320, 320)]):
    hetero_features[f"observation.images.cam{_i}"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, *_res)
    )
torch.manual_seed(0)
hetero_policy = SmolVLAPolicy(
    SmolVLAConfig(
        input_features=hetero_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,))},
        load_vlm_weights=False,
        resize_imgs_with_padding=None,
    )
)
hetero_policy.eval()
_g = torch.Generator().manual_seed(1)
hetero_batch = {
    OBS_STATE: torch.rand(1, STATE_DIM, generator=_g),
    OBS_LANGUAGE_TOKENS: torch.randint(10, 1000, (1, LANG_LEN), generator=_g),
    OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, LANG_LEN, dtype=torch.bool),
    "observation.images.cam0": torch.rand(1, 3, 256, 256, generator=_g),
    "observation.images.cam1": torch.rand(1, 3, 320, 320, generator=_g),
}
try:
    with torch.no_grad():
        hetero_out = hetero_policy.predict_action_chunk(dict(hetero_batch))
    check("mixed-resolution cameras (resize=None) run", hetero_out.shape[-1] == STATE_DIM, str(tuple(hetero_out.shape)))
except Exception as e:  # noqa: BLE001
    check("mixed-resolution cameras (resize=None) run", False, f"{type(e).__name__}: {str(e)[:80]}")
del hetero_policy

# --- Test 12: num_steps must be >= 1 (dt = -1/num_steps) --------------------
try:
    make_config(num_steps=0)
    check("num_steps=0 rejected", False, "accepted")
except ValueError as e:
    check("num_steps=0 rejected", True, str(e)[:60])

# --- Test 13: posmap is a no-op at its reference resolution, active below it -
# The remap must reproduce cumsum(pad_masks)-1 exactly when the token grid already equals
# the reference grid; otherwise it is silently perturbing the layout it claims to restore.
noise13 = fixed_noise(policy)
policy.config.resize_imgs_with_padding = (384, 384)
policy.config.posmap_ref_resolution = None
out_plain = run(policy, batch, noise13)
policy.config.posmap_ref_resolution = 384
out_noop = run(policy, batch, noise13)
check(
    "posmap at ref resolution is bit-exact no-op",
    torch.equal(out_plain, out_noop),
    f"max|diff| = {(out_plain - out_noop).abs().max().item():.2e}",
)
# And below it, the remap must actually change the positions (else the knob does nothing).
policy.config.resize_imgs_with_padding = (256, 256)
policy.config.posmap_ref_resolution = None
out_low_plain = run(policy, batch, noise13)
policy.config.posmap_ref_resolution = 384
out_low_posmap = run(policy, batch, noise13)
check(
    "posmap below ref resolution changes the output",
    not torch.equal(out_low_plain, out_low_posmap),
    f"max|diff| = {(out_low_plain - out_low_posmap).abs().max().item():.2e}",
)
policy.config.resize_imgs_with_padding = (384, 384)
policy.config.posmap_ref_resolution = None

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
