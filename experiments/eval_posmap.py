"""posmap on the DEPLOYED checkpoint: does it survive when the model was finetuned at 384px?

runs/baseline/checkpoint.pt was trained with the fork defaults => 384px => the network's
native token grid is 6x6, NOT the 8x8 of zero-shot smolvla_base. If posmap only helps by
restoring an 8x8 layout the checkpoint never saw, it is a zero-shot-only artifact.

Metric: open-loop MAE vs GROUND TRUTH actions, in degrees (the repo's 5.49 convention),
on the held-out episodes. Fixed seed per config so the noise draws pair exactly.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch

EXP = str(Path(__file__).resolve().parent)
sys.path.insert(0, EXP)
sys.path.insert(0, str(Path(EXP).parent))
import os; os.environ.setdefault("HF_HUB_OFFLINE", "1")

from _local_smolvla import use_local_smolvla
use_local_smolvla()

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from train_ab import CAMERAS, FPS, REPO, build_policy
from eval_ab import EVAL_EPISODES, evaluate, measure_latency, select_windows

DEV = "mps"
CKPT = f"{EXP}/runs/baseline/checkpoint.pt"
SEEDS = [0, 1, 2]
STEPS = 4

policy = build_policy(DEV)
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
missing, unexpected = policy.load_state_dict(ck["state_dict"], strict=False)
assert not [k for k in missing if "lm_head" not in k], missing[:3]
policy.eval()
print(f"loaded {CKPT}: trained {ck['steps']} steps, variant={ck['variant']}", flush=True)
print(f"config at load: resize={policy.config.resize_imgs_with_padding} "
      f"posmap={policy.config.posmap_ref_resolution}", flush=True)

ds = LeRobotDataset(REPO, delta_timestamps={"action": [i / FPS for i in range(50)]},
                    episodes=EVAL_EPISODES)
windows = select_windows(ds)
task = ds[0]["task"]
if not task.endswith("\n"):
    task += "\n"
tok = policy.model.vlm_with_expert.processor.tokenizer
enc = tok(task, max_length=policy.config.tokenizer_max_length, padding="max_length",
          truncation=True, return_tensors="pt")
lt, lm = enc["input_ids"].to(DEV), enc["attention_mask"].bool().to(DEV)
print(f"{len(windows)} held-out windows, task={task!r}", flush=True)

CONFIGS = [
    (384, None), (384, 384),          # baseline + no-op self-check
    (320, None), (320, 384), (320, 512),
    (256, None), (256, 384),
    (448, None), (448, 384),
    (512, None), (512, 384),
]

out = {}
for res, ref in CONFIGS:
    policy.config.resize_imgs_with_padding = (res, res)
    policy.config.posmap_ref_resolution = ref
    maes, per_dim = [], []
    for s in SEEDS:
        torch.manual_seed(s)
        r = evaluate(policy, ds, windows, lt, lm, ds.meta.stats, DEV, STEPS)
        maes.append(r["mae_overall"]); per_dim.append(r["mae_per_dim"])
    lat = measure_latency(policy, ds, windows, lt, lm, ds.meta.stats, DEV, STEPS) * 1e3
    key = f"{res}_{'plain' if ref is None else f'posmap{ref}'}"
    out[key] = {"mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                "mae_seeds": maes, "per_dim": np.mean(per_dim, axis=0).tolist(),
                "grip": float(np.mean(per_dim, axis=0)[5]), "latency_ms": lat}
    print(f"{key:16s} MAE {np.mean(maes):6.3f} deg (+-{np.std(maes):.3f})  "
          f"grip {out[key]['grip']:5.3f}  {lat:6.1f} ms", flush=True)
    json.dump(out, open(f"{EXP}/runs/eval_posmap.json", "w"), indent=1)

b = out["384_plain"]["mae_mean"]
print("\n=== vs 384_plain (배포 기본값) ===", flush=True)
for k, v in out.items():
    d = v["mae_mean"] - b
    print(f"{k:16s} {v['mae_mean']:6.3f}  {d:+6.3f}  {v['latency_ms'] - out['384_plain']['latency_ms']:+7.1f} ms  "
          f"{'WIN' if d < -0.05 else ('same' if abs(d) <= 0.05 else 'lose')}", flush=True)
noop = abs(out["384_posmap384"]["mae_mean"] - b)
print(f"\nno-op self-check (384+posmap384 vs 384 plain): |d| = {noop:.6f}  "
      f"{'PASS' if noop < 1e-6 else 'FAIL'}", flush=True)
