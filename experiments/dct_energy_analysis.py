"""DCT energy-compaction experiment on real SO-100 action chunks.

Question: if SmolVLA's flow matching denoised K DCT coefficients instead of 50 raw
action steps (suffix tokens 50 -> K), how small can K be before reconstruction error
becomes task-relevant?

Method: extract 50-step windows (chunk_size=50 @ 30fps = 1.67s, exactly what the model
generates), orthonormal DCT-II along time per action dim, measure retained energy and
reconstruction error when keeping only the K lowest-frequency coefficients.
"""

import json

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

CHUNK = 50
STRIDE = 25  # overlapping windows for more samples
K_LIST = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 50]
DIM_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Resolved through the HF cache at runtime (downloads once if not already cached).
DATASETS = {
    "pickplace": "lerobot/svla_so100_pickplace",
    "stacking": "lerobot/svla_so100_stacking",
}
PARQUET = "data/chunk-000/file-000.parquet"


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (n x n): X = D @ x."""
    k = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    D = np.cos(np.pi * (2 * t + 1) * k / (2 * n))
    D[0] *= np.sqrt(1.0 / n)
    D[1:] *= np.sqrt(2.0 / n)
    return D


def load_chunks(path: str) -> np.ndarray:
    tbl = pq.read_table(path, columns=["action", "episode_index"])
    actions = np.stack(tbl["action"].to_pylist()).astype(np.float64)  # (N, 6)
    episodes = np.asarray(tbl["episode_index"])
    chunks = []
    for ep in np.unique(episodes):
        a = actions[episodes == ep]
        for s in range(0, len(a) - CHUNK + 1, STRIDE):
            chunks.append(a[s : s + CHUNK])
    return np.stack(chunks)  # (C, 50, 6)


def analyze(name: str, chunks: np.ndarray) -> dict:
    C, T, D_ = chunks.shape
    # Normalize per-dim with dataset stats (what the model sees under MEAN_STD).
    mean = chunks.reshape(-1, D_).mean(0)
    std = chunks.reshape(-1, D_).std(0) + 1e-8
    z = (chunks - mean) / std

    D = dct_matrix(T)
    coeff = np.einsum("kt,ctd->ckd", D, z)  # (C, 50, 6)

    # Energy spectrum per dim, DC(0) excluded for the compaction claim (DC = chunk mean,
    # always kept; AC is where "high-frequency removal" actually operates).
    energy = coeff**2
    ac_energy = energy[:, 1:, :]  # (C, 49, 6)
    ac_total = ac_energy.sum(axis=1)  # (C, 6)

    result = {"n_chunks": C, "mean": mean, "std": std}

    # Per-dim: cumulative AC energy fraction, POOLED over chunks (energy-weighted).
    # A plain per-chunk average is misleading: near-static chunks have only sensor noise
    # as AC content (flat spectrum) and would drag the average toward "need everything",
    # even though their absolute error is negligible.
    pooled = ac_energy.sum(axis=0)  # (49, 6)
    pooled_cum = np.cumsum(pooled, axis=0) / (pooled.sum(axis=0, keepdims=True) + 1e-12)
    k95 = [int(np.searchsorted(pooled_cum[:, d], 0.95) + 1) for d in range(D_)]
    k99 = [int(np.searchsorted(pooled_cum[:, d], 0.99) + 1) for d in range(D_)]
    result["k95_ac"], result["k99_ac"] = k95, k99
    cum = pooled_cum[None]  # keep downstream row["ac_energy"] shape-compatible

    # Reconstruction error vs K (keep K lowest frequencies incl. DC, all dims equally
    # -> matches "K suffix tokens, each carrying all dims' coefficients at one frequency")
    rows = []
    for K in K_LIST:
        kept = coeff.copy()
        kept[:, K:, :] = 0.0
        # Inverse of orthonormal DCT-II: x[t] = sum_k D[k, t] * X[k]
        recon = np.einsum("kt,ckd->ctd", D, kept)
        err_z = np.abs(recon - z)  # normalized units (sigma)
        err_raw = err_z * std  # raw units (deg / gripper units)
        row = {
            "K": K,
            "ac_energy": float(np.mean(cum[:, K - 2, :])) if K >= 2 else 0.0,
            "rms_z": float(np.sqrt((err_z**2).mean())),
            "max_z": float(err_z.max()),
            "p99_raw_per_dim": err_raw.reshape(-1, D_).__abs__(),
        }
        row["p99_raw"] = [float(np.quantile(err_raw[:, :, d], 0.99)) for d in range(D_)]
        row["max_raw"] = [float(err_raw[:, :, d].max()) for d in range(D_)]
        del row["p99_raw_per_dim"]
        rows.append(row)
    result["rows"] = rows

    # Mean AC spectrum shape (for reporting): fraction of AC energy in bands
    spec = ac_energy.mean(axis=0)  # (49, 6)
    spec_frac = spec / spec.sum(axis=0, keepdims=True)
    bands = {
        "f1-4 (<=1.2Hz)": spec_frac[0:4].sum(axis=0),
        "f5-8 (<=2.4Hz)": spec_frac[4:8].sum(axis=0),
        "f9-16 (<=4.8Hz)": spec_frac[8:16].sum(axis=0),
        "f17-49 (>4.8Hz)": spec_frac[16:].sum(axis=0),
    }
    result["bands"] = {k: v.tolist() for k, v in bands.items()}
    return result


print("=" * 100)
print(f"DCT energy-compaction on real SO-100 action chunks (T={CHUNK} @ 30fps = 1.67s)")
print("=" * 100)

for name, repo_id in DATASETS.items():
    path = hf_hub_download(repo_id, PARQUET, repo_type="dataset")
    chunks = load_chunks(path)
    r = analyze(name, chunks)
    print(f"\n### {name}: {r['n_chunks']} chunks")
    print(f"  raw action std per dim: {np.round(r['std'], 2).tolist()}")

    print(f"\n  AC-energy compaction (DC excluded) - #coefficients needed per dim:")
    print(f"    {'dim':>14} | k(95%) | k(99%) | AC energy in bands")
    for d in range(6):
        b = [f"{r['bands'][k][d] * 100:4.1f}%" for k in r["bands"]]
        print(f"    {DIM_NAMES[d]:>14} | {r['k95_ac'][d]:6d} | {r['k99_ac'][d]:6d} | " + " / ".join(b))
    print(f"    (bands: {list(r['bands'].keys())})")

    print(f"\n  Reconstruction error keeping K lowest frequencies (ALL dims, incl. DC):")
    print(f"    {'K':>3} | {'RMS(sigma)':>10} | {'max(sigma)':>10} | p99 raw err per dim [deg, gripper]")
    for row in r["rows"]:
        p99 = ", ".join(f"{v:5.2f}" for v in row["p99_raw"])
        print(f"    {row['K']:>3} | {row['rms_z']:10.4f} | {row['max_z']:10.3f} | [{p99}]")

    print(f"\n  Gripper worst-case (max) raw error vs K:")
    for row in r["rows"]:
        if row["K"] in (4, 8, 12, 16, 24):
            print(f"    K={row['K']:>2}: joints max {max(row['max_raw'][:5]):6.2f} deg | gripper max {row['max_raw'][5]:6.2f}")
