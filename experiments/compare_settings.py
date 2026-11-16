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

"""Compare the settings of one or more eval_libero.py reports against a baseline, paired.

eval_libero.py starts episode i of a task from initial state i with seed i in every setting
and every run with the same --seed, so two settings can be compared episode by episode. That
is much sharper than comparing two success rates: only the episodes where the settings
disagree carry information, and an exact test on those (McNemar) needs no normal
approximation. Reports written before per-episode outcomes were stored fall back to a
sign-flip test over per-task success rates, which is valid but weak with ten tasks.

    python experiments/compare_settings.py runs/spatial_mj323.json runs/spatial_mj323_cam.json \\
        --base "512px stale=10"

Needs only the report files (no torch, no simulator).
"""

import argparse
import itertools
import json
import random
from math import comb, sqrt
from pathlib import Path

COMPARABLE = ("policy", "seed", "episodes_per_task", "n_action_steps", "num_steps", "chunk_size")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(only_base: int, only_other: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = only_base + only_other
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(only_base, only_other) + 1))
    return min(1.0, 2 * tail / 2**n)


def sign_flip(differences: list[float], samples: int = 20000) -> float:
    """Two-sided sign-flip test on per-task differences: exact up to 16 tasks, sampled beyond."""
    m = len(differences)
    if m == 0:
        return 1.0
    observed = abs(sum(differences))
    if m <= 16:
        signs = itertools.product((1, -1), repeat=m)
        total = 2**m
    else:
        rng = random.Random(0)
        signs = ([rng.choice((1, -1)) for _ in range(m)] for _ in range(samples))
        total = samples
    hits = sum(abs(sum(s * d for s, d in zip(sign, differences))) >= observed - 1e-12 for sign in signs)
    return hits / total


def load_rows(paths: list[Path]) -> list[dict]:
    """One dict per (report, setting): label, source report, and outcomes per (suite, task)."""
    rows = []
    for path in paths:
        report = json.loads(path.read_text())
        for setting in report["settings"]:
            tasks = {}
            for suite, result in setting["suites"].items():
                for task_id, task in result["tasks"].items():
                    tasks[(suite, str(task_id))] = task
            rows.append({"label": setting["label"], "file": path.name, "report": report, "tasks": tasks})
    seen: dict[str, int] = {}
    for row in rows:
        seen[row["label"]] = seen.get(row["label"], 0) + 1
    for row in rows:
        row["name"] = row["label"] if seen[row["label"]] == 1 else f"{row['label']} @{row['file']}"
    return rows


def compare(base: dict, other: dict) -> dict:
    """Difference of `other` from `base` on the tasks both ran, with the sharpest valid test."""
    common = sorted(set(base["tasks"]) & set(other["tasks"]))
    if not common:
        return {"test": "no task in common"}
    k_base = sum(base["tasks"][t]["successes"] for t in common)
    k_other = sum(other["tasks"][t]["successes"] for t in common)
    n_base = sum(base["tasks"][t]["episodes"] for t in common)
    n_other = sum(other["tasks"][t]["episodes"] for t in common)
    out = {"diff": k_other / n_other - k_base / n_base, "tasks": len(common)}
    episodes = [(base["tasks"][t].get("episode_success"), other["tasks"][t].get("episode_success")) for t in common]
    if all(a is not None and b is not None and len(a) == len(b) for a, b in episodes):
        only_base = sum(x == 1 and y == 0 for a, b in episodes for x, y in zip(a, b))
        only_other = sum(x == 0 and y == 1 for a, b in episodes for x, y in zip(a, b))
        out.update(test="McNemar", only_base=only_base, only_other=only_other,
                   p=mcnemar_exact(only_base, only_other))
    else:
        differences = [other["tasks"][t]["success_rate"] - base["tasks"][t]["success_rate"] for t in common]
        out.update(test=f"sign-flip over {len(common)} tasks", p=sign_flip(differences))
    return out


def mismatches(base: dict, other: dict) -> list[str]:
    """Run-level differences that make an episode-by-episode comparison unsound."""
    a, b = base["report"], other["report"]
    notes = [f"{key}: {a.get(key)} vs {b.get(key)}" for key in COMPARABLE if a.get(key) != b.get(key)]
    mujoco = (a.get("versions", {}).get("mujoco"), b.get("versions", {}).get("mujoco"))
    if mujoco[0] != mujoco[1]:
        notes.append(f"mujoco: {mujoco[0]} vs {mujoco[1]}")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", type=Path, help="eval_libero.py report files")
    parser.add_argument("--base", default=None, help="label of the baseline setting (default: the first one)")
    args = parser.parse_args()

    rows = load_rows(args.reports)
    if not rows:
        raise SystemExit("no settings in these reports")
    if args.base is None:
        base = rows[0]
    else:
        matches = [r for r in rows if args.base in (r["name"], r["label"])]
        if len(matches) != 1:
            raise SystemExit(f"--base {args.base!r} matches {len(matches)} settings. Available:\n  "
                             + "\n  ".join(r["name"] for r in rows))
        base = matches[0]

    width = max(len(r["name"]) for r in rows) + 7
    print(f"{'setting':<{width}} {'n':>5} {'success':>8} {'95% CI':>10} {'vs base':>8}  paired test")
    warnings = []
    for row in rows:
        k = sum(t["successes"] for t in row["tasks"].values())
        n = sum(t["episodes"] for t in row["tasks"].values())
        lo, hi = wilson(k, n)
        line = f"{row['name'] + (' (base)' if row is base else ''):<{width}} {n:>5} {k / n:>8.1%} {f'[{lo:.0%}-{hi:.0%}]':>10}"
        if row is not base:
            result = compare(base, row)
            if "diff" in result:
                detail = (f"{result['only_base']} lost, {result['only_other']} gained, "
                          if result["test"] == "McNemar" else "")
                line += f" {result['diff'] * 100:>+7.1f}p  {result['test']}: {detail}p={result['p']:.4f}"
            else:
                line += f" {'':>8}  {result['test']}"
            notes = mismatches(base, row)
            if notes:
                warnings.append(f"{row['name']}: " + "; ".join(notes))
        print(line)
    if warnings:
        print("\nWARNING: these runs differ from the baseline run, so the episodes are not the same ones:")
        for warning in warnings:
            print(f"  {warning}")
    print("\n'lost' / 'gained': episodes the baseline solved and this setting did not / the reverse. "
          "p is two-sided and not corrected for the number of settings compared.")


if __name__ == "__main__":
    main()
