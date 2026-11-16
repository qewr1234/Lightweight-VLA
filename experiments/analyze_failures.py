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

"""Why episodes fail: sort recorded episodes into failure categories, and show what a setting
loses against a baseline on the same episodes.

A success rate says how often a setting fails, not how. This script reads the per-step records
that eval_libero.py writes with --trace-dir and answers three questions:

1. How does each setting fail? Every episode is put in one category from where the gripper and
   the target object went: never reached the target, reached it but never lifted it, pushed it
   away, lifted and dropped it, still holding it when time ran out, or put it down at the goal
   outside the success tolerance.
2. What does a setting lose? Episode i starts from the same initial state in every setting, so
   the episodes the baseline solves and another setting does not are exactly the cost of that
   setting. They are listed with the way each one failed.
3. How far off the plan is the robot when a stale chunk takes over? The drift recorded at
   every replan is reported in centimetres of commanded motion.

    python experiments/eval_libero.py --policy-path lerobot/smolvla_libero --suites libero_spatial \\
        --episodes 20 --n-envs 5 --async-envs --n-action-steps 10 --stale-align skip \\
        --settings 512:off:0 512:off:10 512:off:10:comp --trace-dir runs/traces --out runs/traced.json
    python experiments/analyze_failures.py runs/traces --base "512px stale=0"

The categories assume a pick-and-place task whose first object of interest is the thing to
move and whose last is where it goes (all of LIBERO-Spatial and LIBERO-Object). They are a
first automatic pass from positions alone: check the listed episodes against pictures
(inspect_libero_task.py) before building an argument on them.

Needs only the record files (numpy; no torch, no simulator).
"""

import argparse
from pathlib import Path

import numpy as np

POSITION_SCALE_M = 0.05  # robosuite OSC_POSE: an action of 1.0 asks for 5 cm of motion per control step
LIFT_M = 0.03  # the target counts as lifted once it is this far above where it started
REACH_M = 0.07  # gripper within this distance of the target's centre: it got there (a bowl is held by its rim)
PUSH_M = 0.03  # the target moved this far sideways without being lifted: it was pushed
NEAR_GOAL_M = 0.10  # the target ended this close to the goal: it was put down there

CATEGORIES = (
    "success",
    "never reached the target",
    "reached, grasp failed",
    "pushed the target away",
    "dropped on the way",
    "still holding at timeout",
    "placed off-centre",
    "failed (objects not recorded)",
)


def load(trace_dir: Path) -> dict[str, dict[tuple[str, int], dict]]:
    """label -> (suite, task id) -> arrays of that setting and task."""
    settings: dict[str, dict] = {}
    for path in sorted(trace_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            record = {key: data[key] for key in data.files}
        settings.setdefault(str(record["label"]), {})[(str(record["suite"]), int(record["task_id"]))] = record
    return settings


def episode_features(record: dict, episode: int) -> dict | None:
    """Where the gripper and the target went in one episode, or None without object positions."""
    steps = int(record["length"][episode])
    objects = record["objects"][episode, : steps + 1]
    if steps == 0 or objects.shape[1] < 6 or np.isnan(objects).any():
        return None
    eef = record["eef"][episode, : steps + 1]
    grip = record["action"][episode, :steps, -1]
    target, goal = objects[:, :3], objects[:, -3:]
    height = target[:, 2] - target[0, 2]
    above = np.flatnonzero(height >= LIFT_M)
    lifted_at = int(above[0]) if len(above) else None
    before = slice(0, lifted_at + 1) if lifted_at is not None else slice(None)
    reach = np.linalg.norm(eef - target, axis=1)
    closing = (grip > 0).astype(int)
    return {
        "steps": steps,
        "lifted_at": lifted_at,
        "reach_min": float(reach[before].min()),
        "pushed": float(np.linalg.norm(target[before, :2] - target[0, :2], axis=1).max()),
        "final_xy": float(np.linalg.norm(target[-1, :2] - goal[-1, :2])),
        "final_dz": float(target[-1, 2] - goal[-1, 2]),
        "holding": bool(grip[-1] > 0 and reach[-1] <= REACH_M),
        "closes": int(closing[0] + (np.diff(closing) == 1).sum()),
    }


def categorize(success: bool, features: dict | None) -> str:
    if success:
        return "success"
    if features is None:
        return "failed (objects not recorded)"
    if features["lifted_at"] is not None:
        if features["holding"]:
            return "still holding at timeout"
        if features["final_xy"] < NEAR_GOAL_M:
            return "placed off-centre"
        return "dropped on the way"
    if features["pushed"] >= PUSH_M:
        return "pushed the target away"
    if features["reach_min"] <= REACH_M:
        return "reached, grasp failed"
    return "never reached the target"


def drift_cm(record: dict, episode: int) -> np.ndarray:
    """Size of the position drift at every replan of one episode, in cm of commanded motion."""
    drift = record["drift"][episode, : int(record["length"][episode]), :3]
    return np.linalg.norm(drift[~np.isnan(drift).any(axis=1)], axis=1) * POSITION_SCALE_M * 100


def episodes_of(setting: dict[tuple[str, int], dict]) -> dict[tuple[str, int, int], dict]:
    """(suite, task, episode) -> outcome, category, features and drift of every recorded episode."""
    out = {}
    for (suite, task_id), record in sorted(setting.items()):
        for episode in range(len(record["success"])):
            features = episode_features(record, episode)
            success = bool(record["success"][episode])
            out[(suite, task_id, episode)] = {
                "success": success, "features": features, "category": categorize(success, features),
                "drift_cm": drift_cm(record, episode),
                "age_steps": record["age"][episode, : int(record["length"][episode])],
            }
    return out


def category_table(names: list[str], episodes: dict[str, dict]) -> str:
    width = max(len(c) for c in CATEGORIES)
    columns = [max(len(name), 12) for name in names]
    lines = [f"{'':<{width}} " + " ".join(f"{name:>{w}}" for name, w in zip(names, columns))]
    for category in CATEGORIES:
        counts = [sum(e["category"] == category for e in episodes[name].values()) for name in names]
        if not any(counts):
            continue
        cells = [f"{c:>4} {c / max(len(episodes[name]), 1):>5.0%}".rjust(w)
                 for c, name, w in zip(counts, names, columns)]
        lines.append(f"{category:<{width}} " + " ".join(cells))
    lines.append(f"{'episodes':<{width}} " + " ".join(f"{len(episodes[name]):>{w}}" for name, w in zip(names, columns)))
    return "\n".join(lines)


def loss_report(base_name: str, other_name: str, base: dict, other: dict, limit: int) -> str:
    shared = sorted(set(base) & set(other))
    lost = [k for k in shared if base[k]["success"] and not other[k]["success"]]
    gained = [k for k in shared if not base[k]["success"] and other[k]["success"]]
    both = sum(base[k]["success"] and other[k]["success"] for k in shared)
    lines = [f"'{other_name}' against '{base_name}' on the same {len(shared)} episodes: "
             f"{len(lost)} lost, {len(gained)} gained, {both} solved by both, "
             f"{len(shared) - both - len(lost) - len(gained)} by neither"]
    if lost:
        lines.append("  how the lost episodes failed:")
        for category in CATEGORIES[1:]:
            count = sum(other[k]["category"] == category for k in lost)
            if count:
                lines.append(f"    {category:<32} {count:>4}  {count / len(lost):>4.0%}")
    lifted = [(base[k]["features"]["lifted_at"], other[k]["features"]["lifted_at"]) for k in shared
              if base[k]["features"] and other[k]["features"]
              and base[k]["features"]["lifted_at"] is not None and other[k]["features"]["lifted_at"] is not None]
    if lifted:
        delta = np.median([b - a for a, b in lifted])
        lines.append(f"  first lift where both lifted the target ({len(lifted)} episodes): "
                     f"median {delta:+.0f} steps later than the baseline")
    if lost:
        lines.append(f"  lost episodes (suite, task, episode -> category){'' if len(lost) <= limit else f', first {limit}'}:")
        for suite, task_id, episode in lost[:limit]:
            lines.append(f"    {suite} {task_id} {episode:>3} -> {other[(suite, task_id, episode)]['category']}")
    return "\n".join(lines)


def drift_report(names: list[str], episodes: dict[str, dict]) -> str:
    width = max(len(name) for name in names)
    lines = [f"{'':<{width}}  replans  median     p90     max   >1cm   >3cm   episode max: solved / failed"]
    for name in names:
        every = [e["drift_cm"] for e in episodes[name].values() if len(e["drift_cm"])]
        if not every:
            lines.append(f"{name:<{width}}  no stale chunk was recorded")
            continue
        flat = np.concatenate(every)
        solved = [e["drift_cm"].max() for e in episodes[name].values() if len(e["drift_cm"]) and e["success"]]
        failed = [e["drift_cm"].max() for e in episodes[name].values() if len(e["drift_cm"]) and not e["success"]]
        per_episode = " / ".join("   -  " if not v else f"{np.median(v):>5.1f}" for v in (solved, failed))
        lines.append(f"{name:<{width}}  {len(flat):>7}  {np.median(flat):>6.2f}  {np.percentile(flat, 90):>6.2f}  "
                     f"{flat.max():>6.2f}  {np.mean(flat > 1):>5.0%}  {np.mean(flat > 3):>5.0%}   {per_episode} cm")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path, help="directory given to eval_libero.py --trace-dir")
    parser.add_argument("--base", default=None, help="label of the baseline setting (default: the best one)")
    parser.add_argument("--list", type=int, default=40, help="how many lost episodes to list per setting")
    args = parser.parse_args()

    settings = load(args.trace_dir)
    if not settings:
        raise SystemExit(f"no records in {args.trace_dir}. Run eval_libero.py with --trace-dir first.")
    episodes = {name: episodes_of(setting) for name, setting in settings.items()}
    rate = {name: np.mean([e["success"] for e in eps.values()]) for name, eps in episodes.items()}
    if args.base is not None and args.base not in settings:
        raise SystemExit(f"--base {args.base!r} is not among the recorded settings:\n  " + "\n  ".join(settings))
    base = args.base or max(rate, key=rate.get)
    names = [base] + [name for name in settings if name != base]

    print("How each setting ends (episodes, share):\n")
    print(category_table(names, episodes))
    for name in names[1:]:
        print("\n" + loss_report(base, name, episodes[base], episodes[name], args.list))
    print("\nPosition drift when a chunk planned from an old observation takes over "
          "(cm of commanded motion; one row per setting):\n")
    print(drift_report(names, episodes))
    ages = {name: np.concatenate([e["age_steps"] for e in eps.values()]) for name, eps in episodes.items()}
    hz = float(next(iter(next(iter(settings.values())).values()))["control_hz"])
    print("\nMean age of the observation behind an executed action: "
          + ", ".join(f"{name}: {np.nanmean(a) * 1e3 / hz:.0f} ms" if np.isfinite(a).any() else f"{name}: -"
                      for name, a in ages.items()))
    print(f"\nThresholds: lifted = {LIFT_M * 100:.0f} cm above the start, reached = gripper within "
          f"{REACH_M * 100:.0f} cm of the target, pushed = {PUSH_M * 100:.0f} cm sideways without a lift, "
          f"at the goal = within {NEAR_GOAL_M * 100:.0f} cm. Categories come from positions alone.")


if __name__ == "__main__":
    main()
