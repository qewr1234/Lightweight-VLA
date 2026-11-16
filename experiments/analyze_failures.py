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
   the objects went: never reached the target, reached it but never lifted it, pushed it away,
   moved some other object, lifted and dropped it, still holding it when time ran out, or put
   it down at the goal outside the success tolerance.
2. What does a setting lose? Episode i starts from the same initial state in every setting, so
   the episodes the baseline solves and another setting does not are exactly the cost of that
   setting. They are listed with the way each one failed, and --sheets writes a picture of
   each: the baseline's frames above the failing setting's.
3. How far off the plan is the robot when a stale chunk takes over? Reported twice: as the
   commanded drift (what the actions asked for) and as an estimate of the real offset, using
   how far the arm actually moves per unit of action in these records.

    python experiments/eval_libero.py --policy-path lerobot/smolvla_libero --suites libero_spatial \\
        --episodes 20 --n-envs 5 --async-envs --n-action-steps 10 --stale-align skip \\
        --settings 512:off:0 512:off:10 512:off:10:comp --trace-dir runs/traces --out runs/traced.json
    python experiments/analyze_failures.py runs/traces --base "512px stale=0" --sheets runs/sheets

The categories assume a pick-and-place task whose first object of interest is the thing to
move and whose last is where it goes (all of LIBERO-Spatial and LIBERO-Object). They are a
first automatic pass from positions alone: look at the pictures before building an argument
on them.

Needs only the record files (numpy, and Pillow for --sheets; no torch, no simulator).
"""

import argparse
from pathlib import Path

import numpy as np

COMMANDED_M = 0.05  # robosuite OSC_POSE: an action of 1.0 asks for 5 cm of motion per control step
LIFT_M = 0.03  # the target counts as lifted once it is this far above where it started
REACH_M = 0.07  # gripper within this distance of the target's centre: it got there (a bowl is held by its rim)
PUSH_M = 0.03  # the target moved this far sideways without being lifted: it was pushed
OTHER_M = 0.05  # another object moved this far: the robot handled the wrong thing
NEAR_GOAL_M = 0.10  # the target ended this close to the goal: it was put down there
EARLY_STEPS = 50  # the approach and first grasp; later replans of a failing episode are its aftermath
GAIN_WINDOW = 10  # steps over which motion is compared with the actions that asked for it

CATEGORIES = (
    "success",
    "never reached the target",
    "reached, grasp failed",
    "pushed the target away",
    "moved another object",
    "dropped on the way",
    "still holding at timeout",
    "placed off-centre",
    "failed (no pick-and-place record)",
)


def load(trace_dir: Path) -> dict[str, dict[tuple[str, int], dict]]:
    """label -> (suite, task id) -> arrays of that setting and task."""
    settings: dict[str, dict] = {}
    for path in sorted(trace_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            record = {key: data[key] for key in data.files}
        settings.setdefault(str(record["label"]), {})[(str(record["suite"]), int(record["task_id"]))] = record
    return settings


def interest_count(record: dict) -> int:
    """How many of the recorded objects are the task's objects of interest (they come first)."""
    return int(record["n_interest"]) if "n_interest" in record else record["objects"].shape[2] // 3


def episode_features(record: dict, episode: int) -> dict | None:
    """Where the gripper and the objects went in one episode, or None if that was not recorded."""
    steps = int(record["length"][episode])
    k = interest_count(record)
    objects = record["objects"][episode, : steps + 1].astype(np.float64)
    if steps == 0 or k < 2 or np.isnan(objects[:, : 3 * k]).any():
        return None
    eef = record["eef"][episode, : steps + 1].astype(np.float64)
    grip = record["action"][episode, :steps, -1]
    target, goal = objects[:, :3], objects[:, 3 * (k - 1) : 3 * k]
    height = target[:, 2] - target[0, 2]
    above = np.flatnonzero(height >= LIFT_M)
    lifted_at = int(above[0]) if len(above) else None
    before = slice(0, lifted_at + 1) if lifted_at is not None else slice(None)
    reach = np.linalg.norm(eef - target, axis=1)
    closing = (grip > 0).astype(int)
    others = objects[:, 3 * k :].reshape(steps + 1, -1, 3)
    other_moved, other_name = 0.0, None
    if others.shape[1] and not np.isnan(others).any():
        moved = np.linalg.norm(others - others[0], axis=2).max(axis=0)
        other_moved = float(moved.max())
        other_name = str(record["object_names"][k + int(moved.argmax())])
    return {
        "steps": steps,
        "lifted_at": lifted_at,
        "reach_min": float(reach[before].min()),
        "pushed": float(np.linalg.norm(target[before, :2] - target[0, :2], axis=1).max()),
        "final_xy": float(np.linalg.norm(target[-1, :2] - goal[-1, :2])),
        "final_dz": float(target[-1, 2] - goal[-1, 2]),
        "holding": bool(grip[-1] > 0 and reach[-1] <= REACH_M),
        "closes": int(closing[0] + (np.diff(closing) == 1).sum()),
        "other_moved": other_moved,
        "other_name": other_name,
    }


def categorize(success: bool, features: dict | None) -> str:
    if success:
        return "success"
    if features is None:
        return "failed (no pick-and-place record)"
    if features["lifted_at"] is not None:
        if features["holding"]:
            return "still holding at timeout"
        if features["final_xy"] < NEAR_GOAL_M:
            return "placed off-centre"
        return "dropped on the way"
    if features["pushed"] >= PUSH_M:
        return "pushed the target away"
    if features["other_moved"] >= OTHER_M:
        return "moved another object"
    if features["reach_min"] <= REACH_M:
        return "reached, grasp failed"
    return "never reached the target"


def fit_gain(settings: dict) -> tuple[np.ndarray, float]:
    """Metres the gripper really moves per unit of action per step, for x, y and z.

    The controller is asked for COMMANDED_M per unit and follows with a lag, so the arm covers
    less. Fitted through the origin on every GAIN_WINDOW-step stretch of every record; the
    second value is the share of the motion the fit explains.
    """
    moved, asked = [], []
    for setting in settings.values():
        for record in setting.values():
            for episode in range(len(record["success"])):
                steps = int(record["length"][episode])
                if steps <= GAIN_WINDOW:
                    continue
                eef = record["eef"][episode, : steps + 1].astype(np.float64)
                total = np.concatenate([np.zeros((1, 3)), np.cumsum(record["action"][episode, :steps, :3], axis=0)])
                moved.append(eef[GAIN_WINDOW:] - eef[:-GAIN_WINDOW])
                asked.append(total[GAIN_WINDOW:] - total[:-GAIN_WINDOW])
    if not moved:
        return np.full(3, COMMANDED_M), float("nan")
    moved, asked = np.concatenate(moved), np.concatenate(asked)
    gain = (moved * asked).sum(axis=0) / np.maximum((asked * asked).sum(axis=0), 1e-12)
    explained = 1 - ((moved - gain * asked) ** 2).sum() / max(((moved - moved.mean(axis=0)) ** 2).sum(), 1e-12)
    return gain, float(explained)


def replans(record: dict, episode: int, gain: np.ndarray) -> list[dict]:
    """Every point of an episode where a chunk planned from an old observation took over.

    commanded_cm  size of the drift in what the actions asked for
    offset_cm     estimate of how far the gripper really is from where the new plan assumes:
                  the motion measured since the plan's observation, minus the motion the plan
                  itself intended for that time (its skipped actions at the fitted gain)
    """
    steps = int(record["length"][episode])
    drift = record["drift"][episode, :steps, :3].astype(np.float64)
    eef = record["eef"][episode, : steps + 1].astype(np.float64)
    actions = record["action"][episode, :steps, :3].astype(np.float64)
    out = []
    for step in np.flatnonzero(~np.isnan(drift).any(axis=1)):
        age = record["age"][episode, step]
        if not np.isfinite(age) or age <= 0 or step - int(age) < 0:
            continue
        start = step - int(age)
        intended = actions[start:step].sum(axis=0) - drift[step]
        offset = (eef[step] - eef[start]) - gain * intended
        out.append({"step": int(step), "commanded_cm": float(np.linalg.norm(drift[step]) * COMMANDED_M * 100),
                    "offset_cm": float(np.linalg.norm(offset) * 100)})
    return out


def episodes_of(setting: dict[tuple[str, int], dict], gain: np.ndarray) -> dict[tuple[str, int, int], dict]:
    """(suite, task, episode) -> outcome, category, features and replans of every recorded episode."""
    out = {}
    for (suite, task_id), record in sorted(setting.items()):
        for episode in range(len(record["success"])):
            features = episode_features(record, episode)
            success = bool(record["success"][episode])
            out[(suite, task_id, episode)] = {
                "success": success, "features": features, "category": categorize(success, features),
                "replans": replans(record, episode, gain),
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


def lost_and_gained(base: dict, other: dict) -> tuple[list, list, list]:
    shared = sorted(set(base) & set(other))
    lost = [k for k in shared if base[k]["success"] and not other[k]["success"]]
    gained = [k for k in shared if not base[k]["success"] and other[k]["success"]]
    return shared, lost, gained


def loss_report(base_name: str, other_name: str, base: dict, other: dict, limit: int) -> str:
    shared, lost, gained = lost_and_gained(base, other)
    both = sum(base[k]["success"] and other[k]["success"] for k in shared)
    lines = [f"'{other_name}' against '{base_name}' on the same {len(shared)} episodes: "
             f"{len(lost)} lost, {len(gained)} gained, {both} solved by both, "
             f"{len(shared) - both - len(lost) - len(gained)} by neither"]
    if lost:
        lines.append("  how the lost episodes failed:")
        for category in CATEGORIES[1:]:
            count = sum(other[k]["category"] == category for k in lost)
            if count:
                lines.append(f"    {category:<34} {count:>4}  {count / len(lost):>4.0%}")
    lifted = [(base[k]["features"]["lifted_at"], other[k]["features"]["lifted_at"]) for k in shared
              if base[k]["features"] and other[k]["features"]
              and base[k]["features"]["lifted_at"] is not None and other[k]["features"]["lifted_at"] is not None]
    if lifted:
        delta = np.median([b - a for a, b in lifted])
        lines.append(f"  first lift where both lifted the target ({len(lifted)} episodes): "
                     f"median {delta:+.0f} steps later than the baseline")
    if lost:
        lines.append(f"  lost episodes (suite, task, episode -> category){'' if len(lost) <= limit else f', first {limit}'}:")
        for key in lost[:limit]:
            entry = other[key]
            note = ""
            if entry["category"] == "moved another object":
                note = f" ({entry['features']['other_name']}, {entry['features']['other_moved'] * 100:.0f} cm)"
            lines.append(f"    {key[0]} {key[1]} {key[2]:>3} -> {entry['category']}{note}")
    return "\n".join(lines)


def median(values) -> str:
    return "    -" if not len(values) else f"{np.median(values):>5.1f}"


def drift_report(names: list[str], episodes: dict[str, dict]) -> str:
    """One row per setting. 'early' keeps only the replans of the first EARLY_STEPS steps, where
    a failing episode has not yet failed: later ones mix in the confusion that follows."""
    width = max(len(name) for name in names)
    lines = [f"{'':<{width}}  replans   commanded cm      real offset cm    early real offset cm (median)",
             f"{'':<{width}}            median    p90     median    p90     solved   failed   >1cm   >3cm"]
    for name in names:
        every = [r for e in episodes[name].values() for r in e["replans"]]
        if not every:
            lines.append(f"{name:<{width}}  no stale chunk was recorded")
            continue
        commanded = np.array([r["commanded_cm"] for r in every])
        offset = np.array([r["offset_cm"] for r in every])
        early = {flag: np.array([r["offset_cm"] for e in episodes[name].values() if e["success"] == flag
                                 for r in e["replans"] if r["step"] <= EARLY_STEPS]) for flag in (True, False)}
        early_all = np.concatenate([early[True], early[False]])
        share = (lambda t: "    -" if not len(early_all) else f"{np.mean(early_all > t):>5.0%}")
        lines.append(f"{name:<{width}}  {len(every):>7}   {np.median(commanded):>6.1f} {np.percentile(commanded, 90):>6.1f}"
                     f"     {np.median(offset):>6.1f} {np.percentile(offset, 90):>6.1f}"
                     f"      {median(early[True])}    {median(early[False])}  {share(1)}  {share(3)}")
    return "\n".join(lines)


def write_sheets(out_dir: Path, base_name: str, other_name: str, settings: dict, keys: list, limit: int) -> int:
    """One picture per episode: the baseline's frames above the other setting's, same initial state."""
    from PIL import Image, ImageDraw

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for suite, task_id, episode in keys[:limit]:
        rows = []
        for name in (base_name, other_name):
            record = settings[name][(suite, task_id)]
            if "frames" not in record:
                return written
            steps, every = int(record["length"][episode]), int(record["frame_every"])
            kept = list(range(steps // every + 1))
            frames = [record["frames"][episode, j] for j in kept] + [record["last_frame"][episode]]
            labels = [str(j * every) for j in kept] + [f"end {steps}"]
            outcome = "solved" if record["success"][episode] else "FAILED"
            rows.append((f"{name}: {outcome}", frames, labels))
        height, width = rows[0][1][0].shape[:2]
        columns = max(len(frames) for _, frames, _ in rows)
        sheet = Image.new("RGB", (columns * width, len(rows) * (height + 14)))
        draw = ImageDraw.Draw(sheet)
        for r, (title, frames, labels) in enumerate(rows):
            top = r * (height + 14)
            draw.text((3, top + 1), title, fill=(255, 255, 255))
            for c, (frame, label) in enumerate(zip(frames, labels)):
                sheet.paste(Image.fromarray(np.ascontiguousarray(frame)), (c * width, top + 14))
                draw.rectangle([c * width, top + 14, c * width + 7 * len(label) + 5, top + 26], fill=(0, 0, 0))
                draw.text((c * width + 2, top + 14), label, fill=(255, 255, 255))
        slug = "".join(ch if ch.isalnum() else "_" for ch in other_name).strip("_")
        sheet.save(out_dir / f"{slug}__{suite}_task{task_id:02d}_ep{episode:02d}.png")
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace_dir", type=Path, help="directory given to eval_libero.py --trace-dir")
    parser.add_argument("--base", default=None, help="label of the baseline setting (default: the best one)")
    parser.add_argument("--list", type=int, default=40, help="how many lost episodes to list per setting")
    parser.add_argument("--sheets", type=Path, default=None,
                        help="write a picture of every listed lost episode here (baseline above, failing setting below)")
    args = parser.parse_args()

    settings = load(args.trace_dir)
    if not settings:
        raise SystemExit(f"no records in {args.trace_dir}. Run eval_libero.py with --trace-dir first.")
    gain, explained = fit_gain(settings)
    episodes = {name: episodes_of(setting, gain) for name, setting in settings.items()}
    rate = {name: np.mean([e["success"] for e in eps.values()]) for name, eps in episodes.items()}
    if args.base is not None and args.base not in settings:
        raise SystemExit(f"--base {args.base!r} is not among the recorded settings:\n  " + "\n  ".join(settings))
    base = args.base or max(rate, key=rate.get)
    names = [base] + [name for name in settings if name != base]

    print("How each setting ends (episodes, share):\n")
    print(category_table(names, episodes))
    for name in names[1:]:
        print("\n" + loss_report(base, name, episodes[base], episodes[name], args.list))
        if args.sheets is not None:
            _, lost, _ = lost_and_gained(episodes[base], episodes[name])
            written = write_sheets(args.sheets, base, name, settings, lost, args.list)
            print(f"  pictures of {written} lost episodes: {args.sheets}" if written or not lost
                  else "  no pictures: these records were made without camera frames")

    print(f"\nThe arm really moves {' / '.join(f'{g * 100:.2f}' for g in gain)} cm (x / y / z) per unit of action "
          f"per step; the controller is asked for {COMMANDED_M * 100:.0f} cm. "
          f"This fit explains {explained:.0%} of the motion over {GAIN_WINDOW}-step stretches.")
    print("\nHow far off the new plan the robot is when a chunk planned from an old observation takes over:\n")
    print(drift_report(names, episodes))
    ages = {name: np.concatenate([e["age_steps"] for e in eps.values()]) for name, eps in episodes.items()}
    hz = float(next(iter(next(iter(settings.values())).values()))["control_hz"])
    print("\nMean age of the observation behind an executed action: "
          + ", ".join(f"{name}: {np.nanmean(a) * 1e3 / hz:.0f} ms" if np.isfinite(a).any() else f"{name}: -"
                      for name, a in ages.items()))
    print(f"\nThresholds: lifted = {LIFT_M * 100:.0f} cm above the start, reached = gripper within "
          f"{REACH_M * 100:.0f} cm of the target, pushed = {PUSH_M * 100:.0f} cm sideways without a lift, "
          f"another object moved = {OTHER_M * 100:.0f} cm, at the goal = within {NEAR_GOAL_M * 100:.0f} cm, "
          f"early = the first {EARLY_STEPS} steps. Categories come from positions alone.")


if __name__ == "__main__":
    main()
