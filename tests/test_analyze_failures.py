"""analyze_failures.py: failure categories, paired losses and drift, on hand-built records.

Each synthetic episode is built to be one clear case, so the expected category is known.
"""

import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import analyze_failures as A

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


STEPS = 20
TARGET0 = np.array([0.0, 0.0, 0.90])
GOAL = np.array([0.30, 0.0, 0.90])


def episode(target_path, eef_path, grip_end, success=False, steps=STEPS):
    """One episode: target and gripper paths as (steps + 1, 3), gripper command at the end."""
    grip = np.full(steps, -1.0)
    grip[steps // 2:] = grip_end
    return {"target": np.asarray(target_path, float), "eef": np.asarray(eef_path, float), "grip": grip,
            "success": success, "steps": steps}


def still(point, steps=STEPS):
    return np.tile(np.asarray(point, float), (steps + 1, 1))


def move(start, end, steps=STEPS):
    return np.linspace(np.asarray(start, float), np.asarray(end, float), steps + 1)


far = still([0.0, 0.40, 1.10])
near = move([0.0, 0.40, 1.10], TARGET0 + [0.0, 0.03, 0.02])
lifted_then_down = np.concatenate([move(TARGET0, TARGET0 + [0.05, 0, 0.08], 10), move(TARGET0 + [0.05, 0, 0.08], [0.05, 0, 0.90], 10)[1:]])
carried = move(TARGET0, [0.15, 0.0, 1.00])
onto_goal_edge = np.concatenate([move(TARGET0, [0.20, 0, 1.00], 10), move([0.20, 0, 1.00], GOAL + [-0.05, 0, 0.01], 10)[1:]])

CASES = [
    ("never reached the target", episode(still(TARGET0), far, grip_end=1.0)),
    ("reached, grasp failed", episode(still(TARGET0), near, grip_end=1.0)),
    ("pushed the target away", episode(move(TARGET0, TARGET0 + [0.0, 0.06, 0.0]), near, grip_end=1.0)),
    ("dropped on the way", episode(lifted_then_down, far, grip_end=-1.0)),
    ("still holding at timeout", episode(carried, carried + [0.0, 0.0, 0.03], grip_end=1.0)),
    ("placed off-centre", episode(onto_goal_edge, far, grip_end=-1.0)),
    ("success", episode(onto_goal_edge, far, grip_end=-1.0, success=True, steps=12)),
]


def write(path, label, episodes, solved_all=False, drift=None, with_objects=True):
    n = len(episodes)
    objects = np.full((n, STEPS + 1, 6), np.nan, np.float32)
    eef = np.full((n, STEPS + 1, 3), np.nan, np.float32)
    action = np.full((n, STEPS, 7), np.nan, np.float32)
    age = np.full((n, STEPS), np.nan, np.float32)
    drifts = np.full((n, STEPS, 6), np.nan, np.float32)
    length = np.zeros(n, np.int32)
    for i, ep in enumerate(episodes):
        s = ep["steps"]
        if with_objects:
            objects[i, : s + 1, :3] = ep["target"][: s + 1]
            objects[i, : s + 1, 3:] = GOAL
        eef[i, : s + 1] = ep["eef"][: s + 1]
        action[i, :s] = 0.0
        action[i, :s, -1] = ep["grip"][:s]
        age[i, :s] = 10.0
        length[i] = s
        if drift is not None:
            drifts[i, 5] = drift
    success = np.ones(n, bool) if solved_all else np.array([ep["success"] for ep in episodes])
    np.savez_compressed(path, eef=eef, objects=objects, action=action, age=age, drift=drifts, length=length,
                        success=success, object_names=np.array(["bowl_1", "plate_1"]), label=np.array(label),
                        suite=np.array("libero_spatial"), task_id=np.array(0), control_hz=np.array(20))


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    episodes = [ep for _, ep in CASES]
    write(tmp / "base.npz", "base", episodes, solved_all=True)
    write(tmp / "stale.npz", "stale", episodes, drift=[0.6, 0.8, 0.0, 9.0, 9.0, 9.0])
    write(tmp / "blind.npz", "blind", episodes[:2], with_objects=False)

    settings = A.load(tmp)
    check("records are grouped by setting and task", set(settings) == {"base", "stale", "blind"}
          and list(settings["stale"]) == [("libero_spatial", 0)])
    stale = A.episodes_of(settings["stale"])
    got = [stale[("libero_spatial", 0, i)]["category"] for i in range(len(CASES))]
    check("each built episode lands in its category", got == [name for name, _ in CASES], f"{got}")
    f = stale[("libero_spatial", 0, 3)]["features"]
    check("features: the lift is found, and reach/push are measured before it",
          f["lifted_at"] == 4 and f["steps"] == STEPS and abs(f["final_xy"] - 0.25) < 1e-6 and f["closes"] == 0
          and stale[("libero_spatial", 0, 1)]["features"]["closes"] == 1, f"{f}")
    blind = A.episodes_of(settings["blind"])
    check("without object positions a failure is not guessed at",
          {e["category"] for e in blind.values()} == {"failed (objects not recorded)"})
    # 0.6 and 0.8 action units sideways = 1.0 unit = 5 cm; the rotation part is not position drift.
    check("drift is reported in cm of commanded motion", np.allclose(stale[("libero_spatial", 0, 0)]["drift_cm"], [5.0]),
          f"{stale[('libero_spatial', 0, 0)]['drift_cm']}")
    check("the success episode ends early and keeps no later steps", stale[("libero_spatial", 0, 6)]["features"]["steps"] == 12)

    out = io.StringIO()
    sys.argv = ["analyze_failures.py", str(tmp), "--base", "base"]
    with redirect_stdout(out):
        A.main()
    text = out.getvalue()
    check("report: the six episodes the baseline solved and the stale setting did not are counted and listed",
          "'stale' against 'base' on the same 7 episodes: 6 lost, 0 gained, 1 solved by both, 0 by neither" in text
          and "libero_spatial 0   5 -> placed off-centre" in text, text.split("\n\n")[1][:200] if "\n\n" in text else text)
    check("report: category table, drift row and action age",
          "pushed the target away" in text and "5.00" in text and "100%" in text and "stale: 500 ms" in text)
    sys.argv = ["analyze_failures.py", str(tmp)]
    out = io.StringIO()
    with redirect_stdout(out):
        A.main()
    check("report: the default baseline is the setting that solves most", "against 'base'" in out.getvalue())
    try:
        sys.argv = ["analyze_failures.py", str(tmp), "--base", "missing"]
        with redirect_stdout(io.StringIO()):
            A.main()
        refused = False
    except SystemExit:
        refused = True
    check("an unknown baseline label is refused", refused)

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
