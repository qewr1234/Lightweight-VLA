"""analyze_failures.py: failure categories, paired losses, drift and pictures, on hand-built records.

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
OTHER0 = np.array([-0.20, 0.20, 0.90])


def episode(target_path, eef_path, grip_end, success=False, steps=STEPS, other_path=None):
    """One episode: target and gripper paths as (steps + 1, 3), gripper command at the end."""
    grip = np.full(steps, -1.0)
    grip[steps // 2:] = grip_end
    return {"target": np.asarray(target_path, float), "eef": np.asarray(eef_path, float), "grip": grip,
            "success": success, "steps": steps, "other": still(OTHER0) if other_path is None else other_path}


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
    ("moved another object", episode(still(TARGET0), far, grip_end=1.0, other_path=move(OTHER0, OTHER0 + [0.3, 0, 0.1]))),
    ("dropped on the way", episode(lifted_then_down, far, grip_end=-1.0)),
    ("still holding at timeout", episode(carried, carried + [0.0, 0.0, 0.03], grip_end=1.0)),
    ("placed off-centre", episode(onto_goal_edge, far, grip_end=-1.0)),
    ("success", episode(onto_goal_edge, far, grip_end=-1.0, success=True, steps=12)),
]


def write(path, label, episodes, solved_all=False, drift=None, objects="all", frames=False):
    """objects: "all" (target, goal, another object), "legacy" (the first record format: the two
    objects of interest and no count), or None."""
    n = len(episodes)
    width = {"all": 9, "legacy": 6, None: 6}[objects]
    positions = np.full((n, STEPS + 1, width), np.nan, np.float32)
    eef = np.full((n, STEPS + 1, 3), np.nan, np.float32)
    action = np.full((n, STEPS, 7), np.nan, np.float32)
    age = np.full((n, STEPS), np.nan, np.float32)
    drifts = np.full((n, STEPS, 6), np.nan, np.float32)
    length = np.zeros(n, np.int32)
    for i, ep in enumerate(episodes):
        s = ep["steps"]
        if objects:
            positions[i, : s + 1, :3] = ep["target"][: s + 1]
            positions[i, : s + 1, 3:6] = GOAL
            if objects == "all":
                positions[i, : s + 1, 6:] = ep["other"][: s + 1]
        eef[i, : s + 1] = ep["eef"][: s + 1]
        action[i, :s] = 0.0
        action[i, :s, -1] = ep["grip"][:s]
        age[i, :s] = 10.0
        length[i] = s
        if drift is not None:
            drifts[i, 12] = drift
    success = np.ones(n, bool) if solved_all else np.array([ep["success"] for ep in episodes])
    extra = {}
    if objects == "all":
        extra["n_interest"] = np.array(2)
    if frames:
        extra.update(frames=np.full((n, STEPS // 5 + 1, 16, 16, 3), 80, np.uint8),
                     last_frame=np.full((n, 16, 16, 3), 200, np.uint8), frame_every=np.array(5))
    names = ["bowl_1", "plate_1", "ramekin_1"][: width // 3]
    np.savez_compressed(path, eef=eef, objects=positions, action=action, age=age, drift=drifts, length=length,
                        success=success, object_names=np.array(names), label=np.array(label),
                        suite=np.array("libero_spatial"), task_id=np.array(0), control_hz=np.array(20), **extra)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    episodes = [ep for _, ep in CASES]
    write(tmp / "base.npz", "base", episodes, solved_all=True, frames=True)
    write(tmp / "stale.npz", "stale", episodes, drift=[0.6, 0.8, 0.0, 9.0, 9.0, 9.0], frames=True)
    write(tmp / "blind.npz", "blind", episodes[:2], objects=None)
    write(tmp / "legacy.npz", "legacy", episodes[:3], objects="legacy")

    settings = A.load(tmp)
    check("records are grouped by setting and task", set(settings) == {"base", "stale", "blind", "legacy"}
          and list(settings["stale"]) == [("libero_spatial", 0)])
    gain = np.full(3, 0.02)
    stale = A.episodes_of(settings["stale"], gain)
    got = [stale[("libero_spatial", 0, i)]["category"] for i in range(len(CASES))]
    check("each built episode lands in its category", got == [name for name, _ in CASES], f"{got}")
    f = stale[("libero_spatial", 0, 4)]["features"]
    check("features: the lift is found, and reach/push are measured before it",
          f["lifted_at"] == 4 and f["steps"] == STEPS and abs(f["final_xy"] - 0.25) < 1e-6 and f["closes"] == 0
          and stale[("libero_spatial", 0, 1)]["features"]["closes"] == 1, f"{f}")
    f = stale[("libero_spatial", 0, 3)]["features"]
    check("features: names the other object that was moved", f["other_name"] == "ramekin_1" and f["other_moved"] > 0.3)
    blind = A.episodes_of(settings["blind"], gain)
    check("without object positions a failure is not guessed at",
          {e["category"] for e in blind.values()} == {"failed (no pick-and-place record)"})
    legacy = A.episodes_of(settings["legacy"], gain)
    check("records of the first format (two objects, no count) are still read",
          [legacy[("libero_spatial", 0, i)]["category"] for i in range(3)] == [name for name, _ in CASES[:3]])
    check("the success episode ends early and keeps no later steps", stale[("libero_spatial", 0, 7)]["features"]["steps"] == 12)

    # Drift of 0.6 and 0.8 action units sideways = 1.0 unit: 5 cm commanded. The built gripper
    # paths ignore the actions (all zero), so the plan intended -drift at the fitted gain and the
    # real offset is what the gripper moved in those 10 steps plus 0.02 m * 1.0.
    r = stale[("libero_spatial", 0, 0)]["replans"]
    check("replans: commanded drift in cm, and a real offset from measured motion and fitted gain",
          len(r) == 1 and r[0]["step"] == 12 and abs(r[0]["commanded_cm"] - 5.0) < 1e-4
          and abs(r[0]["offset_cm"] - 2.0) < 1e-4, f"{r}")

    out = io.StringIO()
    sheets = tmp / "sheets"
    sys.argv = ["analyze_failures.py", str(tmp), "--base", "base", "--sheets", str(sheets), "--list", "3"]
    with redirect_stdout(out):
        A.main()
    text = out.getvalue()
    check("report: the seven episodes the baseline solved and the stale setting did not are counted and listed",
          "'stale' against 'base' on the same 8 episodes: 7 lost, 0 gained, 1 solved by both, 0 by neither" in text
          and "libero_spatial 0   2 -> pushed the target away" in text and ", first 3:" in text)
    check("report: category table, drift row and action age",
          "moved another object" in text and "commanded cm" in text and "stale: 500 ms" in text)
    from PIL import Image

    pictures = sorted(sheets.glob("stale__*.png"))
    size = Image.open(pictures[0]).size if pictures else None
    # 20 steps, a frame every 5 -> 5 frames + the last one = 6 columns of 16 px; two rows of 16 + 14 px.
    check("sheets: one picture per listed lost episode, baseline row above the failing row",
          len(pictures) == 3 and size == (6 * 16, 2 * 30) and "pictures of 3 lost episodes" in text, f"{size}")
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

# The gain: gripper motion that follows the actions at 2 cm per unit per step is recovered.
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    rng = np.random.default_rng(0)
    steps = 60
    act = rng.uniform(-1, 1, (4, steps, 7)).astype(np.float32)
    true_gain = np.array([0.02, 0.015, 0.01])
    eef = np.concatenate([np.zeros((4, 1, 3)), np.cumsum(act[:, :, :3] * true_gain, axis=1)], axis=1).astype(np.float32)
    drift = np.full((4, steps, 6), np.nan, np.float32)
    drift[:, 30, :3] = [1.0, 0.0, 0.0]
    np.savez_compressed(tmp / "g.npz", eef=eef, objects=np.zeros((4, steps + 1, 6), np.float32), action=act,
                        age=np.full((4, steps), 10, np.float32), drift=drift, length=np.full(4, steps, np.int32),
                        success=np.ones(4, bool), object_names=np.array(["a", "b"]), label=np.array("g"),
                        suite=np.array("s"), task_id=np.array(0), control_hz=np.array(20))
    settings = A.load(tmp)
    gain, explained = A.fit_gain(settings)
    check("gain: metres really moved per unit of action per step are recovered from the records",
          np.allclose(gain, true_gain, atol=1e-5) and explained > 0.999, f"{gain.round(4).tolist()}, explained {explained:.4f}")
    # The gripper followed the executed actions exactly, so it is off the new plan by gain * drift.
    r = A.replans(settings["g"][("s", 0)], 0, gain)
    check("replans: when the arm follows its actions the real offset is the drift at the fitted gain",
          len(r) == 1 and abs(r[0]["commanded_cm"] - 5.0) < 1e-4 and abs(r[0]["offset_cm"] - 2.0) < 1e-3, f"{r}")

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
