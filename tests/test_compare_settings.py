"""compare_settings.py: paired comparison of evaluation reports.

Pure arithmetic on report files, so the expected values here are worked out by hand.
"""

import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import compare_settings as C

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


def task(outcomes=None, rate=None, episodes=20):
    if outcomes is not None:
        return {"successes": sum(outcomes), "episodes": len(outcomes), "success_rate": sum(outcomes) / len(outcomes),
                "episode_success": outcomes}
    return {"successes": round(rate * episodes), "episodes": episodes, "success_rate": rate}


def report(settings, **fields):
    base = {"policy": "p", "seed": 1000, "episodes_per_task": 20, "n_action_steps": 10, "num_steps": 10,
            "chunk_size": 50, "versions": {"mujoco": "3.2.3"}}
    base.update(fields)
    base["settings"] = [{"label": label, "suites": {"libero_spatial": {"tasks": tasks}}} for label, tasks in settings]
    return base


# McNemar: 6 discordant episodes split 5 / 1 -> p = 2 * (C(6,0) + C(6,1)) / 2^6 = 14 / 64.
check("McNemar exact p-value", abs(C.mcnemar_exact(5, 1) - 14 / 64) < 1e-12 and C.mcnemar_exact(0, 0) == 1.0
      and C.mcnemar_exact(3, 3) == 1.0 and abs(C.mcnemar_exact(0, 10) - 2 / 1024) < 1e-12)
# Sign-flip: three equal differences are as extreme only when all signs agree -> 2 of 8.
check("sign-flip exact p-value", C.sign_flip([-0.1, -0.1, -0.1]) == 0.25 and C.sign_flip([]) == 1.0
      and C.sign_flip([0.0, 0.0]) == 1.0)
lo, hi = C.wilson(45, 50)
check("Wilson interval for 45/50", abs(lo - 0.786) < 0.005 and abs(hi - 0.957) < 0.005)

base_out = [1] * 12 + [0] * 8
other_out = [1] * 7 + [0] * 5 + [1] * 1 + [0] * 7  # loses 5 of the baseline's wins, gains 1
with tempfile.TemporaryDirectory() as tmp:
    a, b, c, d = (Path(tmp) / name for name in ("a.json", "b.json", "c.json", "d.json"))
    a.write_text(json.dumps(report([("base", {"0": task(base_out)}), ("worse", {"0": task(other_out)})])))
    b.write_text(json.dumps(report([("worse", {"0": task(other_out)})])))
    c.write_text(json.dumps(report([("old", {"0": task(rate=0.5), "1": task(rate=0.5), "2": task(rate=0.5)})])))
    d.write_text(json.dumps(report([("older", {"0": task(rate=0.4), "1": task(rate=0.4), "2": task(rate=0.4)})],
                                   versions={"mujoco": "3.8.1"}, n_action_steps=50)))

    rows = C.load_rows([a, b])
    check("settings with the same label in two files are told apart by file",
          [r["name"] for r in rows] == ["base", "worse @a.json", "worse @b.json"])
    result = C.compare(rows[0], rows[1])
    check("same episodes: counts the episodes lost and gained and tests them",
          result["test"] == "McNemar" and (result["only_base"], result["only_other"]) == (5, 1)
          and abs(result["p"] - 14 / 64) < 1e-12 and abs(result["diff"] - (8 - 12) / 20) < 1e-12, f"{result}")

    rows = C.load_rows([c, d])
    result = C.compare(rows[0], rows[1])
    check("reports without per-episode outcomes fall back to a per-task test",
          result["test"] == "sign-flip over 3 tasks" and result["p"] == 0.25 and abs(result["diff"] + 0.1) < 1e-12,
          f"{result}")
    notes = C.mismatches(rows[0], rows[1])
    check("flags runs that were not on the same episodes or simulator",
          any("mujoco: 3.2.3 vs 3.8.1" in n for n in notes) and any("n_action_steps: 10 vs 50" in n for n in notes)
          and C.mismatches(*C.load_rows([a, b])[:2]) == [])

    rows = C.load_rows([a, c])
    check("no task in common is reported, not computed", C.compare(rows[1], rows[2])["test"] != "no task in common"
          and C.compare({"tasks": {("s", "9"): task(rate=0.5)}}, rows[0])["test"] == "no task in common")

    out = io.StringIO()
    sys.argv = ["compare_settings.py", str(a), str(b), "--base", "base"]
    with redirect_stdout(out):
        C.main()
    text = out.getvalue()
    check("main: table with the baseline marked, difference in points and the test",
          "base (base)" in text and "-20.0p" in text and "McNemar: 5 lost, 1 gained, p=0.2188" in text
          and "WARNING" not in text, text.splitlines()[2] if len(text.splitlines()) > 2 else text)
    sys.argv = ["compare_settings.py", str(c), str(d)]
    out = io.StringIO()
    with redirect_stdout(out):
        C.main()
    check("main: warns when the runs are not comparable", "WARNING" in out.getvalue() and "mujoco" in out.getvalue())
    sys.argv = ["compare_settings.py", str(a), str(b), "--base", "worse"]
    try:
        with redirect_stdout(io.StringIO()):
            C.main()
        ambiguous = False
    except SystemExit:
        ambiguous = True
    check("main: an ambiguous or unknown baseline label is refused", ambiguous)

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
