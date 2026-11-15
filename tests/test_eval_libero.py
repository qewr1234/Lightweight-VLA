"""eval_libero.py: staleness plumbing, chunk scheduling and success accounting.

The LIBERO simulator needs scene assets from the Hugging Face Hub and a GL backend, so it is
not run here. A stand-in vector env reproduces lerobot's LiberoEnv contract (observation
layout, auto-reset, `final_info`), and everything else is the real code: lerobot's
observation preprocessing, LiberoProcessorStep, the SmolVLA pre/post-processors, the policy
(random weights) and eval_libero's own rollout and report.
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import gymnasium as gym
import numpy as np
import torch

import eval_libero as E  # calls use_local_smolvla()
from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.utils.constants import OBS_STATE

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


class FakeSim:
    """The parts of the LIBERO simulator handle that inspect_libero_task.py reads.

    The bowl rises 1 cm per step for 5 steps of an episode; the ramekin is loaded 4 cm too
    high by set_init_state and sinks 1 cm per simulator step until it rests.
    """

    obj_of_interest = ["bowl_1", "plate_1"]

    def __init__(self, owner):
        self.owner, self.drop = owner, 0.0
        self.env = self
        self.sim = self
        self.data = self
        self.obj_body_id = {"bowl_1": 0, "plate_1": 1, "ramekin_1": 2}

    @property
    def body_xpos(self):
        return np.array([[0.1, 0.2, 0.90 + 0.01 * min(self.owner.t, 5)],
                         [0.3, 0.2, 0.90],
                         [0.1, 0.2, 0.85 + self.drop]])

    def reset(self):
        self.drop = 0.0

    def set_init_state(self, state):
        self.drop = 0.04

    def step(self, action):
        self.drop = max(0.0, self.drop - 0.01)


class FakeLiberoEnv(gym.Env):
    """Same step/observation contract as lerobot.envs.libero.LiberoEnv, no simulator.

    The end-effector x position carries the step counter, so a test can read back which
    observation a plan was made from. Env `index` succeeds at `succeed_at` (None: never).
    """

    def __init__(self, index: int, succeed_at: int | None, max_steps: int = 30):
        self.index, self.succeed_at = index, succeed_at
        self._max_episode_steps = max_steps
        self.task_description = "put the bowl on the plate"
        self.task = self.task_description
        self.t = 0
        self.actions = []
        self.init_state_id = index  # as LiberoEnv: which stored initial state the next reset uses
        self.init_log = []  # init_state_id at every reset the evaluator asked for
        self._env, self._init_states, self.num_steps_wait = FakeSim(self), [0, 1, 2], 10
        image = gym.spaces.Box(0, 255, (64, 64, 3), np.uint8)
        vec = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float64)  # noqa: E731
        self.observation_space = gym.spaces.Dict({
            "pixels": gym.spaces.Dict({"image": image, "image2": image}),
            "robot_state": gym.spaces.Dict({
                "eef": gym.spaces.Dict({"pos": vec(3), "quat": vec(4),
                                        "mat": gym.spaces.Box(-np.inf, np.inf, (3, 3), np.float64)}),
                "gripper": gym.spaces.Dict({"qpos": vec(2), "qvel": vec(2)}),
                "joints": gym.spaces.Dict({"pos": vec(7), "vel": vec(7)}),
            }),
        })
        self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

    def _obs(self):
        img = np.full((64, 64, 3), (self.t * 7 + self.index) % 256, np.uint8)
        return {
            "pixels": {"image": img, "image2": img.copy()},
            "robot_state": {
                "eef": {"pos": np.array([float(self.t), 0.0, 0.0]), "quat": np.array([0.0, 0.0, 0.0, 1.0]),
                        "mat": np.eye(3)},
                "gripper": {"qpos": np.zeros(2), "qvel": np.zeros(2)},
                "joints": {"pos": np.zeros(7), "vel": np.zeros(7)},
            },
        }

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:  # the evaluator's reset; the auto-reset on success passes none
            self.init_log.append(self.init_state_id)
        self.init_state_id += 1  # LiberoEnv moves on to another initial state on every reset
        self.t = 0
        return self._obs(), {"is_success": False}

    def step(self, action):
        assert action.shape == (7,), action.shape
        self.actions.append(np.array(action))
        self.t += 1
        is_success = self.succeed_at is not None and self.t >= self.succeed_at
        info = {"task": self.task, "task_id": 0, "done": False, "is_success": is_success}
        obs = self._obs()
        if is_success:
            info["final_info"] = {"task": self.task, "task_id": 0, "done": False, "is_success": True}
            self.reset()
        return obs, 0.0, is_success, False, info


def fake_env(n_envs, succeed_at, max_steps=30, async_envs=False):
    env_cls = gym.vector.AsyncVectorEnv if async_envs else gym.vector.SyncVectorEnv
    env = env_cls([lambda i=i: FakeLiberoEnv(i, succeed_at[i], max_steps) for i in range(n_envs)])
    env_pre, env_post = make_env_pre_post_processors(env_cfg=LiberoEnvConfig(), policy_cfg=None)
    return env, env_pre, env_post


class Args:
    random, policy_path, rename_map = True, None, None
    num_steps, n_action_steps, dtype, device = 2, 10, "float32", "cpu"


print("=" * 70)
print("Building random-weight policy + real processors ...")
torch.manual_seed(0)
policy, preprocessor, postprocessor = E.load_policy(Args)
cfg = policy.config

# 1) plan_chunk with a fresh observation is predict_action_chunk.
env, env_pre, env_post = fake_env(2, [None, None])
from lerobot.envs.utils import preprocess_observation

raw, _ = env.reset(seed=[0, 1])
obs = preprocess_observation(raw)
obs["task"] = list(env.call("task_description"))
obs = preprocessor(env_pre(obs))
noise = torch.randn(2, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(3))
with torch.no_grad():
    ref = policy.predict_action_chunk(dict(obs), noise=noise.clone())
got = E.plan_chunk(policy, obs, obs, noise=noise.clone())
check("plan_chunk(obs, obs) == predict_action_chunk", (ref - got).abs().max().item() == 0,
      f"max|diff| = {(ref - got).abs().max().item():.2e}, chunk {tuple(got.shape)}")
check("LIBERO state reaches the policy as 8 dims with the step counter in x",
      obs[OBS_STATE].shape == (2, 8) and obs[OBS_STATE][0, 0].item() == 0.0)
env.close()

# 2) Staleness: every plan reads the observation `d` steps old; the fast path would read `t`.
real_plan = E.plan_chunk


def spy_rollout(d, align, n_action_steps=10, max_steps=30):
    log = []

    def spy(policy_, slow, fresh, noise=None):
        log.append((int(slow[OBS_STATE][0, 0].item()), int(fresh[OBS_STATE][0, 0].item())))
        return real_plan(policy_, slow, fresh, noise)

    policy.config.n_action_steps = n_action_steps
    env, env_pre, env_post = fake_env(1, [None], max_steps)
    E.plan_chunk = spy
    try:
        out = E.rollout(env, env_pre, env_post, policy, preprocessor, postprocessor,
                        stale_steps=d, stale_align=align, seeds=[0], device="cpu")
    finally:
        E.plan_chunk = real_plan
        env.close()
    return log, out


log, out = spy_rollout(0, "none")
check("sync: plans at t = 0, 10, 20 from the current observation",
      log == [(0, 0), (10, 10), (20, 20)], f"(slow, fresh) = {log}")
log, out = spy_rollout(5, "none")
check("stale 5: slow input is 5 steps old (clamped at episode start), fresh is current",
      log == [(0, 0), (5, 10), (15, 20)], f"(slow, fresh) = {log}")
check("rollout runs the full episode when nothing succeeds", out["steps"] == 30 and not out["success"].any())

# skip alignment: the first plan has age 0 (nothing to skip), later ones drop 5 of 50 actions
# and still execute 10, so the replan cadence is unchanged.
log, out = spy_rollout(5, "skip")
check("stale 5 + skip: same cadence while the chunk is long enough", log == [(0, 0), (5, 10), (15, 20)],
      f"(slow, fresh) = {log}")
# With n_action_steps = chunk_size = 50 and age 5, only 45 actions remain after the skip.
log, out = spy_rollout(5, "skip", n_action_steps=50, max_steps=120)
check("stale 5 + skip, full-chunk execution: replans after 50, then every 45",
      [f for _, f in log] == [0, 50, 95], f"(slow, fresh) = {log}")

# 3) The skipped chunk really is the tail of the unskipped one.
policy.config.n_action_steps = 10
executed = {}
for align in ("none", "skip"):
    env, env_pre, env_post = fake_env(1, [None], 20)
    torch.manual_seed(11)
    E.rollout(env, env_pre, env_post, policy, preprocessor, postprocessor,
              stale_steps=5, stale_align=align, seeds=[0], device="cpu")
    executed[align] = np.stack(env.envs[0].actions)
    env.close()
# Same seed -> same noise per plan. First plan (age 0) is identical; the second plan's
# first executed action differs because `skip` starts 5 actions into the same chunk.
check("skip changes which actions run after the first chunk, not the first chunk",
      np.array_equal(executed["none"][:10], executed["skip"][:10])
      and not np.array_equal(executed["none"][10:], executed["skip"][10:]))

# 4) Success accounting through the vector env's final_info, first episode only.
env, env_pre, env_post = fake_env(4, [3, None, 12, None])
out = E.rollout(env, env_pre, env_post, policy, preprocessor, postprocessor,
                stale_steps=0, stale_align="none", seeds=[0, 1, 2, 3], device="cpu")
check("per-env success: [True, False, True, False]", out["success"].tolist() == [True, False, True, False],
      f"got {out['success'].tolist()}, {out['steps']} steps")
env.close()

# Same episodes with one process per env: identical outcome.
env, env_pre, env_post = fake_env(4, [3, None, 12, None], async_envs=True)
out_async = E.rollout(env, env_pre, env_post, policy, preprocessor, postprocessor,
                      stale_steps=5, stale_align="skip", seeds=[0, 1, 2, 3], device="cpu")
check("async vector env (one process per simulator): same per-env success",
      out_async["success"].tolist() == [True, False, True, False] and out_async["steps"] == out["steps"],
      f"got {out_async['success'].tolist()}, {out_async['steps']} steps")
env.close()

lo, hi = E.wilson_interval(45, 50)
check("Wilson 95% interval for 45/50", abs(lo - 0.786) < 0.005 and abs(hi - 0.957) < 0.005, f"[{lo:.3f}, {hi:.3f}]")

# 5) main(): sweep, aggregation and the JSON report, with the env factory swapped out.
E.make_task_env = lambda suite, task_id, n_envs, max_steps, async_envs=False: fake_env(
    n_envs, [4 if (i + task_id) % 2 == 0 else None for i in range(n_envs)], max_steps or 30, async_envs
)
with tempfile.TemporaryDirectory() as tmp:
    out_file = Path(tmp) / "report.json"
    sys.argv = ["eval_libero.py", "--random", "--suites", "libero_spatial", "--task-ids", "0", "1",
                "--episodes", "4", "--n-envs", "2", "--async-envs", "--max-steps", "12",
                "--stale-steps", "0", "5",
                "--stale-align", "skip", "--num-steps", "2", "--n-action-steps", "10",
                "--device", "cpu", "--out", str(out_file)]
    E.main()
    report = json.loads(out_file.read_text())
rows = report["settings"]
suite0 = rows[0]["suites"]["libero_spatial"]
check("report: one row per staleness setting, ms from 20 Hz",
      [r["stale_steps"] for r in rows] == [0, 5] and [r["stale_ms"] for r in rows] == [0, 250])
check("report: 2 tasks x 4 episodes, half succeed", suite0["episodes"] == 8 and suite0["successes"] == 4
      and suite0["tasks"]["0"]["success_rate"] == 0.5, f"{suite0['successes']}/{suite0['episodes']}")
check("report: records the evaluated configuration",
      report["n_action_steps"] == 10 and report["num_steps"] == 2 and report["stale_align"] == "skip"
      and rows[0]["plan_ms_median"] > 0)

# 6) Resolution / posmap settings.
train_res = cfg.resize_imgs_with_padding[0]
low_res = train_res - 64
ns = type("NS", (), {"settings": [f"{train_res}:off:0", f"{low_res}:off:0", f"{low_res}:on:3"], "stale_steps": [0]})
sts = E.parse_settings(ns, train_res)
check("settings: parsed into resolution / posmap / stale steps with labels",
      [(s["resolution"], s["posmap"], s["stale_steps"], s["stale_ms"]) for s in sts]
      == [(train_res, False, 0, 0), (low_res, False, 0, 0), (low_res, True, 3, 150)]
      and sts[2]["label"] == f"{low_res}px+posmap stale=3", f"{[s['label'] for s in sts]}")


def rejected(specs, **kw):
    bad = type("NS", (), {"settings": specs, "stale_steps": [0]})
    try:
        E.parse_settings(bad, train_res, **kw)
    except SystemExit:
        return True
    return False


check("settings: malformed, off-grid, no-op posmap and unsupported posmap are refused",
      rejected(["384"]) and rejected(["300:off:0"]) and rejected([f"{low_res}:maybe:0"])
      and rejected([f"{train_res}:on:0"]) and rejected([f"{low_res}:on:0"], posmap_supported=False)
      and not rejected([f"{low_res}:on:0"]))

# What the model sees under each setting: image tokens follow the resolution, and posmap
# puts the action tokens back at the position they start from at the training resolution.
env, env_pre, env_post = fake_env(1, [None])
raw, _ = env.reset(seed=[0])
obs = preprocess_observation(raw)
obs["task"] = list(env.call("task_description"))
obs = preprocessor(env_pre(obs))
env.close()
seen = []
real_encode = policy.model.encode_prefix


def spy_encode(*a, **kw):
    prefix = real_encode(*a, **kw)
    seen.append((prefix.prefix_pad_masks.shape[1],
                 None if prefix.prefix_offsets is None else int(prefix.prefix_offsets[0, 0])))
    return prefix


policy.model.encode_prefix = spy_encode
chunks = []
for st in sts:
    E.apply_setting(policy, st, train_res)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(5))
    chunks.append(E.plan_chunk(policy, obs, obs, noise=noise))
policy.model.encode_prefix = real_encode
n_cam = 2
drop = n_cam * ((train_res // 64) ** 2 - (low_res // 64) ** 2)
check("lower resolution: fewer image tokens in the prefix", seen[0][0] - seen[1][0] == drop
      and seen[1][0] == seen[2][0], f"prefix lengths {[n for n, _ in seen]}, expected drop {drop}")
# All prefix tokens are valid here (both cameras present), so the suffix starts at the prefix length.
check("posmap: action tokens start where they do at the training resolution",
      seen[0][1] is None and seen[1][1] is None and seen[2][1] == seen[0][0],
      f"suffix offset {seen[2][1]} vs training-resolution prefix length {seen[0][0]}")
check("resolution and posmap each change the planned actions",
      not torch.equal(chunks[0], chunks[1]) and not torch.equal(chunks[1], chunks[2]))
E.apply_setting(policy, sts[0], train_res)
check("apply_setting restores the training resolution and turns posmap off",
      tuple(cfg.resize_imgs_with_padding) == (train_res, train_res) and cfg.posmap_ref_resolution is None)

# main() with --settings: every setting replays the same initial states in the same simulators.
made = []


def recording_env(suite, task_id, n_envs, max_steps, async_envs=False):
    made.append(fake_env(n_envs, [4 if (i + task_id) % 2 == 0 else None for i in range(n_envs)], max_steps or 30))
    return made[-1]


E.make_task_env = recording_env
with tempfile.TemporaryDirectory() as tmp:
    out_file = Path(tmp) / "report.json"
    sys.argv = ["eval_libero.py", "--random", "--suites", "libero_spatial", "--task-ids", "0", "1",
                "--episodes", "4", "--n-envs", "2", "--max-steps", "12", "--stale-align", "skip",
                "--settings", f"{train_res}:off:0", f"{low_res}:on:3",
                "--num-steps", "2", "--n-action-steps", "10", "--device", "cpu", "--out", str(out_file)]
    snapshots = []
    real_write = Path.write_text

    def recording_write(self, text, *a, **kw):
        if self == out_file:
            snapshots.append(json.loads(text))
        return real_write(self, text, *a, **kw)

    Path.write_text = recording_write
    try:
        E.main()
    finally:
        Path.write_text = real_write
    report = json.loads(out_file.read_text())
rows = report["settings"]
check("report: one row per setting with its resolution and posmap",
      [(r["resolution"], r["posmap"], r["stale_steps"]) for r in rows] == [(train_res, False, 0), (low_res, True, 3)]
      and report["train_resolution"] == train_res
      and all(r["suites"]["libero_spatial"]["episodes"] == 8 for r in rows))
logs = [[e.init_log for e in env.envs] for env, _, _ in made]
check("paired episodes: one set of simulators per task, same initial states for every setting",
      len(made) == 2 and all(log == [[0, 2, 0, 2], [1, 3, 1, 3]] for log in logs), f"init states per env: {logs[0]}")
first = snapshots[0]["settings"][0]["suites"]["libero_spatial"]
check("partial report after the first task, complete report at the end",
      len(snapshots) == 3 and list(first["tasks"]) == ["0"] and first["episodes"] == 4
      and "plan_ms_median" in snapshots[0]["settings"][0])

# 7) inspect_libero_task.py: scene settling, episode trace and pictures.
import inspect_libero_task as I

E.make_task_env = lambda suite, task_id, n_envs, max_steps, async_envs=False: fake_env(
    n_envs, [8], max_steps or 30)
with tempfile.TemporaryDirectory() as tmp:
    sys.argv = ["inspect_libero_task.py", "--random", "--suite", "libero_spatial", "--task-id", "5",
                "--episodes", "2", "--max-steps", "12", "--num-steps", "2", "--n-action-steps", "10",
                "--device", "cpu", "--out", tmp]
    I.main()
    text = (Path(tmp) / "report.txt").read_text()
    from PIL import Image

    sizes = {p.name: Image.open(p).size for p in Path(tmp).glob("*.png")}
check("inspect: reports the object that moves while the scene settles, and only that one",
      "ramekin_1" in text and text.count("<-- MOVED") == 2 and "4.0 cm  <-- MOVED" in text
      and "nothing moved" not in text)
check("inspect: episode trace - outcome, lift and distance to the goal",
      text.count("SUCCESS after 8 steps") == 2 and "target lifted at most 5.0 cm (step 4)" in text
      and "ends 20.0 cm from the goal horizontally, +5.0 cm in height" in text, text.split("[2]")[1][:300])
check("inspect: one start picture and one 8-frame sheet per episode",
      sizes == {"init_0.png": (128, 64), "init_1.png": (128, 64),
                "episode_0.png": (256, 128), "episode_1.png": (256, 128)}, f"{sizes}")

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
