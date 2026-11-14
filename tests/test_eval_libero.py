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


def fake_env(n_envs, succeed_at, max_steps=30):
    env = gym.vector.SyncVectorEnv(
        [lambda i=i: FakeLiberoEnv(i, succeed_at[i], max_steps) for i in range(n_envs)]
    )
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
from lerobot.envs.utils import add_envs_task, preprocess_observation

raw, _ = env.reset(seed=[0, 1])
obs = preprocessor(env_pre(add_envs_task(env, preprocess_observation(raw))))
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

lo, hi = E.wilson_interval(45, 50)
check("Wilson 95% interval for 45/50", abs(lo - 0.786) < 0.005 and abs(hi - 0.957) < 0.005, f"[{lo:.3f}, {hi:.3f}]")

# 5) main(): sweep, aggregation and the JSON report, with the env factory swapped out.
E.make_task_env = lambda suite, task_id, n_envs, max_steps: fake_env(
    n_envs, [4 if (i + task_id) % 2 == 0 else None for i in range(n_envs)], max_steps or 30
)
with tempfile.TemporaryDirectory() as tmp:
    out_file = Path(tmp) / "report.json"
    sys.argv = ["eval_libero.py", "--random", "--suites", "libero_spatial", "--task-ids", "0", "1",
                "--episodes", "4", "--n-envs", "2", "--max-steps", "12", "--stale-steps", "0", "5",
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

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
