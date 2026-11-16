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

"""LIBERO task success rate for this repo's SmolVLA, with simulated async staleness.

Runs the policy closed-loop in the LIBERO simulator (through lerobot's own LIBERO env) and
reports success rate per task and per suite. On top of the standard synchronous evaluation
it can reproduce what the async vision/action pipeline does to the policy's inputs:

    --stale-steps 0     synchronous: every chunk is planned from the current observation
                        (what LIBERO numbers in papers measure; the upper bound)
    --stale-steps 5 10  every chunk is planned from the observation `d` control steps ago,
                        i.e. the prefix KV cache the action thread would be holding. LIBERO
                        runs at 20 Hz, so 5 steps = 250 ms and 10 steps = 500 ms - one and
                        two vision periods of the Jetson pipeline.

    --stale-align none  execute the chunk from its first action (the plan lags by d steps)
    --stale-align skip  drop the first d actions, which were meant for steps already past
                        (what lerobot's async inference does)
    --stale-align comp  skip, then steer back onto the plan (drift compensation, see rollout)
    --stale-align fresh encode the stale cameras together with the current robot state and
                        execute from the first action

    --settings 512:off:10/0:fresh
                        a separate age for each camera, in the policy's camera order (LIBERO:
                        front/wrist): here the front camera is 10 steps old and the wrist camera
                        current. Only with the fresh alignment, which takes the robot state from
                        now. Shows which camera's freshness the policy depends on.

    --settings 512:off:5 384:on:3
                        explicit configurations instead: input resolution, posmap on/off and
                        stale steps for each. A lower resolution makes the vision encoder
                        faster, so the slow inputs are refreshed sooner (fewer stale steps);
                        posmap restores the position layout the checkpoint was trained with.

    --pipelines vision=260,action=430
                        instead of a fixed age, replay the timing of the two-thread pipeline:
                        a vision thread that needs `vision` ms to turn the newest camera frames
                        into a prefix cache, and an action thread that needs `action` ms to turn
                        the newest cache into a chunk, while the robot keeps executing at 20 Hz.
                        The age of what the robot acts on then follows from the two numbers,
                        which can be measured on the target device. Options, comma separated:
                        align=none|skip|comp (default skip), sched=all|rr (encode all cameras
                        per vision job, or one camera per job in turn), state=cache|job (robot
                        state as old as the cache, or read when the action job starts).

    --trace-dir runs/traces
                        also write, per setting and task, what happened at every step of every
                        episode: gripper position, action, object positions, how old the
                        observation behind the action was, and the drift at every replan.
                        analyze_failures.py turns these into failure categories.

A policy that carries a fast vision path (fast_vision.attach_fast_vision) gets the current
observation for it automatically; the slow inputs alone are stale.

The model is one simplification away from the real pipeline: here a chunk is planned once
every `n_action_steps` and always from an observation exactly d steps old, while on the
robot the cache age moves between one and two vision periods. Sweep d to cover the range.

    # baseline, no training needed: the public LIBERO checkpoint
    python experiments/eval_libero.py --policy-path lerobot/smolvla_libero \\
        --suites libero_spatial --episodes 50 --n-envs 5 --async-envs

    # staleness sweep on the same checkpoint
    python experiments/eval_libero.py --policy-path lerobot/smolvla_libero \\
        --suites libero_spatial --stale-steps 0 5 10 --stale-align skip

    # resolution vs freshness: a smaller input is blurrier but refreshed sooner
    python experiments/eval_libero.py --policy-path lerobot/smolvla_libero \\
        --suites libero_spatial --stale-align skip \\
        --settings 512:off:0 512:off:5 384:on:0 384:on:3 320:on:0 320:on:2

    # wiring check: random weights, one task, a few steps
    python experiments/eval_libero.py --random --suites libero_spatial --task-ids 0 \\
        --episodes 1 --max-steps 20

Needs `pip install "lerobot[libero]"` (Linux only) and a rendering backend: MUJOCO_GL=egl on
a machine with an NVIDIA driver (the default here), MUJOCO_GL=osmesa for CPU rendering.
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from functools import partial
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _local_smolvla import use_local_smolvla

use_local_smolvla()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors  # noqa: E402
from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

CONTROL_HZ = 20  # LIBERO / robosuite control frequency
STATE_DIM = 8  # eef pos (3) + axis-angle (3) + gripper qpos (2), built by LiberoProcessorStep
ACTION_DIM = 7
POSE_DIMS = 6  # delta position (3) + delta rotation (3); the last action dim is the gripper
ALIGNS = ("none", "skip", "comp", "fresh")
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


# Simulator version the public LIBERO checkpoints are known to work with here. LIBERO builds
# each scene by dropping the objects, so where a stacked object comes to rest depends on the
# physics engine. Under mujoco 3.8.1 the bowl of libero_spatial task 5 lands tilted on the
# ramekin, a scene the training data does not contain, and the task fails every time; under
# 3.2.3 it lands as in the data and succeeds. Versions in between have not been tried.
VERIFIED_MUJOCO = "3.2.3"


def library_versions() -> dict[str, str]:
    versions = {}
    for module in ("mujoco", "robosuite", "lerobot", "torch"):
        try:
            versions[module] = str(__import__(module).__version__)
        except Exception:  # noqa: BLE001  (a missing version string must not stop an evaluation)
            versions[module] = "?"
    return versions


def warn_if_unverified_simulator(versions: dict[str, str]) -> None:
    if versions.get("mujoco") != VERIFIED_MUJOCO:
        print(f"WARNING: mujoco {versions.get('mujoco')} is installed. Success rates here were only checked "
              f"against the training scenes with mujoco {VERIFIED_MUJOCO}; under 3.8.1 stacked objects settle "
              f'differently and libero_spatial task 5 always fails. Fix: pip install "mujoco=={VERIFIED_MUJOCO}"',
              flush=True)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# --------------------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------------------


def load_policy(args):
    """Returns (policy, preprocessor, postprocessor) on args.device."""
    if args.random:
        features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
            "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        }
        config = SmolVLAConfig(
            input_features=features,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))},
            load_vlm_weights=False,
            device=args.device,
        )
        policy = SmolVLAPolicy(config)
        preprocessor, postprocessor = make_smolvla_pre_post_processors(config, dataset_stats=None)
    else:
        policy = SmolVLAPolicy.from_pretrained(args.policy_path)
        overrides = {"device_processor": {"device": args.device}}
        if args.rename_map:
            overrides["rename_observations_processor"] = {"rename_map": json.loads(args.rename_map)}
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=args.policy_path,
            config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
            overrides=overrides,
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=args.policy_path,
            config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

    policy.config.device = args.device
    if args.num_steps is not None:
        policy.config.num_steps = args.num_steps
    if args.n_action_steps is not None:
        if args.n_action_steps > policy.config.chunk_size:
            raise ValueError(f"--n-action-steps {args.n_action_steps} > chunk_size {policy.config.chunk_size}")
        policy.config.n_action_steps = args.n_action_steps
    policy.to(args.device).eval()
    if args.dtype == "bfloat16":
        policy.model.to(torch.bfloat16)
    return policy, preprocessor, postprocessor


@torch.no_grad()
def plan_chunk(policy, slow_obs: dict, fresh_obs: dict, noise=None, state_obs: dict | None = None) -> torch.Tensor:
    """One normalized action chunk (B, chunk_size, action_dim).

    The prefix (cameras, language, state) is encoded from `slow_obs`; a fast vision path, if
    the policy has one, reads `fresh_obs`. With slow_obs is fresh_obs this is exactly
    `policy.predict_action_chunk`. `state_obs` replaces the robot state alone, leaving the
    cameras of `slow_obs`.
    """
    if hasattr(policy, "encode_slow"):  # fast_vision.FastVisionSmolVLAPolicy
        prefix = policy.encode_slow(slow_obs)
        return policy.act_fast(prefix, fresh_obs, noise=noise)
    images, img_masks = policy.prepare_images(slow_obs)
    state = policy.prepare_state(slow_obs if state_obs is None else state_obs)
    prefix = policy.model.encode_prefix(
        images, img_masks, slow_obs[OBS_LANGUAGE_TOKENS], slow_obs[OBS_LANGUAGE_ATTENTION_MASK], state
    )
    actions = policy.model.sample_from_prefix(prefix, noise=noise)
    return actions[:, :, : policy.config.action_feature.shape[0]]


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------


def ensure_libero_config() -> None:
    """LIBERO asks an interactive question on first import; answer it with the defaults."""
    import importlib.util

    import yaml

    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise ModuleNotFoundError('LIBERO is not installed. Run: pip install "lerobot[libero]"  (Linux only)')
    root = Path(list(spec.submodule_search_locations)[0]) / "libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        yaml.dump(
            {
                "benchmark_root": str(root),
                "bddl_files": str(root / "bddl_files"),
                "init_states": str(root / "init_files"),
                "datasets": str(root.parent / "datasets"),
                "assets": str(root / "assets"),
            }
        )
    )
    print(f"wrote default LIBERO config to {config_file}")


def suite_task_ids(suite: str) -> list[int]:
    ensure_libero_config()
    from libero.libero import benchmark

    return list(range(benchmark.get_benchmark_dict()[suite]().n_tasks))


_TRACED_CLASS = None


def _traced_class():
    """LiberoEnv that also reports where the task's objects are (built on first use: importing
    LIBERO needs its config file, which ensure_libero_config writes)."""
    global _TRACED_CLASS
    if _TRACED_CLASS is None:
        from lerobot.envs.libero import LiberoEnv

        class TracedLiberoEnv(LiberoEnv):
            def trace_positions(self):
                core = self._env.env
                return np.concatenate([np.asarray(core.sim.data.body_xpos[core.obj_body_id[name]], dtype=np.float64)
                                       for name in self.trace_names])

            def reset(self, seed=None, **kwargs):
                # LiberoEnv.step resets the simulator itself when an episode ends, so the
                # positions the episode ended with have to be read here, before they are gone.
                self._positions_before_reset = self.trace_positions()
                return super().reset(seed=seed, **kwargs)

            def step(self, action):
                observation, reward, terminated, truncated, info = super().step(action)
                info["object_positions"] = self._positions_before_reset if terminated else self.trace_positions()
                return observation, reward, terminated, truncated, info

        _TRACED_CLASS = TracedLiberoEnv
    return _TRACED_CLASS


def _traced(make_env):
    """Build one simulator and switch it to the traced class. Module-level so that it can be
    sent to the simulator processes."""
    env = make_env()
    interest = list(env._env.obj_of_interest)
    # The task's objects of interest first (what to move, then where to), then everything else
    # in the scene, so that an episode which moves the wrong thing can be told apart.
    env.trace_names = interest + [name for name in env._env.env.obj_body_id if name not in interest]
    env.trace_interest = len(interest)
    env.__class__ = _traced_class()
    return env


def _traced_vector_env(env_cls, make_envs):
    return env_cls([partial(_traced, make_env) for make_env in make_envs])


def make_task_env(suite: str, task_id: int, n_envs: int, max_steps: int | None, async_envs: bool = False,
                  trace: bool = False):
    """(vector env, env preprocessor, env postprocessor) for one LIBERO task.

    With `async_envs` each simulator runs in its own process, so physics and rendering use
    one CPU core per env instead of taking turns on a single core. The processes are
    spawned, not forked: by now this process holds CUDA and thread pools, and a fork of it
    deadlocks in the child before the simulator is even built.

    With `trace` every step also reports the positions of the task's objects of interest.
    """
    ensure_libero_config()
    import gymnasium as gym

    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs

    cfg = LiberoEnv(task=suite, task_ids=[task_id], episode_length=max_steps)
    env_cls = partial(gym.vector.AsyncVectorEnv, context="spawn") if async_envs else gym.vector.SyncVectorEnv
    if trace:
        env_cls = partial(_traced_vector_env, env_cls)
    env = create_libero_envs(
        task=cfg.task,
        n_envs=n_envs,
        camera_name=cfg.camera_name,
        init_states=cfg.init_states,
        gym_kwargs=cfg.gym_kwargs,
        env_cls=env_cls,
        control_mode=cfg.control_mode,
        episode_length=cfg.episode_length,
    )[suite][task_id]
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg, policy_cfg=None)
    return env, env_pre, env_post


# --------------------------------------------------------------------------------------
# Rollout
# --------------------------------------------------------------------------------------


class Trace:
    """What happened at every step of one batch of episodes (analyze_failures.py reads it).

    Arrays are indexed (env, step, ...) and stay NaN after an env's episode has ended. Positions
    have one more entry than actions: index 0 is the state before the first action. A few
    small front-camera frames per episode are kept too, so that an episode can be looked at
    afterwards without running it again (a rerun would not reproduce it exactly).
    """

    def __init__(self, env, max_steps: int, raw_obs: dict, frames: int = 12):
        n = env.num_envs
        self.names = list(env.call("trace_names")[0])
        self.interest = int(env.call("trace_interest")[0])
        start = np.stack(env.call("trace_positions"))
        self.objects = np.full((n, max_steps + 1, start.shape[1]), np.nan, np.float32)
        self.objects[:, 0] = start
        self.eef = np.full((n, max_steps + 1, 3), np.nan, np.float32)
        self.gripper = np.full((n, max_steps + 1, 2), np.nan, np.float32)  # finger joint positions
        self.action = np.full((n, max_steps, env.single_action_space.shape[0]), np.nan, np.float32)
        self.age = np.full((n, max_steps), np.nan, np.float32)  # steps since the observation behind the action
        self.drift = np.full((n, max_steps, POSE_DIMS), np.nan, np.float32)  # where a stale chunk took over
        self.chunks = None  # (env, step, chunk position, action dim): every chunk, at the step it took over
        self.length = np.zeros(n, np.int32)
        self.max_steps = max_steps
        self.frame_every = max(1, int(np.ceil(max_steps / frames))) if frames else 0
        self.frames = self.last_frame = None
        self._state(0, raw_obs, np.ones(n, bool))

    @staticmethod
    def _small(raw_obs: dict) -> np.ndarray:
        """Front camera, the right way up, at half size."""
        return np.asarray(raw_obs["pixels"]["image"])[:, ::-1, ::-1][:, ::2, ::2]

    def _state(self, index: int, raw_obs: dict, live: np.ndarray) -> None:
        """Robot state and camera after `index` actions."""
        self.eef[live, index] = np.asarray(raw_obs["robot_state"]["eef"]["pos"])[live]
        self.gripper[live, index] = np.asarray(raw_obs["robot_state"]["gripper"]["qpos"])[live]
        if not self.frame_every:
            return
        small = self._small(raw_obs)
        if self.frames is None:
            count = self.max_steps // self.frame_every + 1
            self.frames = np.zeros((len(live), count, *small.shape[1:]), np.uint8)
            self.last_frame = np.zeros_like(small)
        if index % self.frame_every == 0:
            self.frames[live, index // self.frame_every] = small[live]
        self.last_frame[live] = small[live]

    def record(self, step: int, raw_obs: dict, action: np.ndarray, info: dict, done: np.ndarray, age) -> None:
        live = ~done
        self._state(step + 1, raw_obs, live)
        self.action[live, step] = action[live]
        if age is not None:
            self.age[live, step] = age
        if "object_positions" in info:
            seen = live & np.asarray(info.get("_object_positions", np.ones(len(live), bool)), dtype=bool)
            self.objects[seen, step + 1] = np.asarray(info["object_positions"])[seen]
        self.length[live] = step + 1

    def record_drift(self, step: int, drift: torch.Tensor, done: np.ndarray) -> None:
        self.drift[~done, step] = drift[:, :POSE_DIMS].numpy()[~done]

    def record_chunk(self, step: int, chunk: torch.Tensor, done: np.ndarray) -> None:
        """`chunk` in simulator units, (env, chunk position, action dim), from its first action."""
        if self.chunks is None:
            self.chunks = np.full((len(done), self.max_steps, *chunk.shape[1:]), np.nan, np.float32)
        self.chunks[~done, step] = chunk.numpy()[~done]

    def arrays(self) -> dict:
        out = {"eef": self.eef, "gripper": self.gripper, "objects": self.objects, "action": self.action,
               "age": self.age, "drift": self.drift, "length": self.length,
               "object_names": np.array(self.names), "n_interest": np.array(self.interest)}
        if self.chunks is not None:
            out["chunks"] = self.chunks
        if self.frames is not None:
            out.update(frames=self.frames, last_frame=self.last_frame, frame_every=np.array(self.frame_every))
        return out


def chunk_to_env(chunk: torch.Tensor, to_env) -> torch.Tensor:
    """A whole normalized chunk (env, position, action dim) in simulator units."""
    return to_env(chunk.reshape(-1, chunk.shape[-1])).reshape(chunk.shape[0], chunk.shape[1], -1)


def stale_drift(chunk, skipped: int, executed: list, to_env) -> torch.Tensor:
    """Summed difference between what ran and what `chunk` planned for the steps it skips.

    A chunk planned from an old observation describes a path that starts where the robot was
    then. The robot spent the skipped steps on the previous chunk instead, so with delta
    actions it is off the new path by exactly this sum (pose dims; the gripper is left out).
    """
    planned = torch.stack([to_env(chunk[:, i]) for i in range(skipped)])
    drift = (torch.stack(list(executed)[-skipped:]) - planned).sum(dim=0)
    drift[:, POSE_DIMS:] = 0
    return drift


def rollout(env, env_pre, env_post, policy, preprocessor, postprocessor, *, stale_steps, stale_align,
            seeds, device, on_step=None, comp_steps=5, trace=False, trace_frames=12):
    """One batch of episodes, one per env. Returns per-env success and timing.

    `on_step(step, raw_obs, action, terminated)` is called after every simulator step with
    the new raw observation and the action that produced it (inspect_libero_task.py).

    Drift compensation (stale_align="comp"). A chunk planned from the observation of `d`
    steps ago describes a path that starts where the robot was then. Its first d actions are
    skipped because that time has passed, but what the robot did in those d steps came from
    the previous chunk, so the robot is not where this plan assumes. With delta actions that
    offset stays for the whole chunk. Both sequences are known, so the offset is too:

        drift = sum_{i<d} (executed_i - planned_i)        (pose dims; the gripper is left alone)

    and subtracting it, spread over the next `comp_steps` actions, puts the robot back on the
    path the new plan meant. No model call, no training.
    """
    from lerobot.envs.utils import preprocess_observation

    if stale_align not in ALIGNS:
        raise ValueError(f"stale_align must be one of {ALIGNS}, got {stale_align!r}")
    camera_ages = None
    if not isinstance(stale_steps, int):  # one age per camera
        camera_keys = list(policy.config.image_features)
        camera_ages = dict(zip(camera_keys, stale_steps, strict=True))
        if stale_align != "fresh":
            raise ValueError("per-camera ages need stale_align='fresh' (the robot state is taken from now)")
        stale_steps = max(stale_steps)
    n_envs = env.num_envs
    n_action_steps = policy.config.n_action_steps
    max_steps = env.call("_max_episode_steps")[0]
    # The instruction is fixed for a task. Read through env.call, which works for both sync
    # and async vector envs (lerobot's add_envs_task needs env.envs, which async lacks).
    tasks = list(env.call("task_description"))
    low = torch.as_tensor(env.single_action_space.low[:POSE_DIMS], dtype=torch.float32)
    high = torch.as_tensor(env.single_action_space.high[:POSE_DIMS], dtype=torch.float32)

    def to_env(normalized):
        """Normalized policy action -> the action the simulator takes, on CPU."""
        return env_post({ACTION: postprocessor(normalized)})[ACTION].to("cpu", torch.float32)

    raw_obs, _ = env.reset(seed=seeds)
    record = Trace(env, max_steps, raw_obs, trace_frames) if trace else None
    history = deque(maxlen=stale_steps + 1)  # policy-ready observations, newest last
    executed = deque(maxlen=max(stale_steps, 1))  # actions sent to the simulator, newest last
    queue: deque = deque()
    corrections: deque = deque()
    success = np.zeros(n_envs, dtype=bool)
    done = np.zeros(n_envs, dtype=bool)
    plan_ms = []
    drift_norms = []
    observed_at = 0  # step of the observation the queued actions were planned from

    for step in range(max_steps):
        obs = preprocess_observation(raw_obs)
        obs["task"] = tasks
        obs = preprocessor(env_pre(obs))
        history.append(obs)

        if not queue:
            slow = history[0]  # oldest kept observation: stale_steps old once the buffer is full
            age = len(history) - 1  # how stale `slow` really is; < stale_steps early in the episode
            if camera_ages is not None:
                slow = dict(obs)
                for key, camera_age in camera_ages.items():
                    slow[key] = history[-1 - min(camera_age, age)][key]
            extra = {"state_obs": obs} if stale_align == "fresh" else {}
            synchronize(device)
            t0 = time.perf_counter()
            chunk = plan_chunk(policy, slow, obs, **extra)
            synchronize(device)
            plan_ms.append((time.perf_counter() - t0) * 1e3)
            start = min(age, chunk.shape[1] - 1) if stale_align in ("skip", "comp") else 0
            queue.extend(chunk[:, start : start + n_action_steps].transpose(0, 1))
            observed_at = step - age
            if record is not None:
                record.record_chunk(step, chunk_to_env(chunk, to_env), done)
            if start > 0:
                # Measured for skip as well, where nothing is done about it: it is the offset
                # the robot then carries through the chunk.
                drift = stale_drift(chunk, start, executed, to_env)
                if (~done).any():
                    drift_norms.append(float(drift[~done, :3].norm(dim=1).mean()))
                if record is not None:
                    record.record_drift(step, drift, done)
                if stale_align == "comp":
                    spread = max(1, min(comp_steps, len(queue)))
                    corrections = deque([-drift / spread] * spread)

        action = to_env(queue.popleft())
        if stale_align == "comp":
            if corrections:
                action = action + corrections.popleft()
            # What the controller would clip anyway; done here so `executed` is what really ran.
            action[:, :POSE_DIMS] = torch.minimum(torch.maximum(action[:, :POSE_DIMS], low), high)
        executed.append(action)
        action = action.numpy()
        raw_obs, _, terminated, truncated, info = env.step(action)
        if on_step is not None:
            on_step(step, raw_obs, action, terminated)
        if record is not None:
            record.record(step, raw_obs, action, info, done, step - observed_at)

        if "final_info" in info:
            finished = np.asarray(info["final_info"]["is_success"], dtype=bool)
            success |= finished & ~done  # only an env's first episode counts
        done |= terminated | truncated
        if done.all():
            break

    return {"success": success, "steps": step + 1, "plan_ms": plan_ms, "drift": drift_norms,
            "trace": None if record is None else record.arrays()}


def parse_pipeline(spec: str) -> dict:
    """'vision=260,action=430,align=comp' -> validated dict with defaults."""
    out = {"vision": None, "action": None, "align": "skip", "sched": "all", "state": "cache"}
    try:
        for part in spec.split(","):
            key, value = part.split("=")
            if key not in out:
                raise ValueError(key)
            out[key] = float(value) if key in ("vision", "action") else value
        if out["vision"] is None or out["action"] is None or out["vision"] <= 0 or out["action"] <= 0:
            raise ValueError("vision and action times")
        if out["align"] not in ("none", "skip", "comp") or out["sched"] not in ("all", "rr") \
                or out["state"] not in ("cache", "job"):
            raise ValueError("option value")
    except ValueError as err:
        raise SystemExit(f"--pipelines entry {spec!r} is not valid ({err}). Example: "
                         f"vision=260,action=430,align=skip,sched=all,state=cache") from None
    return out


def rollout_pipeline(env, env_pre, env_post, policy, preprocessor, postprocessor, *, pipeline, seeds, device,
                     comp_steps=5, trace=False, trace_frames=12):
    """One batch of episodes under the timing of the asynchronous two-thread pipeline.

    Time advances in control steps of 1/CONTROL_HZ. Two jobs run alongside the robot:

      vision job   takes the newest camera frame(s) when it starts and, `vision` ms later
                   (divided by the number of cameras when sched=rr encodes one per job),
                   publishes them as the prefix cache. It restarts at once.
      action job   starts when it is idle and a newer cache exists; `action` ms later its
                   chunk replaces the action queue. With align skip/comp the actions meant
                   for the steps that have passed since the chunk's reference time are
                   dropped. The reference time is that of the robot state the chunk was
                   planned from: the cache's (state=cache) or the job's start (state=job).

    The robot executes one queued action per step and holds still (no motion, gripper as it
    was) when the queue is empty, as at the start of an episode. The policy itself is
    evaluated when a chunk becomes ready, from the frames the job would have used.
    """
    from lerobot.envs.utils import preprocess_observation

    dt = 1e3 / CONTROL_HZ
    cameras = list(policy.config.image_features)
    jobs = [cameras] if pipeline["sched"] == "all" else [[c] for c in cameras]
    job_ms = pipeline["vision"] * len(jobs[0]) / len(cameras)
    action_ms, align = pipeline["action"], pipeline["align"]
    n_envs = env.num_envs
    max_steps = env.call("_max_episode_steps")[0]
    tasks = list(env.call("task_description"))
    low = torch.as_tensor(env.single_action_space.low[:POSE_DIMS], dtype=torch.float32)
    high = torch.as_tensor(env.single_action_space.high[:POSE_DIMS], dtype=torch.float32)

    def to_env(normalized):
        return env_post({ACTION: postprocessor(normalized)})[ACTION].to("cpu", torch.float32)

    raw_obs, _ = env.reset(seed=seeds)
    record = Trace(env, max_steps, raw_obs, trace_frames) if trace else None
    keep = int(np.ceil((2 * pipeline["vision"] + action_ms) / dt)) + 8
    recent: deque = deque(maxlen=keep)  # (step, policy-ready observation), newest last

    def obs_at(step):
        return recent[step - recent[0][0]][1]

    cache: dict = {}  # camera -> step of the frame in the published cache
    cache_version, cache_ready = 0, 0.0
    vision_job, vision_step, vision_done = 0, 0, job_ms
    busy, action_done, used_version = False, 0.0, 0
    job_frames, job_state_step = {}, 0
    queue: deque = deque()
    corrections: deque = deque()
    chunk_ref = 0
    executed = []
    hold = torch.zeros(n_envs, env.single_action_space.shape[0])
    hold[:, POSE_DIMS:] = -1.0  # gripper open until the policy says otherwise
    success = np.zeros(n_envs, dtype=bool)
    done = np.zeros(n_envs, dtype=bool)
    plan_ms, ages, drift_norms = [], [], []
    hold_steps = expired = 0

    for step in range(max_steps):
        now = step * dt
        obs = preprocess_observation(raw_obs)
        obs["task"] = tasks
        obs = preprocessor(env_pre(obs))
        recent.append((step, obs))

        while vision_done <= now:  # a vision job finished: publish, restart on the newest frame
            for camera in jobs[vision_job]:
                cache[camera] = vision_step
            cache_version, cache_ready = cache_version + 1, vision_done
            vision_job = (vision_job + 1) % len(jobs)
            vision_step = min(step, int(vision_done // dt))
            vision_done += job_ms

        if busy and action_done <= now:  # a chunk is ready
            busy = False
            slow = dict(obs_at(job_state_step))
            for camera, frame_step in job_frames.items():
                slow[camera] = obs_at(frame_step)[camera]
            synchronize(device)
            t0 = time.perf_counter()
            chunk = plan_chunk(policy, slow, obs, state_obs=obs_at(job_state_step))
            synchronize(device)
            plan_ms.append((time.perf_counter() - t0) * 1e3)
            skip = step - job_state_step if align in ("skip", "comp") else 0
            if record is not None:
                record.record_chunk(step, chunk_to_env(chunk, to_env), done)
            if skip >= chunk.shape[1]:
                expired += 1  # every action of this chunk was meant for a step already past
            else:
                queue = deque(chunk[:, skip:].transpose(0, 1))
                chunk_ref = job_state_step
                corrections = deque()
                if skip > 0:
                    drift = stale_drift(chunk, skip, executed, to_env)
                    if (~done).any():
                        drift_norms.append(float(drift[~done, :3].norm(dim=1).mean()))
                    if record is not None:
                        record.record_drift(step, drift, done)
                    if align == "comp":
                        spread = max(1, min(comp_steps, len(queue)))
                        corrections = deque([-drift / spread] * spread)

        if not busy and len(cache) == len(cameras) and cache_version != used_version:
            start = max(action_done, cache_ready)
            busy, used_version, action_done = True, cache_version, start + action_ms
            job_frames = dict(cache)
            job_state_step = max(cache.values()) if pipeline["state"] == "cache" else min(step, int(start // dt))

        if queue:
            action = to_env(queue.popleft())
            if corrections:
                action = action + corrections.popleft()
            if align == "comp":
                action[:, :POSE_DIMS] = torch.minimum(torch.maximum(action[:, :POSE_DIMS], low), high)
            hold[:, POSE_DIMS:] = action[:, POSE_DIMS:]
            ages.append(step - chunk_ref)
            action_age = step - chunk_ref
        else:
            action = hold.clone()
            hold_steps += 1
            action_age = None  # nothing was planned: the robot is standing still
        executed.append(action)
        raw_obs, _, terminated, truncated, info = env.step(action.numpy())
        if record is not None:
            record.record(step, raw_obs, action.numpy(), info, done, action_age)

        if "final_info" in info:
            finished = np.asarray(info["final_info"]["is_success"], dtype=bool)
            success |= finished & ~done
        done |= terminated | truncated
        if done.all():
            break

    return {"success": success, "steps": step + 1, "plan_ms": plan_ms, "drift": drift_norms,
            "age_steps": ages, "hold_steps": hold_steps, "expired_chunks": expired,
            "trace": None if record is None else record.arrays()}


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def parse_settings(args, train_resolution: int | None, posmap_supported: bool = True) -> list[dict]:
    """One dict per evaluated configuration: resolution, posmap, stale_steps."""
    settings = []
    if args.settings:
        for spec in args.settings:
            try:
                res, posmap, stale, *align = spec.split(":")
                res = int(res)
                stale = [int(a) for a in stale.split("/")] if "/" in stale else int(stale)
                posmap = {"on": True, "off": False}[posmap]
                if len(align) > 1 or (align and align[0] not in ALIGNS):
                    raise ValueError(spec)
            except (ValueError, KeyError):
                raise SystemExit(f"--settings entry {spec!r} is not RESOLUTION:on|off:STALE_STEPS[:ALIGN], "
                                 f"e.g. 384:on:3 or 512:off:10:comp (ALIGN is one of {', '.join(ALIGNS)})")
            settings.append({"resolution": res, "posmap": posmap, "stale_steps": stale,
                             "align": align[0] if align else args.stale_align, "_explicit_align": bool(align)})
    else:
        settings = [{"resolution": train_resolution, "posmap": False, "stale_steps": d, "align": args.stale_align}
                    for d in args.stale_steps]
    if getattr(args, "pipelines", None):
        if not args.settings:
            settings = []  # only the pipeline rows, unless --settings asks for more
        for spec in args.pipelines:
            pipe = parse_pipeline(spec)
            label = (f"pipe v{pipe['vision']:g} a{pipe['action']:g} {pipe['align']}"
                     + ("" if pipe["sched"] == "all" else f" {pipe['sched']}")
                     + ("" if pipe["state"] == "cache" else " state=job"))
            settings.append({"resolution": train_resolution, "posmap": False, "stale_steps": None,
                             "align": pipe["align"], "pipeline": pipe, "label": label, "stale_ms": None})
    for st in settings:
        if st.get("pipeline"):
            continue
        ages = st["stale_steps"] if isinstance(st["stale_steps"], list) else [st["stale_steps"]]
        if min(ages) < 0:
            raise SystemExit("stale steps must be >= 0")
        if len(ages) > 1 and st["align"] != "fresh":
            raise SystemExit("a separate age per camera (e.g. 10/0) needs the fresh alignment: 512:off:10/0:fresh")
        if st["posmap"] and not posmap_supported:
            # The model would ignore the flag and the row would be mislabelled.
            raise SystemExit("posmap needs a checkpoint with add_image_special_tokens=false")
        if st["posmap"] and st["resolution"] == train_resolution:
            raise SystemExit(f"posmap at the training resolution ({train_resolution}) changes nothing; "
                             f"write {train_resolution}:off:{st['stale_steps']}")
        if st["resolution"] != train_resolution:
            if train_resolution is None:
                raise SystemExit("this checkpoint has no fixed input resolution; --settings cannot change it")
            if st["resolution"] % 64 or st["resolution"] <= 0:
                raise SystemExit(f"resolution {st['resolution']} must be a positive multiple of 64")
        st["label"] = (f"{st['resolution']}px" if st["resolution"] else "native") + \
                      ("+posmap" if st["posmap"] else "") + f" stale={'/'.join(map(str, ages))}" + \
                      (f" {st['align']}" if st.pop("_explicit_align", False) else "")
        st["stale_ms"] = round(max(ages) * 1e3 / CONTROL_HZ)
    return settings


def apply_setting(policy, setting: dict, train_resolution: int | None) -> None:
    """Point the policy at this setting's input resolution and position layout."""
    res = setting["resolution"]
    if res is not None:
        policy.config.resize_imgs_with_padding = (res, res)
    # posmap puts the image tokens back on the position grid the checkpoint was trained with.
    policy.config.posmap_ref_resolution = train_resolution if setting["posmap"] else None


def summarize(setting: dict, suites: list[str]) -> None:
    """Fill per-suite totals and the Wilson interval from the per-task counts."""
    for suite in suites:
        r = setting["suites"][suite]
        k = sum(t["successes"] for t in r["tasks"].values())
        n = sum(t["episodes"] for t in r["tasks"].values())
        lo, hi = wilson_interval(k, n)
        r.update({"successes": k, "episodes": n, "success_rate": k / n if n else 0.0, "ci95": [lo, hi]})
    plan = setting.pop("_plan_ms", None)
    if plan:
        setting["plan_ms_median"] = float(np.median(plan))
    drift = setting.pop("_drift", None)
    if drift:  # position part of the compensated drift, in action units, per replan
        setting["drift_mean"] = float(np.mean(drift))
    age, hold, expired = setting.pop("_age", None), setting.pop("_hold", None), setting.pop("_expired", 0)
    if setting.get("pipeline"):
        # How old the observation behind an executed action was, the share of steps the robot
        # stood still for lack of an action, and chunks that arrived entirely too late.
        setting["action_age_ms_mean"] = float(np.mean(age)) * 1e3 / CONTROL_HZ if age else None
        setting["hold_fraction"] = float(np.mean(hold)) if hold else None
        setting["expired_chunks"] = expired


def trace_file(trace_dir: Path, label: str, suite: str, task_id: int) -> Path:
    """One file per setting and task; running the same setting again replaces it."""
    slug = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    return trace_dir / f"{slug}__{suite}__task{task_id:02d}.npz"


def save_trace(path: Path, batches: list[dict], outcomes: list[int], label: str, suite: str, task_id: int) -> None:
    """Join the batches of one setting and task along the episode axis and write them."""
    episodes = len(outcomes)
    per_episode = [key for key, value in batches[0].items()
                   if key not in ("object_names", "n_interest", "frame_every") and all(key in b for b in batches)]
    arrays = {key: np.concatenate([b[key] for b in batches])[:episodes] for key in per_episode}
    for key in ("object_names", "n_interest", "frame_every"):
        if key in batches[0]:
            arrays[key] = batches[0][key]
    np.savez_compressed(path, **arrays, success=np.asarray(outcomes, dtype=bool), label=np.array(label),
                        suite=np.array(suite), task_id=np.array(task_id), control_hz=np.array(CONTROL_HZ))


def evaluate(args, policy, preprocessor, postprocessor, settings, train_resolution, report, out_path) -> None:
    """Every setting on every task. The simulators for a task are started once and shared by
    all settings, and each setting replays the same initial states, seeds and noise, so the
    settings are compared on the same episodes. The report is rewritten after every task."""
    for st in settings:
        st["suites"] = {suite: {"tasks": {}} for suite in args.suites}
        st["_plan_ms"] = []
        st["_drift"] = []
        st["_age"], st["_hold"], st["_expired"] = [], [], 0
    t_start = time.perf_counter()
    for suite in args.suites:
        task_ids = args.task_ids if args.task_ids is not None else suite_task_ids(suite)
        for task_id in task_ids:
            print(f"[{suite} task {task_id}] starting {args.n_envs} simulator(s)"
                  f"{' in separate processes' if args.async_envs else ''} ...", flush=True)
            trace_dir = Path(args.trace_dir) if getattr(args, "trace_dir", None) else None
            env, env_pre, env_post = make_task_env(suite, task_id, args.n_envs, args.max_steps, args.async_envs,
                                                   **({"trace": True} if trace_dir else {}))
            try:
                for st in settings:
                    apply_setting(policy, st, train_resolution)
                    torch.manual_seed(args.seed + task_id)
                    t_task, wins, episodes = time.perf_counter(), 0, 0
                    outcomes = []  # 1/0 per episode; episode i always starts from initial state i
                    traces = []
                    while episodes < args.episodes:
                        # Same initial states for every setting: LiberoEnv otherwise advances
                        # them on every reset, including the ones it does itself on success.
                        env.set_attr("init_state_id", [episodes + i for i in range(args.n_envs)])
                        seeds = [args.seed + episodes + i for i in range(args.n_envs)]
                        if st.get("pipeline"):
                            out = rollout_pipeline(env, env_pre, env_post, policy, preprocessor, postprocessor,
                                                   pipeline=st["pipeline"], seeds=seeds, device=args.device,
                                                   comp_steps=args.comp_steps, trace=bool(trace_dir),
                                                   trace_frames=args.trace_frames)
                            st["_age"] += out["age_steps"]
                            st["_hold"] += [out["hold_steps"] / out["steps"]]
                            st["_expired"] += out["expired_chunks"]
                        else:
                            out = rollout(env, env_pre, env_post, policy, preprocessor, postprocessor,
                                          stale_steps=(tuple(st["stale_steps"]) if isinstance(st["stale_steps"], list)
                                                       else st["stale_steps"]), stale_align=st["align"],
                                          seeds=seeds, device=args.device, comp_steps=args.comp_steps,
                                          trace=bool(trace_dir), trace_frames=args.trace_frames)
                        keep = min(args.n_envs, args.episodes - episodes)
                        wins += int(out["success"][:keep].sum())
                        outcomes += [int(x) for x in out["success"][:keep]]
                        episodes += keep
                        if trace_dir:
                            traces.append(out["trace"])
                        st["_plan_ms"] += out["plan_ms"]
                        st["_drift"] += out["drift"]
                        print(f"  [{suite} task {task_id}] {st['label']}: {wins}/{episodes} so far "
                              f"(of {args.episodes}), {time.perf_counter() - t_task:.0f} s", flush=True)
                    st["suites"][suite]["tasks"][task_id] = {
                        "successes": wins, "episodes": episodes, "success_rate": wins / episodes,
                        # Kept per episode so two settings can be compared on the same episodes
                        # (compare_settings.py), which needs far fewer episodes than comparing rates.
                        "episode_success": outcomes}
                    if trace_dir:
                        trace_dir.mkdir(parents=True, exist_ok=True)
                        save_trace(trace_file(trace_dir, st["label"], suite, task_id), traces, outcomes,
                                   st["label"], suite, task_id)
            finally:
                env.close()
            # Partial results survive an interrupted run.
            snapshot = json.loads(json.dumps(settings, default=lambda o: None))
            for st in snapshot:
                summarize(st, args.suites)
            report["settings"] = snapshot
            report["wall_s"] = round(time.perf_counter() - t_start, 1)
            out_path.write_text(json.dumps(report, indent=2))
    for st in settings:
        summarize(st, args.suites)
    report["settings"] = settings
    report["wall_s"] = round(time.perf_counter() - t_start, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy-path", help="checkpoint dir or hub id, e.g. lerobot/smolvla_libero")
    src.add_argument("--random", action="store_true", help="random weights: wiring check only")
    parser.add_argument("--suites", nargs="+", default=["libero_spatial"], choices=SUITES)
    parser.add_argument("--task-ids", nargs="+", type=int, default=None, help="default: every task in the suite")
    parser.add_argument("--episodes", type=int, default=50, help="episodes per task")
    parser.add_argument("--n-envs", type=int, default=1, help="simulators per task, stepped as one batch")
    parser.add_argument("--async-envs", action="store_true",
                        help="one process per simulator: uses a CPU core each (much faster with n-envs > 1)")
    parser.add_argument("--max-steps", type=int, default=None, help="episode length (default: the suite's)")
    parser.add_argument("--stale-steps", nargs="+", type=int, default=[0],
                        help=f"age of the slow inputs in control steps ({CONTROL_HZ} Hz); several values = a sweep")
    parser.add_argument("--stale-align", choices=ALIGNS, default="none")
    parser.add_argument("--comp-steps", type=int, default=5,
                        help="with align comp: number of actions the drift correction is spread over")
    parser.add_argument("--settings", nargs="+", default=None, metavar="RES:on|off:STALE",
                        help="explicit list of configurations, replacing --stale-steps: input resolution "
                             "(multiple of 64), posmap on/off, stale steps and optionally the alignment. "
                             "e.g. 512:off:5 384:on:3 512:off:10:comp")
    parser.add_argument("--pipelines", nargs="+", default=None, metavar="vision=MS,action=MS[,...]",
                        help="replay the two-thread pipeline's timing, e.g. vision=260,action=430,align=comp")
    parser.add_argument("--trace-dir", default=None,
                        help="also record every step of every episode here (for analyze_failures.py)")
    parser.add_argument("--trace-frames", type=int, default=12,
                        help="with --trace-dir: small front-camera frames kept per episode (0: none). "
                             "About 0.6 MB per episode before compression")
    parser.add_argument("--num-steps", type=int, default=None, help="denoising steps (default: the checkpoint's)")
    parser.add_argument("--n-action-steps", type=int, default=None,
                        help="actions executed per chunk (default: the checkpoint's)")
    parser.add_argument("--rename-map", default=None,
                        help="JSON overriding the checkpoint's observation rename map")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default="./runs/eval_libero.json")
    args = parser.parse_args()
    if min(args.stale_steps) < 0 or args.episodes < 1 or args.n_envs < 1:
        parser.error("--stale-steps must be >= 0, --episodes and --n-envs >= 1")
    if args.episodes % args.n_envs:
        parser.error("--episodes must be a multiple of --n-envs (each batch runs one episode per simulator)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    policy, preprocessor, postprocessor = load_policy(args)
    cfg = policy.config
    print(f"policy: {'random weights' if args.random else args.policy_path} | "
          f"vlm_layers={policy.model.vlm_with_expert.num_vlm_layers} chunk={cfg.chunk_size} "
          f"n_action_steps={cfg.n_action_steps} num_steps={cfg.num_steps} | device={args.device} {args.dtype}")

    resize = cfg.resize_imgs_with_padding
    train_resolution = resize[0] if resize and resize[0] == resize[1] else None
    settings = parse_settings(args, train_resolution, not policy.model.add_image_special_tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "policy": "random" if args.random else args.policy_path,
        "num_vlm_layers": policy.model.vlm_with_expert.num_vlm_layers,
        "chunk_size": cfg.chunk_size, "n_action_steps": cfg.n_action_steps, "num_steps": cfg.num_steps,
        "train_resolution": train_resolution,
        "stale_align": args.stale_align, "episodes_per_task": args.episodes, "seed": args.seed,
        "control_hz": CONTROL_HZ, "versions": library_versions(), "settings": [],
    }
    warn_if_unverified_simulator(report["versions"])
    print(f"settings ({len(settings)}), align={args.stale_align}: " + " | ".join(st["label"] for st in settings))
    evaluate(args, policy, preprocessor, postprocessor, settings, train_resolution, report, out)

    width = max(len(st["label"]) for st in settings)
    print(f"\n{'setting':>{width}} | " + " | ".join(f"{su:>24}" for su in args.suites))
    for st in report["settings"]:
        cells = []
        for suite in args.suites:
            r = st["suites"][suite]
            cells.append(f"{r['success_rate']:6.1%} [{r['ci95'][0]:.0%}-{r['ci95'][1]:.0%}] n={r['episodes']}")
        print(f"{st['label']:>{width}} | " + " | ".join(f"{c:>24}" for c in cells))

    out.write_text(json.dumps(report, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
