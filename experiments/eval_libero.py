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

    --settings 512:off:5 384:on:3
                        explicit configurations instead: input resolution, posmap on/off and
                        stale steps for each. A lower resolution makes the vision encoder
                        faster, so the slow inputs are refreshed sooner (fewer stale steps);
                        posmap restores the position layout the checkpoint was trained with.

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
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


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
def plan_chunk(policy, slow_obs: dict, fresh_obs: dict, noise=None) -> torch.Tensor:
    """One normalized action chunk (B, chunk_size, action_dim).

    The prefix (cameras, language, state) is encoded from `slow_obs`; a fast vision path, if
    the policy has one, reads `fresh_obs`. With slow_obs is fresh_obs this is exactly
    `policy.predict_action_chunk`.
    """
    if hasattr(policy, "encode_slow"):  # fast_vision.FastVisionSmolVLAPolicy
        prefix = policy.encode_slow(slow_obs)
        return policy.act_fast(prefix, fresh_obs, noise=noise)
    images, img_masks = policy.prepare_images(slow_obs)
    state = policy.prepare_state(slow_obs)
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


def make_task_env(suite: str, task_id: int, n_envs: int, max_steps: int | None, async_envs: bool = False):
    """(vector env, env preprocessor, env postprocessor) for one LIBERO task.

    With `async_envs` each simulator runs in its own process, so physics and rendering use
    one CPU core per env instead of taking turns on a single core. The processes are
    spawned, not forked: by now this process holds CUDA and thread pools, and a fork of it
    deadlocks in the child before the simulator is even built.
    """
    ensure_libero_config()
    import gymnasium as gym

    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs

    cfg = LiberoEnv(task=suite, task_ids=[task_id], episode_length=max_steps)
    env_cls = partial(gym.vector.AsyncVectorEnv, context="spawn") if async_envs else gym.vector.SyncVectorEnv
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


def rollout(env, env_pre, env_post, policy, preprocessor, postprocessor, *, stale_steps, stale_align,
            seeds, device, on_step=None):
    """One batch of episodes, one per env. Returns per-env success and timing.

    `on_step(step, raw_obs, action, terminated)` is called after every simulator step with
    the new raw observation and the action that produced it (inspect_libero_task.py).
    """
    from lerobot.envs.utils import preprocess_observation

    n_envs = env.num_envs
    n_action_steps = policy.config.n_action_steps
    max_steps = env.call("_max_episode_steps")[0]
    # The instruction is fixed for a task. Read through env.call, which works for both sync
    # and async vector envs (lerobot's add_envs_task needs env.envs, which async lacks).
    tasks = list(env.call("task_description"))

    raw_obs, _ = env.reset(seed=seeds)
    history = deque(maxlen=stale_steps + 1)  # policy-ready observations, newest last
    queue: deque = deque()
    success = np.zeros(n_envs, dtype=bool)
    done = np.zeros(n_envs, dtype=bool)
    plan_ms = []

    for step in range(max_steps):
        obs = preprocess_observation(raw_obs)
        obs["task"] = tasks
        obs = preprocessor(env_pre(obs))
        history.append(obs)

        if not queue:
            slow = history[0]  # oldest kept observation: stale_steps old once the buffer is full
            age = len(history) - 1  # how stale `slow` really is; < stale_steps early in the episode
            synchronize(device)
            t0 = time.perf_counter()
            chunk = plan_chunk(policy, slow, obs)
            synchronize(device)
            plan_ms.append((time.perf_counter() - t0) * 1e3)
            if stale_align == "skip":
                chunk = chunk[:, min(age, chunk.shape[1] - 1) :]
            queue.extend(chunk[:, :n_action_steps].transpose(0, 1))

        action = postprocessor(queue.popleft())
        action = env_post({ACTION: action})[ACTION]
        action = action.to("cpu").numpy()
        raw_obs, _, terminated, truncated, info = env.step(action)
        if on_step is not None:
            on_step(step, raw_obs, action, terminated)

        if "final_info" in info:
            finished = np.asarray(info["final_info"]["is_success"], dtype=bool)
            success |= finished & ~done  # only an env's first episode counts
        done |= terminated | truncated
        if done.all():
            break

    return {"success": success, "steps": step + 1, "plan_ms": plan_ms}


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
                res, posmap, stale = spec.split(":")
                res, stale = int(res), int(stale)
                posmap = {"on": True, "off": False}[posmap]
            except (ValueError, KeyError):
                raise SystemExit(f"--settings entry {spec!r} is not RESOLUTION:on|off:STALE_STEPS, e.g. 384:on:3")
            settings.append({"resolution": res, "posmap": posmap, "stale_steps": stale})
    else:
        settings = [{"resolution": train_resolution, "posmap": False, "stale_steps": d} for d in args.stale_steps]
    for st in settings:
        if st["stale_steps"] < 0:
            raise SystemExit("stale steps must be >= 0")
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
                      ("+posmap" if st["posmap"] else "") + f" stale={st['stale_steps']}"
        st["stale_ms"] = round(st["stale_steps"] * 1e3 / CONTROL_HZ)
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


def evaluate(args, policy, preprocessor, postprocessor, settings, train_resolution, report, out_path) -> None:
    """Every setting on every task. The simulators for a task are started once and shared by
    all settings, and each setting replays the same initial states, seeds and noise, so the
    settings are compared on the same episodes. The report is rewritten after every task."""
    for st in settings:
        st["suites"] = {suite: {"tasks": {}} for suite in args.suites}
        st["_plan_ms"] = []
    t_start = time.perf_counter()
    for suite in args.suites:
        task_ids = args.task_ids if args.task_ids is not None else suite_task_ids(suite)
        for task_id in task_ids:
            print(f"[{suite} task {task_id}] starting {args.n_envs} simulator(s)"
                  f"{' in separate processes' if args.async_envs else ''} ...", flush=True)
            env, env_pre, env_post = make_task_env(suite, task_id, args.n_envs, args.max_steps,
                                                   args.async_envs)
            try:
                for st in settings:
                    apply_setting(policy, st, train_resolution)
                    torch.manual_seed(args.seed + task_id)
                    t_task, wins, episodes = time.perf_counter(), 0, 0
                    while episodes < args.episodes:
                        # Same initial states for every setting: LiberoEnv otherwise advances
                        # them on every reset, including the ones it does itself on success.
                        env.set_attr("init_state_id", [episodes + i for i in range(args.n_envs)])
                        seeds = [args.seed + episodes + i for i in range(args.n_envs)]
                        out = rollout(env, env_pre, env_post, policy, preprocessor, postprocessor,
                                      stale_steps=st["stale_steps"], stale_align=args.stale_align,
                                      seeds=seeds, device=args.device)
                        keep = min(args.n_envs, args.episodes - episodes)
                        wins += int(out["success"][:keep].sum())
                        episodes += keep
                        st["_plan_ms"] += out["plan_ms"]
                        print(f"  [{suite} task {task_id}] {st['label']}: {wins}/{episodes} so far "
                              f"(of {args.episodes}), {time.perf_counter() - t_task:.0f} s", flush=True)
                    st["suites"][suite]["tasks"][task_id] = {
                        "successes": wins, "episodes": episodes, "success_rate": wins / episodes}
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
    parser.add_argument("--stale-align", choices=["none", "skip"], default="none")
    parser.add_argument("--settings", nargs="+", default=None, metavar="RES:on|off:STALE",
                        help="explicit list of configurations, replacing --stale-steps: input resolution "
                             "(multiple of 64), posmap on/off, stale steps. e.g. 512:off:5 384:on:3")
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
        "control_hz": CONTROL_HZ, "settings": [],
    }
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
