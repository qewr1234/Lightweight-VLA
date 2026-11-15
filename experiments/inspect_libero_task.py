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

"""Look inside one LIBERO task: is the scene what the instruction says, and what does the
policy do in it?

For a task that fails every time the success rate says nothing about why. This script
answers two questions with numbers and pictures:

1. Does the scene hold still? Each initial state is loaded and the simulator is stepped
   with the no-op action, as the evaluation does before every episode. An object that moves
   during those steps (for example a bowl sliding off what it was stacked on) means the
   episode starts from a scene the policy was never trained on.
2. What happens in an episode? The same rollout as eval_libero.py is run for a few
   episodes while the objects, the gripper and the camera are recorded: how close the
   gripper gets to the target, whether anything is lifted, where the target ends up.

    python experiments/inspect_libero_task.py --policy-path lerobot/smolvla_libero \\
        --suite libero_spatial --task-id 5 --n-action-steps 10 --episodes 3 \\
        --out runs/inspect_task5

Writes report.txt, init_<k>.png (the camera view each episode starts from) and
episode_<k>.png (12 frames across the episode) into --out.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_libero as E  # noqa: E402  (calls use_local_smolvla())

import numpy as np  # noqa: E402
import torch  # noqa: E402

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]  # lerobot's no-op: no motion, gripper open
MOVED = 0.01  # metres; less than this is contact settling


def object_positions(inner) -> dict[str, np.ndarray]:
    """World position of every object and fixture in the scene, by name."""
    core = inner._env.env
    return {name: np.array(core.sim.data.body_xpos[body_id]) for name, body_id in core.obj_body_id.items()}


def view(image: np.ndarray) -> np.ndarray:
    """Camera frame the right way up (the simulator renders it rotated by 180 degrees)."""
    return np.ascontiguousarray(image[::-1, ::-1])


def save_sheet(frames: list[np.ndarray], labels: list[str], path: Path, columns: int = 4) -> None:
    from PIL import Image, ImageDraw

    h, w = frames[0].shape[:2]
    rows = -(-len(frames) // columns)
    sheet = Image.new("RGB", (columns * w, rows * h))
    draw = ImageDraw.Draw(sheet)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        x, y = (i % columns) * w, (i // columns) * h
        sheet.paste(Image.fromarray(frame), (x, y))
        draw.rectangle([x, y, x + 8 * len(label) + 6, y + 14], fill=(0, 0, 0))
        draw.text((x + 3, y + 1), label, fill=(255, 255, 255))
    sheet.save(path)


def fmt(v) -> str:
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def settle_check(inner, init_id: int, wait_steps: int, say) -> None:
    """Load one initial state and report what moves while the simulator settles."""
    sim_env = inner._env
    sim_env.reset()
    sim_env.set_init_state(inner._init_states[init_id % len(inner._init_states)])
    before = object_positions(inner)
    for _ in range(wait_steps):
        sim_env.step(DUMMY_ACTION)
    after = object_positions(inner)
    moved = {n: float(np.linalg.norm(after[n] - before[n])) for n in before}
    say(f"  init state {init_id}: after {wait_steps} no-op steps")
    for name in sorted(before, key=lambda n: -moved[n]):
        flag = "  <-- MOVED" if moved[name] > MOVED else ""
        if moved[name] > MOVED or name in inner._env.obj_of_interest:
            say(f"    {name:42s} {fmt(before[name])} -> {fmt(after[name])}  {moved[name] * 100:5.1f} cm{flag}")
    if not any(m > MOVED for m in moved.values()):
        say("    nothing moved more than 1 cm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--policy-path")
    src.add_argument("--random", action="store_true")
    parser.add_argument("--suite", default="libero_spatial", choices=E.SUITES)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=3, help="initial states 0..N-1")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--rename-map", default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default=E.default_device())
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out", default="./runs/inspect_task")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    torch.manual_seed(args.seed)
    policy, preprocessor, postprocessor = E.load_policy(args)
    env, env_pre, env_post = E.make_task_env(args.suite, args.task_id, 1, args.max_steps, False)
    inner = env.envs[0]
    interest = list(inner._env.obj_of_interest)
    target, goal = interest[0], interest[-1]
    say(f"{args.suite} task {args.task_id}: {inner.task_description}")
    say(f"objects of interest: {interest} (target {target}, goal {goal})")
    say(f"policy: {'random weights' if args.random else args.policy_path}, "
        f"n_action_steps={policy.config.n_action_steps}, num_steps={policy.config.num_steps}")

    versions = []
    for module in ("mujoco", "robosuite", "lerobot"):
        try:
            versions.append(f"{module} {__import__(module).__version__}")
        except Exception:  # noqa: BLE001  (a missing version string must not stop the inspection)
            versions.append(f"{module} ?")
    say("versions: " + ", ".join(versions))

    say("\n[1] Does the scene hold still before the episode starts?")
    for k in range(args.episodes):
        settle_check(inner, k, inner.num_steps_wait, say)

    say("\n[2] What happens in an episode?")
    for k in range(args.episodes):
        track = {"frames": [], "eef": [], "grip": [], "objects": []}

        def on_step(step, raw_obs, action, terminated, track=track):
            track["frames"].append(view(np.asarray(raw_obs["pixels"]["image"][0])))
            track["eef"].append(np.array(raw_obs["robot_state"]["eef"]["pos"][0], dtype=float))
            track["grip"].append(float(action[0, -1]))
            # On the last step of a successful episode the env has already reset itself,
            # so the simulator no longer holds this episode: keep the previous reading.
            track["objects"].append(track["objects"][-1] if terminated[0] and track["objects"]
                                    else object_positions(inner))

        env.set_attr("init_state_id", [k])
        torch.manual_seed(args.seed + args.task_id)
        # Start of the episode as the rollout will see it (reset + settling), for the picture.
        raw, _ = env.reset(seed=[args.seed + k])
        start = object_positions(inner)
        save_sheet([view(np.asarray(raw["pixels"]["image"][0])), view(np.asarray(raw["pixels"]["image2"][0]))],
                   ["front", "wrist"], out_dir / f"init_{k}.png", columns=2)
        env.set_attr("init_state_id", [k])
        result = E.rollout(env, env_pre, env_post, policy, preprocessor, postprocessor, stale_steps=0,
                           stale_align="none", seeds=[args.seed + k], device=args.device, on_step=on_step)

        steps = len(track["eef"])
        eef = np.stack(track["eef"])
        pos = {n: np.stack([o[n] for o in track["objects"]]) for n in start}
        reach = np.linalg.norm(eef - pos[target], axis=1)
        lift = pos[target][:, 2] - start[target][2]
        closing = [i for i, g in enumerate(track["grip"]) if g > 0]
        say(f"  episode {k}: {'SUCCESS' if result['success'][0] else 'FAIL'} after {steps} steps")
        say(f"    {target} starts at {fmt(start[target])}, {goal} at {fmt(start[goal])}")
        say(f"    gripper closest to the target: {reach.min() * 100:.1f} cm at step {int(reach.argmin())}")
        say(f"    gripper told to close on {len(closing)} of {steps} steps"
            + (f", first at step {closing[0]}" if closing else ""))
        say(f"    target lifted at most {lift.max() * 100:.1f} cm (step {int(lift.argmax())}), "
            f"ends {np.linalg.norm(pos[target][-1, :2] - pos[goal][-1, :2]) * 100:.1f} cm from the goal "
            f"horizontally, {(pos[target][-1, 2] - pos[goal][-1, 2]) * 100:+.1f} cm in height")
        travelled = {n: float(np.linalg.norm(p - start[n], axis=1).max()) for n, p in pos.items()}
        others = [f"{n} {d * 100:.0f} cm" for n, d in sorted(travelled.items(), key=lambda x: -x[1])
                  if d > 0.02 and n != target]
        say(f"    other things that moved more than 2 cm: {', '.join(others) if others else 'none'}")
        say("    step   gripper xyz               target xyz                grip")
        for i in range(0, steps, max(1, steps // 10)):
            say(f"    {i:4d}   {fmt(eef[i])}   {fmt(pos[target][i])}   {track['grip'][i]:+.1f}")
        picks = np.linspace(0, steps - 1, min(12, steps)).astype(int)
        save_sheet([track["frames"][i] for i in picks], [f"step {i}" for i in picks], out_dir / f"episode_{k}.png")

    env.close()
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out_dir}/report.txt, init_*.png, episode_*.png")


if __name__ == "__main__":
    main()
