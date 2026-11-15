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

"""Save the first camera frames of training episodes for one task instruction.

The pictures are laid out like inspect_libero_task.py's init_<k>.png (front camera, wrist
camera), so the scene a checkpoint was trained on can be put next to the scene the
simulator starts an evaluation episode from.

    python experiments/show_dataset_start.py \\
        --task "pick up the black bowl on the ramekin and place it on the plate" \\
        --episodes 3 --out runs/inspect_task5

Writes dataset_<k>.png into --out.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from inspect_libero_task import save_sheet  # noqa: E402


def to_uint8(image) -> np.ndarray:
    """Dataset frame (CHW float in [0, 1]) as an HWC uint8 picture."""
    array = np.asarray(image, dtype=np.float32)
    return np.ascontiguousarray((array.transpose(1, 2, 0) * 255).round().clip(0, 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="lerobot/libero")
    parser.add_argument("--root", default=None, help="local dataset directory (default: download from the Hub)")
    parser.add_argument("--task", required=True, help="instruction, exactly as in the dataset")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--cameras", nargs=2, default=["observation.images.image", "observation.images.image2"])
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--out", default="./runs/inspect_task")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.dataset, root=args.root)
    wanted = args.task.strip().lower()
    matches = [int(ep["episode_index"]) for ep in meta.episodes
               if any(str(t).strip().lower() == wanted for t in ep["tasks"])]
    if not matches:
        known = sorted({str(t) for ep in meta.episodes for t in ep["tasks"]})
        raise SystemExit("no episode with that instruction. The dataset has:\n  " + "\n  ".join(known))
    chosen = matches[: args.episodes]
    print(f"{len(matches)} episodes for this instruction; showing {chosen}")

    dataset = LeRobotDataset(args.dataset, root=args.root, episodes=chosen, video_backend=args.video_backend)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    for i in range(len(dataset)):
        frame = dataset[i]
        episode = int(frame["episode_index"])
        if episode in seen or int(frame["frame_index"]) != 0:
            continue
        seen.add(episode)
        path = out_dir / f"dataset_{len(seen) - 1}.png"
        save_sheet([to_uint8(frame[c]) for c in args.cameras], ["front", "wrist"], path, columns=2)
        print(f"episode {episode}: saved {path}")
        if len(seen) == len(chosen):
            break


if __name__ == "__main__":
    main()
