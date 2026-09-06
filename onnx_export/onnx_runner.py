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

"""Torch-free SmolVLA inference over the exported ONNX graphs (numpy + onnxruntime).

Host responsibilities (kept out of the graphs on purpose):
  - tokenize the task string (with a trailing "\\n") -> `lang_tokens`/`lang_masks`,
    padded to the exported language length (48). Use `tokenizers` or `transformers`.
  - normalize the state with dataset stats (MEAN_STD) and zero-pad to dim 32.
  - un-normalize the returned actions and slice to the robot's action dim.
Images go in RAW: (1, 3, 480, 640) float32 in [0, 1] - resize/rescale runs in-graph.

Example:
    runner = SmolVLAOnnx("./onnx_out", num_steps=4)
    actions = runner.predict_chunk(img0, img1, lang_tokens, lang_masks, state_norm)

Benchmark (uses random inputs):
    python onnx_runner.py --model-dir ./onnx_out --num-steps 4
On Jetson, pass TensorRT first: --providers TensorrtExecutionProvider CUDAExecutionProvider
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

CHUNK_SIZE = 50
ACTION_PAD_DIM = 32
LANG_LEN = 48
RAW_HW = (480, 640)


class SmolVLAOnnx:
    def __init__(self, model_dir: str, num_steps: int = 4, providers=None):
        providers = providers or ["CPUExecutionProvider"]
        model_dir = Path(model_dir)
        self.sess_prefill = ort.InferenceSession(str(model_dir / "prefill.onnx"), providers=providers)
        self.sess_denoise = ort.InferenceSession(str(model_dir / "denoise.onnx"), providers=providers)
        self.cache_names = [o.name for o in self.sess_prefill.get_outputs()]
        self.num_steps = num_steps

    def predict_chunk(self, img0, img1, lang_tokens, lang_masks, state_norm, noise=None):
        """Returns (1, 50, 32) normalized padded actions; slice/unnormalize on the host."""
        cache = dict(zip(self.cache_names, self.sess_prefill.run(None, {
            "img0": img0.astype(np.float32),
            "img1": img1.astype(np.float32),
            "lang_tokens": lang_tokens.astype(np.int64),
            "lang_masks": lang_masks.astype(bool),
            "state": state_norm.astype(np.float32),
        })))

        if noise is None:
            noise = np.random.randn(1, CHUNK_SIZE, ACTION_PAD_DIM)
        x_t = noise.astype(np.float32)
        dt = -1.0 / self.num_steps
        for step in range(self.num_steps):
            t = 1.0 + step * dt
            (v_t,) = self.sess_denoise.run(None, {
                "x_t": x_t, "timestep": np.array([t], dtype=np.float32), **cache,
            })
            x_t = (x_t + dt * v_t).astype(np.float32)
        return x_t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="./onnx_out")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--providers", nargs="+", default=None,
                        help="e.g. TensorrtExecutionProvider CUDAExecutionProvider")
    args = parser.parse_args()

    runner = SmolVLAOnnx(args.model_dir, num_steps=args.num_steps, providers=args.providers)
    print("providers:", runner.sess_prefill.get_providers())

    rng = np.random.default_rng(0)
    img0 = rng.random((1, 3, *RAW_HW), dtype=np.float32)
    img1 = rng.random((1, 3, *RAW_HW), dtype=np.float32)
    lang_tokens = rng.integers(10, 1000, (1, LANG_LEN))
    lang_masks = np.zeros((1, LANG_LEN), dtype=bool)
    lang_masks[:, :20] = True
    state = np.zeros((1, ACTION_PAD_DIM), dtype=np.float32)

    runner.predict_chunk(img0, img1, lang_tokens, lang_masks, state)  # warmup
    times = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        runner.predict_chunk(img0, img1, lang_tokens, lang_masks, state)
        times.append(time.perf_counter() - t0)
    times.sort()
    median = times[len(times) // 2]
    print(f"chunk latency: median {median * 1e3:.1f} ms "
          f"(open-loop replay rate {CHUNK_SIZE / median:.1f} Hz; closed-loop rate depends "
          f"on the execution model - measure it with lerobot async inference)")


if __name__ == "__main__":
    main()
