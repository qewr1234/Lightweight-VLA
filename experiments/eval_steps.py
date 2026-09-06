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

"""Zero-training probe: how few Euler steps can the finetuned baseline survive?

Evaluates the action-space baseline checkpoint at num_steps in {1, 2, 4, 10} on the
held-out windows, with matching latency numbers. If 1-2 steps hold accuracy, the
deployment config gets faster for free - no distillation needed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _local_smolvla import use_local_smolvla

use_local_smolvla()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from eval_ab import (  # noqa: E402
    DIM_NAMES,
    EVAL_EPISODES,
    evaluate,
    load_variant,
    measure_latency,
    select_windows,
)
from train_ab import FPS, REPO  # noqa: E402

DEVICE = "mps"

delta = {"action": [i / FPS for i in range(50)]}
ds = LeRobotDataset(REPO, delta_timestamps=delta, episodes=EVAL_EPISODES)
windows = select_windows(ds)
print(f"eval: {len(windows)} held-out windows")

policy = load_variant("baseline", "./runs/baseline/checkpoint.pt", DEVICE)
task = ds[0]["task"]
if not task.endswith("\n"):
    task += "\n"
tok = policy.model.vlm_with_expert.processor.tokenizer
enc = tok(task, max_length=policy.config.tokenizer_max_length, truncation=True, return_tensors="pt")
lang_tokens = enc["input_ids"].to(DEVICE)
lang_masks = enc["attention_mask"].bool().to(DEVICE)

results = {}
for ns in (1, 2, 4, 10):
    r = evaluate(policy, ds, windows, lang_tokens, lang_masks, ds.meta.stats, DEVICE, num_steps=ns)
    lat = measure_latency(policy, ds, windows, lang_tokens, lang_masks, ds.meta.stats, DEVICE,
                          num_steps=ns)
    results[ns] = {**r, "latency_ms": lat * 1e3}
    print(f"steps={ns}: MAE {r['mae_overall']:.3f} | latency {lat * 1e3:.1f} ms | per-dim "
          + ", ".join(f"{n}={v:.2f}" for n, v in zip(DIM_NAMES, r["mae_per_dim"])), flush=True)

with open("./runs/eval_steps.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
print("saved ./runs/eval_steps.json")
