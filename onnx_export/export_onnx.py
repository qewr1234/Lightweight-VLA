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

"""Export SmolVLA as two ONNX graphs (prefill + denoise-step) for TensorRT/ORT deployment.

Graph split mirrors the runtime structure:
  A `prefill.onnx`  : raw images + language tokens/mask + normalized state
                      -> per-layer VLM prefix KV cache (+ prefix pad mask)
  B `denoise.onnx`  : x_t + timestep + prefix KV cache -> v_t
The host runs A once per chunk, then B x num_steps with Euler integration (see
onnx_runner.py). Everything is exported static-shape (batch 1, fixed cameras/resolution/
language length) - the friendliest layout for TensorRT.

Notes
- optimum has no Idefics3/SmolVLM support, hence this custom export.
- Exported in fp32; convert precision at runtime (`trtexec --fp16`/`--best`).
- The timestep sinusoid uses float64 in stock code; it is patched to float32 during
  export (max error ~1e-7) because TensorRT rejects f64.

Usage (iterate with random weights, final with the real checkpoint):
    python export_onnx.py --random --verify
    python export_onnx.py --policy-path lerobot/smolvla_base --verify --out ./onnx_out
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _local_smolvla import use_local_smolvla

use_local_smolvla()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

import lerobot.policies.smolvla.modeling_smolvla as modeling  # noqa: E402
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402
from lerobot.policies.smolvla.modeling_smolvla import (  # noqa: E402
    SmolVLAPolicy,
    resize_with_pad,
)


def make_att_2d_masks_onnx(pad_masks, att_masks):
    """ONNX-safe make_att_2d_masks: ONNX CumSum/Mul reject bool, so cast first.

    Identical semantics to modeling_smolvla.make_att_2d_masks (bool & replaces bool *).
    """
    cumsum = torch.cumsum(att_masks.to(torch.int32), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_b = pad_masks.to(torch.bool)
    pad_2d_masks = pad_b[:, None, :] & pad_b[:, :, None]
    return att_2d_masks & pad_2d_masks
from lerobot.utils.constants import OBS_STATE  # noqa: E402

CAMERAS = 2
RAW_HW = (480, 640)
LANG_LEN = 48
STATE_DIM = 6


class PrefillGraph(nn.Module):
    """images (raw [0,1]) + language + normalized state -> prefix KV cache."""

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__()
        self.model = policy.model
        self.num_layers = policy.model.vlm_with_expert.num_vlm_layers
        self.resolution = policy.config.resize_imgs_with_padding

    def forward(self, img0, img1, lang_tokens, lang_masks, state):
        images = []
        img_masks = []
        for img in (img0, img1):
            img = resize_with_pad(img, *self.resolution, pad_value=0)
            img = img * 2.0 - 1.0  # [0,1] -> [-1,1] as SigLIP expects
            images.append(img)
            img_masks.append(torch.ones(img.shape[0], dtype=torch.bool, device=img.device))

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks_onnx(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1
        _, past = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        outputs = [prefix_pad_masks]
        for i in range(self.num_layers):
            outputs.append(past[i]["key_states"])
            outputs.append(past[i]["value_states"])
        return tuple(outputs)


class DenoiseGraph(nn.Module):
    """One Euler step: x_t + timestep + prefix KV -> v_t."""

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__()
        self.model = policy.model
        self.chunk_size = policy.config.chunk_size
        self.num_layers = policy.model.vlm_with_expert.num_vlm_layers
        # Expert KV projections are recomputed each call inside the graph; the
        # cross-step python cache does not exist across ONNX calls.
        policy.model.vlm_with_expert.cache_expert_kv_projections = False

    def forward(self, x_t, timestep, prefix_pad_masks, *kv_flat):
        past = {}
        for i in range(self.num_layers):
            past[i] = {"key_states": kv_flat[2 * i], "value_states": kv_flat[2 * i + 1]}

        suffix_embs, suffix_pad_masks, suffix_att_masks = self.model.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks_onnx(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks.to(torch.bool), suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks.to(torch.int64), dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks.to(torch.int64), dim=1) - 1

        outputs_embeds, _ = self.model.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1][:, -self.chunk_size :]
        suffix_out = suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
        return self.model.action_out_proj(suffix_out)


def patch_vision_embeddings_static(policy: SmolVLAPolicy) -> None:
    """Replace SmolVLM vision embeddings' variable-resolution position-id scatter with a
    baked constant.

    Stock code supports variable resolutions via a boolean-mask scatter
    (`position_ids[mask] = pos_ids[mask]`), which exports as a ScatterND that
    onnxruntime rejects (type mismatch). Our images are always full square
    `resize_imgs_with_padding` crops with an all-True patch mask, where the position ids
    are the deterministic full-grid bucketization - so we precompute them once with the
    ORIGINAL formula and bake them as a constant. Numerically identical for this export.
    """
    vision = policy.model.vlm_with_expert.get_vlm_model().vision_model
    emb = vision.embeddings
    side_px = policy.config.resize_imgs_with_padding[0]
    nb = side_px // emb.patch_size  # patches per side at export resolution

    # Original bucketization for a full all-True mask.
    boundaries = torch.arange(1 / emb.num_patches_per_side, 1.0, 1 / emb.num_patches_per_side)
    frac = torch.clamp(torch.arange(nb, dtype=torch.float32) / nb, max=1.0 - 1e-6)
    bucket = torch.bucketize(frac, boundaries, right=True)
    position_ids = (bucket[:, None] * emb.num_patches_per_side + bucket[None, :]).reshape(1, -1)
    emb.register_buffer("static_position_ids", position_ids, persistent=False)

    def static_forward(pixel_values, patch_attention_mask):
        patch_embeds = emb.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        pos = emb.static_position_ids.expand(pixel_values.shape[0], -1)
        return embeddings + emb.position_embedding(pos)

    emb.forward = static_forward


def build_policy(args) -> SmolVLAPolicy:
    if args.random:
        input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,))}
        for i in range(CAMERAS):
            input_features[f"observation.images.cam{i}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, *RAW_HW)
            )
        torch.manual_seed(0)
        policy = SmolVLAPolicy(SmolVLAConfig(
            input_features=input_features,
            output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,))},
            load_vlm_weights=False,
        ))
    else:
        policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.eval()
    policy.config.num_steps = args.num_steps
    # Export in fp32 on CPU: the VLM loads in bf16 (and from_pretrained may place it on
    # mps/cuda), but ORT CPU lacks bf16 kernels, tracing needs one device, and TensorRT
    # prefers an fp32 graph with runtime precision flags (trtexec --fp16/--best).
    policy.to("cpu")
    policy.model = policy.model.to(torch.float32)
    return policy


def example_inputs(policy):
    g = torch.Generator().manual_seed(7)
    img0 = torch.rand(1, 3, *RAW_HW, generator=g)
    img1 = torch.rand(1, 3, *RAW_HW, generator=g)
    lang_tokens = torch.randint(10, 1000, (1, LANG_LEN), generator=g)
    lang_masks = torch.zeros(1, LANG_LEN, dtype=torch.bool)
    lang_masks[:, :20] = True  # exercise the padded-language path
    state = torch.zeros(1, policy.config.max_state_dim)
    state[:, :STATE_DIM] = torch.rand(1, STATE_DIM, generator=g)
    x_t = torch.randn(1, policy.config.chunk_size, policy.config.max_action_dim, generator=g)
    timestep = torch.tensor([1.0], dtype=torch.float32)
    return img0, img1, lang_tokens, lang_masks, state, x_t, timestep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", default="lerobot/smolvla_base")
    parser.add_argument("--random", action="store_true", help="random weights (fast iteration)")
    parser.add_argument("--out", default="./onnx_out")
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true", help="check ONNX vs PyTorch equivalence")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # TensorRT rejects float64: run the timestep sinusoid in float32 during export
    # (max abs error ~1e-7 vs stock).
    modeling.get_safe_dtype = lambda dtype, device_type: torch.float32

    policy = build_policy(args)
    patch_vision_embeddings_static(policy)
    num_layers = policy.model.vlm_with_expert.num_vlm_layers
    img0, img1, lang_tokens, lang_masks, state, x_t, timestep = example_inputs(policy)

    print("Tracing prefill graph ...")
    prefill = PrefillGraph(policy)
    with torch.no_grad():
        prefill_out = prefill(img0, img1, lang_tokens, lang_masks, state)
    prefix_pad_masks = prefill_out[0]
    kv_flat = prefill_out[1:]
    print(f"  prefix len = {prefix_pad_masks.shape[1]}, kv tensors = {len(kv_flat)}, "
          f"kv shape = {tuple(kv_flat[0].shape)}")

    kv_names = [f"kv_{i}_{kind}" for i in range(num_layers) for kind in ("k", "v")]
    torch.onnx.export(
        prefill,
        (img0, img1, lang_tokens, lang_masks, state),
        str(out / "prefill.onnx"),
        input_names=["img0", "img1", "lang_tokens", "lang_masks", "state"],
        output_names=["prefix_pad_masks", *kv_names],
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"  saved {out / 'prefill.onnx'}")

    print("Tracing denoise graph ...")
    denoise = DenoiseGraph(policy)
    with torch.no_grad():
        v_t = denoise(x_t, timestep, prefix_pad_masks, *kv_flat)
    print(f"  v_t shape = {tuple(v_t.shape)}")

    torch.onnx.export(
        denoise,
        (x_t, timestep, prefix_pad_masks, *kv_flat),
        str(out / "denoise.onnx"),
        input_names=["x_t", "timestep", "prefix_pad_masks", *kv_names],
        output_names=["v_t"],
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"  saved {out / 'denoise.onnx'}")

    if args.verify:
        verify(policy, out, (img0, img1, lang_tokens, lang_masks, state), x_t, args.num_steps)


def verify(policy, out_dir: Path, prefill_inputs, noise, num_steps: int) -> None:
    """End-to-end: ONNX host loop vs PyTorch predict, same noise -> actions must match."""
    import onnxruntime as ort

    img0, img1, lang_tokens, lang_masks, state = prefill_inputs

    print("Verifying against PyTorch ...")
    sess_a = ort.InferenceSession(str(out_dir / "prefill.onnx"), providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(str(out_dir / "denoise.onnx"), providers=["CPUExecutionProvider"])

    feed_a = {
        "img0": img0.numpy(), "img1": img1.numpy(),
        "lang_tokens": lang_tokens.numpy(), "lang_masks": lang_masks.numpy(),
        "state": state.numpy(),
    }
    t0 = time.perf_counter()
    outs = sess_a.run(None, feed_a)
    t_prefill = time.perf_counter() - t0
    names = [o.name for o in sess_a.get_outputs()]
    cache = dict(zip(names, outs))

    x_t = noise.numpy().copy()
    dt = -1.0 / num_steps
    t_denoise = 0.0
    for step in range(num_steps):
        t = 1.0 + step * dt
        feed_b = {"x_t": x_t.astype(np.float32), "timestep": np.array([t], dtype=np.float32), **cache}
        t0 = time.perf_counter()
        (v_t,) = sess_b.run(None, feed_b)
        t_denoise += time.perf_counter() - t0
        x_t = x_t + dt * v_t
    onnx_actions = x_t

    with torch.no_grad():
        torch_actions = policy.model.sample_actions(
            [resize_with_pad(i, *policy.config.resize_imgs_with_padding, pad_value=0) * 2.0 - 1.0
             for i in (img0, img1)],
            [torch.ones(1, dtype=torch.bool), torch.ones(1, dtype=torch.bool)],
            lang_tokens, lang_masks, state, noise=noise.clone(),
        ).numpy()

    diff = np.abs(onnx_actions - torch_actions).max()
    status = "PASS" if diff < 1e-3 else "FAIL"
    print(f"[{status}] ONNX vs PyTorch actions max|diff| = {diff:.2e}")
    print(f"ORT CPU latency: prefill {t_prefill * 1e3:.0f} ms + "
          f"{num_steps} denoise steps {t_denoise * 1e3:.0f} ms "
          f"= {(t_prefill + t_denoise) * 1e3:.0f} ms/chunk (Mac CPU, informational only)")
    if status == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
