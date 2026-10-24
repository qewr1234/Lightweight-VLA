#!/usr/bin/env python
"""Fast vision path for the action thread (dual-rate SmolVLA).

The async pipeline splits inference into a slow thread (SigLIP + VLM prefill -> prefix KV
cache, ~260 ms on Jetson Orin Nano) and a fast thread (denoising against that cache). The
action thread therefore always acts on a stale picture of the world: a cache is built from a
frame captured one vision period before it is published, and is then used until the next one
is published, so its age runs from one to two vision periods (~260-520 ms).

This module gives the action thread its own fresh, cheap view:

    slow (unchanged)   front/top cams @512 -> SigLIP -> VLM prefill -> PrefixCache   ~4 Hz
    fast (new)         one cam @96x128 -> ResNet-18 trunk -> 48 tokens  ┐
                       current joint state -> 1 token                    ├─ gated cross-attn
                       noisy action tokens (after the timestep MLP) ─────┘   -> action expert

Design choices, and why:

* Injection point - the suffix embedding, right after SmolVLA fuses the timestep into the
  action tokens, through a Flamingo-style gated cross-attention block. Action tokens query
  the fast tokens; nothing else in the prefix/suffix attention pattern changes, so the
  cached prefix, posmap, RTC and the expert KV-projection cache all keep working.
* Exact identity at initialization - both residual gates start at 0 (tanh(0) = 0), so an
  attached-but-untrained model produces bit-identical actions to the original. Training
  starts from the pretrained policy and only learns to use the new input as far as it
  helps. (Appending raw tokens to the sequence would not have this property: even all-zero
  keys take softmax mass from the real ones.)
* Small trunk - ResNet-18 cut after layer3 (stride 16, ~2.8M params) at 96x128 keeps the 3:4
  aspect of 480x640 and gives a 6x8 token grid. Frozen BatchNorm, because batch statistics
  from a robot-learning batch of 8 are noise.
* Fresh proprioception - the prefix's state token is as stale as its images; one projected
  state token in the fast set gives the expert the current joint positions.
* Computed once per chunk - the fast tokens are encoded once per `sample_from_prefix` call
  and reused across all denoising steps.

Training must simulate the staleness, or the model learns to ignore the fast path: if both
paths see the same frame, the fast one carries no extra information. `apply_staleness`
picks, per sample, a slow-path frame `d` steps in the past (d drawn from the deployment age
range) while the fast path and the action targets stay at time t.

Usage:
    from fast_vision import FastVisionConfig, attach_fast_vision
    attach_fast_vision(policy, FastVisionConfig(camera_key="observation.images.wrist"))

    prefix = policy.encode_slow(batch)          # vision thread, every ~260 ms
    actions = policy.act_fast(prefix, batch)    # action thread, every chunk

Saving: the new modules live in `policy.model`, so `save_pretrained` writes their weights,
but `config.json` does not describe them. To reload, build the policy, call
`attach_fast_vision` with the same `FastVisionConfig`, then load the state dict.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.policies.smolvla.modeling_smolvla import (
    PrefixCache,
    SmolVLAPolicy,
    VLAFlowMatching,
    pad_vector,
)
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

# Batch keys written by `apply_staleness`; read by the policy before the camera/state keys.
FAST_IMAGE = "fast_vision.image"
FAST_STATE = "fast_vision.state"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class FastVisionConfig:
    camera_key: str  # camera feeding the fast path; may also be one of the slow cameras
    image_hw: tuple[int, int] = (96, 128)  # 3:4 like 480x640
    trunk: str = "layer3"  # "layer3": stride 16, 256 ch, 6x8 tokens | "layer4": stride 32, 512 ch
    use_state: bool = True  # append a fresh proprioception token
    num_heads: int = 8
    pretrained_backbone: bool = True  # ImageNet ResNet-18 (torchvision download)


class FastVisionEncoder(nn.Module):
    """(B, 3, H, W) image in [0, 1] -> (B, h*w, dim) tokens."""

    def __init__(self, cfg: FastVisionConfig, dim: int):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        from torchvision.ops.misc import FrozenBatchNorm2d

        if cfg.trunk not in ("layer3", "layer4"):
            raise ValueError(f"trunk must be 'layer3' or 'layer4', got {cfg.trunk!r}")
        weights = ResNet18_Weights.IMAGENET1K_V1 if cfg.pretrained_backbone else None
        net = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)
        layers = [net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3]
        channels, stride = 256, 16
        if cfg.trunk == "layer4":
            layers.append(net.layer4)
            channels, stride = 512, 32
        self.trunk = nn.Sequential(*layers)

        h, w = cfg.image_hw
        if h % stride or w % stride:
            raise ValueError(f"image_hw {cfg.image_hw} must be divisible by the trunk stride {stride}")
        self.image_hw = (h, w)
        self.num_tokens = (h // stride) * (w // stride)
        self.proj = nn.Linear(channels, dim)
        self.pos = nn.Parameter(torch.randn(1, self.num_tokens, dim) * 0.02)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: Tensor) -> Tensor:
        x = F.interpolate(image.float(), size=self.image_hw, mode="bilinear", antialias=True, align_corners=False)
        x = ((x - self.mean.float()) / self.std.float()).to(self.proj.weight.dtype)
        feats = self.trunk(x)  # (B, C, h, w)
        tokens = feats.flatten(2).transpose(1, 2)  # (B, h*w, C)
        return self.proj(tokens) + self.pos


class GatedCrossAttention(nn.Module):
    """x + tanh(a) * Attn(x, ctx), then x + tanh(b) * MLP(x), with a = b = 0 at init.

    An exact identity until training moves the gates (Flamingo's gated cross-attention).
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_ctx = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.ff_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, ctx: Tensor) -> Tensor:
        ctx = self.norm_ctx(ctx)
        attn_out, _ = self.attn(self.norm_q(x), ctx, ctx, need_weights=False)
        x = x + torch.tanh(self.attn_gate) * attn_out
        return x + torch.tanh(self.ff_gate) * self.ff(self.norm_ff(x))


class FastVisionFlowMatching(VLAFlowMatching):
    """VLAFlowMatching whose action tokens also cross-attend to fresh fast-path tokens.

    Not constructed directly - `attach_fast_vision` swaps the class of an existing model in
    place and adds `fast_encoder`, `fast_state_proj` and `fast_adapter`.

    The fast tokens reach `embed_suffix` through `_fast_tokens`, set by `fast_conditioning`
    for the duration of one forward or one chunk. Every parent code path (training forward,
    sample_actions, RTC, posmap) therefore picks them up without being duplicated here. One
    model instance must not run two conditioned calls concurrently - in the async pipeline
    only the action thread conditions; the vision thread only calls `encode_prefix`.
    """

    _fast_tokens: Tensor | None = None

    def encode_fast(self, fast_image: Tensor, fast_state: Tensor | None = None) -> Tensor:
        tokens = self.fast_encoder(fast_image)
        if self.fast_state_proj is not None and fast_state is not None:
            state = pad_vector(fast_state, self.config.max_state_dim)
            state_token = self.fast_state_proj(state.to(self.fast_state_proj.weight.dtype))
            tokens = torch.cat([tokens, state_token[:, None]], dim=1)
        return tokens

    @contextmanager
    def fast_conditioning(self, fast_image: Tensor | None, fast_state: Tensor | None = None):
        """Condition every suffix embedding inside the block on these fast inputs."""
        if fast_image is None:  # nothing to add: leave any outer conditioning in place
            yield
            return
        previous = self._fast_tokens
        self._fast_tokens = self.encode_fast(fast_image, fast_state)
        try:
            yield
        finally:
            self._fast_tokens = previous

    def embed_suffix(self, noisy_actions, timestep):
        embs, pad_masks, att_masks = super().embed_suffix(noisy_actions, timestep)
        if self._fast_tokens is not None:
            embs = self.fast_adapter(embs, self._fast_tokens.to(embs.dtype))
        return embs, pad_masks, att_masks

    def forward(self, *args, fast_image=None, fast_state=None, **kwargs):
        with self.fast_conditioning(fast_image, fast_state):
            return super().forward(*args, **kwargs)

    def sample_from_prefix(self, prefix: PrefixCache, noise=None, fast_image=None, fast_state=None, **kwargs):
        with self.fast_conditioning(fast_image, fast_state):
            return super().sample_from_prefix(prefix, noise=noise, **kwargs)


class FastVisionSmolVLAPolicy(SmolVLAPolicy):
    """SmolVLAPolicy that feeds the fast path, plus the two halves of async inference."""

    def _fast_inputs(self, batch: dict[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        """Fresh fast inputs: `apply_staleness` outputs if present, else the live batch."""
        image = batch.get(FAST_IMAGE, batch.get(self.fast_cfg.camera_key))
        if image is not None and image.ndim == 5:  # (B, T, C, H, W) from observation queues
            image = image[:, -1]
        state = None
        if self.fast_cfg.use_state:
            state = batch.get(FAST_STATE, batch.get(OBS_STATE))
            if state is not None and state.ndim == 3:
                state = state[:, -1]
        return image, state

    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        with self.model.fast_conditioning(*self._fast_inputs(batch)):
            return super().forward(batch, noise=noise, time=time, reduction=reduction)

    def _get_action_chunk(self, batch, noise=None, **kwargs):
        with self.model.fast_conditioning(*self._fast_inputs(batch)):
            return super()._get_action_chunk(batch, noise, **kwargs)

    @torch.no_grad()
    def encode_slow(self, batch: dict[str, Tensor]) -> PrefixCache:
        """Vision thread: cameras + language + state -> reusable prefix cache."""
        self.eval()
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        return self.model.encode_prefix(
            images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], state
        )

    @torch.no_grad()
    def act_fast(self, prefix: PrefixCache, batch: dict[str, Tensor], noise=None, **kwargs) -> Tensor:
        """Action thread: one chunk from the latest prefix + this tick's fast inputs."""
        self.eval()
        fast_image, fast_state = self._fast_inputs(batch)
        actions = self.model.sample_from_prefix(
            prefix, noise=noise, fast_image=fast_image, fast_state=fast_state, **kwargs
        )
        return actions[:, :, : self.config.action_feature.shape[0]]


def attach_fast_vision(policy: SmolVLAPolicy, cfg: FastVisionConfig) -> FastVisionSmolVLAPolicy:
    """Add the fast path to an existing (pretrained-initialized) policy, in place.

    The result is bit-identical to `policy` until the adapter gates are trained.
    """
    if policy.config.adapt_to_pi_aloha:
        raise NotImplementedError("fast vision does not apply the pi-aloha state transform")
    model = policy.model
    dim = model.vlm_with_expert.expert_hidden_size
    ref = model.action_in_proj.weight

    model.fast_cfg = cfg
    model.fast_encoder = FastVisionEncoder(cfg, dim).to(device=ref.device, dtype=ref.dtype)
    model.fast_state_proj = (
        nn.Linear(policy.config.max_state_dim, dim).to(device=ref.device, dtype=ref.dtype)
        if cfg.use_state
        else None
    )
    model.fast_adapter = GatedCrossAttention(dim, cfg.num_heads).to(device=ref.device, dtype=ref.dtype)
    model.__class__ = FastVisionFlowMatching
    model._fast_tokens = None

    policy.fast_cfg = cfg
    policy.__class__ = FastVisionSmolVLAPolicy
    return policy


# --------------------------------------------------------------------------------------
# Training-time staleness simulation
# --------------------------------------------------------------------------------------


def stale_frame_offsets(vision_period_s: float, fps: int, num: int = 5) -> list[int]:
    """Frame offsets covering the prefix cache's age at use time.

    A cache is built from a frame captured when the vision cycle starts, published one
    period later, and replaced one period after that - so its age when the action thread
    reads it spans [period, 2 * period]. 0.26 s at 30 fps -> 8..16 frames.
    """
    lo = max(1, round(vision_period_s * fps))
    hi = max(lo, round(2 * vision_period_s * fps))
    return sorted({int(x) for x in np.linspace(lo, hi, num).round()})


def staleness_delta_timestamps(
    fps: int, offsets: list[int], slow_keys: list[str], fast_key: str, state_key: str = OBS_STATE
) -> dict[str, list[float]]:
    """`delta_timestamps` entries for LeRobotDataset: index 0 is time t, index i >= 1 is
    t - offsets[i-1] frames. Merge with the action entry. Frames before an episode start are
    clamped to its first frame by LeRobotDataset (the robot is still there)."""
    stamps = [0.0] + [-o / fps for o in offsets]
    delta = {key: stamps for key in slow_keys}
    delta[state_key] = stamps
    delta.setdefault(fast_key, [0.0])
    return delta


def apply_staleness(
    batch: dict[str, Tensor],
    slow_keys: list[str],
    fast_key: str,
    state_key: str = OBS_STATE,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Split a batch fetched with `staleness_delta_timestamps` into stale slow inputs (one
    random past frame per sample, shared by all slow cameras and the state) and fresh fast
    inputs (`FAST_IMAGE`, `FAST_STATE` at time t). Action targets are untouched (time t)."""
    stacked_state = batch[state_key]
    bsize, n_frames = stacked_state.shape[:2]
    if n_frames < 2:
        raise ValueError(f"{state_key} has no past frames - fetch with staleness_delta_timestamps")
    pick = torch.randint(1, n_frames, (bsize,), generator=generator)
    rows = torch.arange(bsize)

    out = dict(batch)
    fast = batch[fast_key]
    out[FAST_IMAGE] = fast[:, 0] if fast.ndim == 5 else fast
    out[FAST_STATE] = stacked_state[:, 0]
    for key in slow_keys:
        out[key] = batch[key][rows, pick]
    out[state_key] = stacked_state[rows, pick]
    out["fast_vision.stale_index"] = pick  # 1-based index into the offsets, for logging
    return out
