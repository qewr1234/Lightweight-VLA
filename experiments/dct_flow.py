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

"""DCT-coefficient-space flow matching for SmolVLA (research experiment).

Instead of denoising `chunk_size` (50) raw action tokens, the expert denoises K (e.g. 16)
DCT coefficient tokens; the final actions are reconstructed with the inverse (truncated)
orthonormal DCT. Motivation, measured on real SO-100 chunks: 95% of AC trajectory energy
lives in 2-3 coefficients, and K=16 reconstructs with worst-case error ~2 deg (below servo
repeatability), so the representation loses almost nothing while the suffix - and with it
the per-denoising-step expert cost - shrinks ~3x.

Coefficient scales are extremely imbalanced across frequencies (DC/f1 dominate), which
would wreck a flow matched against N(0,1) noise; each (frequency, dim) coefficient is
therefore z-scored with dataset statistics before flow matching (`coeff_mean/std`).

Weight shapes are untouched (the projections see the same padded dim-32 vectors, just a
shorter sequence), so a DCT policy initializes directly from `lerobot/smolvla_base`.
"""

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.smolvla.modeling_smolvla import (
    VLAFlowMatching,
    make_att_2d_masks,
)


def dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix (n x n): X = D @ x, x = D.T @ X."""
    k = np.arange(n)[:, None]
    t = np.arange(n)[None, :]
    D = np.cos(np.pi * (2 * t + 1) * k / (2 * n))
    D[0] *= np.sqrt(1.0 / n)
    D[1:] *= np.sqrt(2.0 / n)
    return D


def compute_coeff_stats(
    normalized_chunks: np.ndarray, k: int, max_action_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-(frequency, dim) mean/std of DCT coefficients over training chunks.

    normalized_chunks: (N, T, action_dim) actions already MEAN_STD-normalized.
    Returns mean/std of shape (k, max_action_dim); padded dims get mean 0 / std 1 so the
    zero-padding stays exactly zero through the (de)normalization.
    """
    n, t, dim = normalized_chunks.shape
    D = dct_matrix(t)[:k]  # (k, T)
    coeffs = np.einsum("kt,ntd->nkd", D, normalized_chunks)  # (N, k, dim)
    mean = np.zeros((k, max_action_dim))
    std = np.ones((k, max_action_dim))
    mean[:, :dim] = coeffs.mean(axis=0)
    std[:, :dim] = coeffs.std(axis=0) + 1e-6
    return mean, std


class DCTFlowMatching(VLAFlowMatching):
    """VLAFlowMatching that denoises K DCT coefficient tokens instead of chunk_size actions.

    Not constructed directly - use `convert_policy_to_dct(policy, ...)`, which swaps the
    class of an existing (pretrained-initialized) model in place and attaches the DCT
    basis + coefficient statistics as buffers.
    """

    def actions_to_coeffs(self, actions: Tensor) -> Tensor:
        """(B, T, 32) normalized actions -> (B, K, 32) z-scored DCT coefficients."""
        c = torch.einsum("kt,btd->bkd", self.dct_mat, actions.to(self.dct_mat.dtype))
        return (c - self.coeff_mean) / self.coeff_std

    def coeffs_to_actions(self, coeffs: Tensor) -> Tensor:
        """(B, K, 32) z-scored coefficients -> (B, T, 32) normalized actions."""
        c = coeffs.to(self.dct_mat.dtype) * self.coeff_std + self.coeff_mean
        return torch.einsum("kt,bkd->btd", self.dct_mat, c)

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Flow-matching loss in z-scored DCT coefficient space, (B, K, 32)."""
        coeffs = self.actions_to_coeffs(actions)

        if noise is None:
            noise = self.sample_noise(coeffs.shape, coeffs.device)
        if time is None:
            time = self.sample_time(coeffs.shape[0], coeffs.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * coeffs
        u_t = noise - coeffs

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.dct_k :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)
        losses = torch.nn.functional.mse_loss(u_t, v_t.to(dtype=u_t.dtype), reduction="none")
        # Energy weighting: in z-scored space a coefficient error contributes to the
        # reconstructed action error scaled by its std, so weighting the MSE by std^2
        # makes this objective equivalent to action-space MSE of the kept-frequency
        # component - the same thing the action-space baseline optimizes. Uniform (all
        # ones) when energy weighting is off.
        return losses * self.coeff_loss_weight

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None, **kwargs
    ) -> Tensor:
        """Denoise K coefficient tokens, then reconstruct (B, chunk_size, 32) actions."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            noise = self.sample_noise((bsize, self.dct_k, self.config.max_action_dim), device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(
                x_t=x_t,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t

        return self.coeffs_to_actions(x_t)

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        """One denoising step on K coefficient tokens."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.dct_k :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out)
        return v_t


def convert_policy_to_dct(
    policy,
    k: int,
    coeff_mean: np.ndarray,
    coeff_std: np.ndarray,
    energy_weighted: bool = False,
    action_dim: int = 6,
):
    """Swap a SmolVLAPolicy's model to DCT-coefficient flow matching, in place.

    The model keeps every weight (shapes are unchanged); only the denoising target space
    changes. `coeff_mean/std` come from `compute_coeff_stats` on the training set.
    With `energy_weighted`, the training loss is weighted by std^2 per (frequency, dim),
    normalized to mean 1 over the real action dims, which makes it equivalent to
    action-space MSE of the kept frequencies (see DCTFlowMatching.forward).
    """
    model = policy.model
    assert type(model) is VLAFlowMatching, f"expected stock VLAFlowMatching, got {type(model)}"
    model.__class__ = DCTFlowMatching
    model.dct_k = k
    t = policy.config.chunk_size
    device = next(model.parameters()).device
    model.register_buffer(
        "dct_mat", torch.tensor(dct_matrix(t)[:k], dtype=torch.float32, device=device)
    )
    model.register_buffer(
        "coeff_mean", torch.tensor(coeff_mean, dtype=torch.float32, device=device)
    )
    model.register_buffer("coeff_std", torch.tensor(coeff_std, dtype=torch.float32, device=device))
    if energy_weighted:
        w = np.asarray(coeff_std, dtype=np.float64) ** 2
        w = w / w[:, :action_dim].mean()  # padded dims (std 1) get a small, harmless weight
    else:
        w = np.ones_like(coeff_std)
    model.register_buffer(
        "coeff_loss_weight", torch.tensor(w, dtype=torch.float32, device=device)
    )
    return policy
