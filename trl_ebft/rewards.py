# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

"""
Reward computation for Energy-Based Fine-Tuning (EBFT).

Implements the exact per-paper feature-matching rewards (REINFORCE-style estimator)
used to optimize the conditional feature-matching loss L_CFM.

Paper: https://huggingface.co/papers/2603.12248
See especially:
- Eq (6), (7): base reward r_j = 2 φ_j^T φ_GT - 2/(n-1) Σ φ_j^T φ_{j'}
- Eq (8), (9): whitened variant (normalize *only* the alignment term; keep whitened diversity)
- Algorithm 1 + Section 2.3 (whitening), Section E (RLOO baseline details)

This module is deliberately pure-tensor and self-contained (no trainer state).
It can be tested in isolation. The actual EBFTTrainer (Phase 2) will call these
and duplicate small pieces of orchestration logic per TRL self-contained-trainer rule.

Also re-exports feature helpers for convenience.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .feature_extractor import (
    EBFTFeatureExtractor,
    apply_embed_method,
    extract_hidden_states,
    get_layer_indices_from_fractions,
)

__all__ = [
    "compute_feature_matching_rewards",
    "whiten_embeddings_batched",
    "get_alignment_rewards",
    "get_diversity_rewards",
    # re-export core extractors
    "EBFTFeatureExtractor",
    "extract_hidden_states",
    "apply_embed_method",
    "get_layer_indices_from_fractions",
]


@torch.no_grad()
def get_alignment_rewards(
    gen_embedding: torch.Tensor,
    gt_embedding: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Alignment term: how well generated features match the ground-truth feature.

    Paper base form (non-whitened): φ_j ^T φ_GT   (after per-block norm)
    Whitened form (eq 9): cosine( ~φ_j , ~φ_GT )   i.e. normalized dot only for alignment.

    Args:
        gen_embedding (`torch.Tensor` of shape `(B, D)`):
        gt_embedding (`torch.Tensor` of shape `(B, D)` or broadcastable):
        normalize (`bool`, defaults to `True`):
            If True, use cosine similarity (normalized). Used for whitened alignment.

    Returns:
        `torch.Tensor` of shape `(B,)` — raw alignment scores (not yet *2).
    """
    if normalize:
        return F.cosine_similarity(gen_embedding, gt_embedding, dim=-1)
    else:
        # Raw dot product (paper primary when no whitening)
        return (gen_embedding * gt_embedding).sum(dim=-1)


@torch.no_grad()
def get_diversity_rewards(
    gen_embedding: torch.Tensor,
    num_generations: int,
) -> torch.Tensor:
    """
    Diversity penalty (anti-collapse term).

    For each sample, mean dot-product similarity to the *other* (n-1) samples
    from the same prompt/group. Self-similarity is excluded.

    Used (with sign flip) in both paper eq (7) and whitened eq (9).
    In whitened case, the *whitened* (non-renormalized) vectors are used here
    so that full whitened geometry is retained for the diversity term.

    Args:
        gen_embedding (`torch.Tensor` of shape `(B, D)`):
            B must be num_prompts * num_generations, contiguous per prompt.
        num_generations (`int`): n in paper.

    Returns:
        `torch.Tensor` of shape `(B,)` — mean pairwise similarities (not yet scaled).
    """
    if num_generations <= 1:
        return torch.zeros(gen_embedding.shape[0], device=gen_embedding.device, dtype=gen_embedding.dtype)

    num_prompts = gen_embedding.shape[0] // num_generations
    reshaped = gen_embedding.view(num_prompts, num_generations, -1)  # (P, n, D)

    # (P, n, n) pairwise dots
    sims = torch.bmm(reshaped, reshaped.transpose(1, 2))

    # Mask out self
    eye = torch.eye(num_generations, device=sims.device, dtype=torch.bool)
    sims = sims.masked_fill(eye.unsqueeze(0), 0.0)

    diversity = sims.sum(dim=-1) / (num_generations - 1)  # (P, n)
    return diversity.view(-1)


def whiten_embeddings_batched(
    phi: torch.Tensor,
    phi_gt: torch.Tensor,
    whiten_tol: float = 1e-5,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Whitening transform estimated from the generated samples of one group.

    Implements paper eq (8):  ~φ = (Σ̂†)^{1/2} φ
    Σ̂ estimated from the n rollouts only (per context).

    Uses SVD on the feature matrix for numerical stability + pseudo-inverse.
    Singular values are scaled such that small ones are zeroed.

    Important (per paper Sec 2.3 + appendix):
    - Apply the *same* W to both rollouts and the GT.
    - Later, only alignment is renormalized (cosine); diversity keeps raw whitened dots.

    Args:
        phi (`torch.Tensor` of shape `(n, D)`): rollouts for one prompt (used to fit Σ̂).
        phi_gt (`torch.Tensor` of shape `(1 or n, D)`): corresponding GT (will be broadcast).
        whiten_tol (`float`):
            Relative cutoff for singular values (tol * s_max).
        normalize (`bool`):
            If True, additionally L2-normalize after whitening (rarely wanted; see paper).

    Returns:
        (phi_whitened, phi_gt_whitened) both same dtype as input, on same device.
    """
    phi_f = phi.float()
    phi_gt_f = phi_gt.float()

    n_samples = phi_f.shape[0]
    if n_samples < 2:
        # Degenerate: return normalized or raw
        if normalize:
            return F.normalize(phi, p=2, dim=-1), F.normalize(phi_gt, p=2, dim=-1)
        return phi, phi_gt

    try:
        # SVD on (D, n) view: U (D, k), S (k,)
        U, S, _ = torch.linalg.svd(phi_f.T.unsqueeze(0), full_matrices=False)
    except torch.linalg.LinAlgError:  # type: ignore[attr-defined]
        # Add tiny noise and retry
        noise = 1e-6 * (phi_f.abs().mean() + 1e-12)
        try:
            U, S, _ = torch.linalg.svd(
                (phi_f.T + noise * torch.randn_like(phi_f.T)).unsqueeze(0),
                full_matrices=False,
            )
        except torch.linalg.LinAlgError:  # type: ignore[attr-defined]
            if normalize:
                return F.normalize(phi, p=2, dim=-1), F.normalize(phi_gt, p=2, dim=-1)
            return phi, phi_gt

    U = U.squeeze(0)  # (D, k)
    S = S.squeeze(0)  # (k,)

    s_max = S.max().clamp(min=1e-12)
    # Safe pseudo-inverse of singular values
    inv_s = torch.where(S > (whiten_tol * s_max), 1.0 / (S + 1e-12), torch.zeros_like(S))

    # Whitening matrix W = U @ diag(inv_s) @ U^T   (D, D)
    W = (U * inv_s.unsqueeze(0)) @ U.T
    phi_w = (phi_f @ W).to(phi.dtype)
    phi_gt_w = (phi_gt_f @ W).to(phi_gt.dtype)

    if normalize:
        phi_w = F.normalize(phi_w, p=2, dim=-1)
        phi_gt_w = F.normalize(phi_gt_w, p=2, dim=-1)

    return phi_w, phi_gt_w


@torch.no_grad()
def compute_feature_matching_rewards(
    features_samples: torch.Tensor,
    features_gt: torch.Tensor,
    num_generations: int,
    whitening: bool = True,
    whiten_tol: float = 1e-5,
    alignment_coef: float = 1.0,
    diversity_coef: float = 1.0,
    return_diagnostics: bool = False,
) -> dict[str, Any]:
    """
    Core EBFT reward function.

    Given per-group features, computes the unbiased estimator of the gradient
    of L_CFM (paper eq 7, whitened variant eq 9).

    Exactly:
        r_j = 2 * align_j  -  2/(n-1) * sum_{j'≠j} dot_jj'
        (with align being raw dot or normalized-cosine depending on whitening)

    Then final reward passed to policy gradient = alignment_coef * align_scaled
                                                 - diversity_coef * diversity_scaled

    The trainer will later subtract RLOO baseline b_j (leave-one-out mean of r).

    Args:
        features_samples (`torch.Tensor` of shape `(P * n, D)`):
            Embeddings for all generated samples. Grouped by prompt: first n for prompt 0, etc.
        features_gt (`torch.Tensor` of shape `(P, D)` or `(P * n, D)`):
            Ground-truth embedding per prompt. Will be repeated internally as needed.
        num_generations (`int`): n (samples per prompt / group size).
        whitening (`bool`, defaults to `True`):
            Whether to apply per-group whitening (recommended for stability on real features).
        whiten_tol (`float`):
            Tolerance for SVD in whitening.
        alignment_coef (`float`):
            Scale on alignment term (paper uses 1.0; <1.0 biases toward diversity).
        diversity_coef (`float`):
            Scale on (positive) diversity penalty.
        return_diagnostics (`bool`):
            If True, also return raw alignment/diversity/cfm_proxy etc for logging.

    Returns:
        `dict` with at minimum:
            - "rewards": `torch.Tensor` (P*n,) — the r_j (to be used as reward signal)
            - "baselines" (optional in future): leave-one-out baselines can be computed outside
        When return_diagnostics=True also includes:
            - "alignment", "diversity", "cfm_proxy", "rewards_raw" etc.

    Notes:
        - Whitening is performed **per prompt group** using only the generated samples
          to estimate Σ̂ (paper).
        - For whitened: alignment uses cosine on whitened vectors; diversity uses raw dots on whitened.
        - For non-whitened: both use (block-normed) dot products.
    """
    if features_samples.ndim != 2:
        raise ValueError(f"features_samples must be 2D (B, D), got {features_samples.shape}")
    if num_generations < 1:
        raise ValueError("num_generations must be >= 1")

    B = features_samples.shape[0]
    if B % num_generations != 0:
        raise ValueError(
            f"Batch size {B} not divisible by num_generations {num_generations}. "
            "Features must be grouped contiguously per prompt."
        )
    P = B // num_generations  # num prompts in this microbatch

    device = features_samples.device
    dtype = features_samples.dtype

    # Prepare GT: allow (P, D) or already expanded; make (P, D)
    if features_gt.shape[0] == B:
        # Take every n'th (assumes caller repeated or we slice)
        gt = features_gt[::num_generations].to(device=device, dtype=torch.float32)
    else:
        gt = features_gt.to(device=device, dtype=torch.float32)
    if gt.shape[0] != P:
        # Try to broadcast or error helpfully
        if gt.shape[0] == 1:
            gt = gt.expand(P, -1)
        else:
            raise ValueError(f"features_gt batch {gt.shape[0]} does not match expected P={P}")

    rewards_list: list[torch.Tensor] = []
    alignment_list: list[torch.Tensor] = []
    diversity_list: list[torch.Tensor] = []
    cfm_terms: list[torch.Tensor] = []

    samples_f = features_samples.to(torch.float32)

    for p in range(P):
        start = p * num_generations
        end = start + num_generations
        phi = samples_f[start:end]  # (n, D)
        phi_gt = gt[p : p + 1]  # (1, D)

        if whitening and num_generations > 1:
            # Estimate whitening from the n samples of this group only
            phi_w, phi_gt_w = whiten_embeddings_batched(
                phi, phi_gt, whiten_tol=whiten_tol, normalize=False
            )
            # Alignment: *normalized* only (eq 9)
            align = get_alignment_rewards(phi_w, phi_gt_w.repeat(num_generations, 1), normalize=True)
            # Diversity: keep whitened (non-renormalized) geometry
            divers = get_diversity_rewards(phi_w, num_generations)
        else:
            # Paper base form: use dots on the (per-block-normed) features
            # We do *not* globally renormalize here (unless caller did).
            align = get_alignment_rewards(phi, phi_gt.repeat(num_generations, 1), normalize=False)
            divers = get_diversity_rewards(phi, num_generations)

        # Scale exactly per paper (the 2*)
        align = align * 2.0
        divers = divers * 2.0

        r = alignment_coef * align - diversity_coef * divers  # (n,)

        rewards_list.append(r)

        if return_diagnostics:
            alignment_list.append(align)
            diversity_list.append(divers)
            # CFM proxy: || mean(φ samples) - φ_GT ||^2   (paper eq 2 / L_CFM component)
            mean_phi = phi.mean(dim=0, keepdim=True)
            cfm = ((mean_phi - phi_gt) ** 2).sum(dim=-1)
            cfm_terms.append(cfm)

    rewards = torch.cat(rewards_list, dim=0)  # (B,)

    out: dict[str, Any] = {
        "rewards": rewards.to(dtype),
    }

    if return_diagnostics:
        out["alignment"] = torch.cat(alignment_list, dim=0).to(dtype) if alignment_list else torch.empty(0)
        out["diversity"] = torch.cat(diversity_list, dim=0).to(dtype) if diversity_list else torch.empty(0)
        out["cfm_proxy"] = torch.cat(cfm_terms, dim=0).to(dtype) if cfm_terms else torch.empty(0)
        # Also expose the raw (pre-coef) r for inspection
        out["rewards_pre_coef"] = rewards.to(dtype)  # already has coefs applied in current design
        out["num_prompts"] = P
        out["num_generations"] = num_generations

    return out


# Convenience: a small wrapper that runs the full pipeline from raw ids
# (mostly useful for smoke tests and debugging).
@torch.no_grad()
def compute_ebft_rewards_from_ids(
    feature_extractor: EBFTFeatureExtractor,
    gen_input_ids: torch.Tensor,
    gen_attention_mask: torch.Tensor,
    gt_input_ids: torch.Tensor,
    gt_attention_mask: torch.Tensor,
    num_generations: int,
    whitening: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """
    High-level helper: tokenize-level inputs -> features -> rewards.

    gt_* are expected to be one per prompt (will be internally matched).
    All sequences are prompt + completion.
    """
    # Extract for generations (B = P*n)
    phi_gen = feature_extractor.get_features(gen_input_ids, gen_attention_mask)

    # Extract GT (P,)
    phi_gt = feature_extractor.get_features(gt_input_ids, gt_attention_mask)

    return compute_feature_matching_rewards(
        phi_gen,
        phi_gt,
        num_generations=num_generations,
        whitening=whitening,
        **kwargs,
    )
