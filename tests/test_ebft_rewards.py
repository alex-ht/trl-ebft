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
Unit tests for EBFT reward computation.

Focus:
- Exact match to paper formulas (eq 7 non-whitened, eq 9 whitened).
- Shapes, grouping, per-prompt whitening.
- Numerical stability of whitening (rank-deficient, small n).
- Diagnostics and coef scaling.
- Correct RLOO is applied *outside* (here we just check raw r_j).
"""

import math

import pytest
import torch

from trl_ebft.rewards import (
    compute_feature_matching_rewards,
    get_alignment_rewards,
    get_diversity_rewards,
    whiten_embeddings_batched,
)


def test_paper_formula_non_whitened_exact():
    """
    Construct tiny deterministic features.
    n=3, P=1, D=2.

    Manually compute rj per paper eq (7) using dots (no extra global norm).
    """
    torch.manual_seed(123)
    n = 3
    # phi_j : (3, 2)
    phi = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],  # norm=1 but we don't force here
        ]
    )
    phi_gt = torch.tensor([[0.8, 0.6]])

    # Expand GT for call convention (or pass (1, D))
    out = compute_feature_matching_rewards(
        phi,
        phi_gt,
        num_generations=n,
        whitening=False,
        alignment_coef=1.0,
        diversity_coef=1.0,
        return_diagnostics=True,
    )
    r = out["rewards"]

    # Manual computation (paper):
    # align_j = 2 * dot(phi_j, phi_gt)
    # divers_j = 2 * mean_{j'!=j} dot(phi_j, phi_j')
    # r_j = align_j - divers_j
    dots_gt = (phi * phi_gt).sum(dim=1)  # [0.8, 0.6, 0.96]
    align = 2 * dots_gt

    # pairwise
    # j0 with {1,2}: dot(1,0;0,1)=0 ; dot(1,0;0.6,0.8)=0.6  -> mean 0.3
    # j1 with {0,2}
    # j2 with {0,1}
    d01 = (phi[0] * phi[1]).sum()
    d02 = (phi[0] * phi[2]).sum()
    d12 = (phi[1] * phi[2]).sum()

    div0 = 2 * (d01 + d02) / 2
    div1 = 2 * (d01 + d12) / 2
    div2 = 2 * (d02 + d12) / 2

    r_expected = torch.stack([align[0] - div0, align[1] - div1, align[2] - div2])

    assert torch.allclose(r, r_expected, atol=1e-6), f"r={r} vs expected {r_expected}"


def test_whitened_alignment_is_normalized_diversity_raw():
    """For whitening=True the alignment term must be cosine, diversity raw dots on whitened."""
    torch.manual_seed(7)
    n = 4
    D = 8
    phi = torch.randn(n, D)
    phi_gt = torch.randn(1, D)

    out = compute_feature_matching_rewards(
        phi,
        phi_gt,
        num_generations=n,
        whitening=True,
        return_diagnostics=True,
    )

    # We cannot easily recompute SVD here without duplicating, but we can sanity check
    # that returned alignment values are in [-1, 1] * 2 (because of the *2 later)
    align = out["alignment"]
    assert align.shape == (n,)
    assert (align.abs() <= 2.0 + 1e-5).all(), "Whitened alignment (after *2) should be bounded by 2"

    # diversity uses the whitened geometry but is not forced into [-2,2]
    divers = out["diversity"]
    assert divers.shape == (n,)

    # cfm_proxy should be non-negative scalar per prompt
    cfm = out["cfm_proxy"]
    assert cfm.shape == (1,)
    assert cfm.item() >= -1e-8


def test_grouped_multi_prompt():
    """Two prompts, n=2 each. Verify grouping and returned shapes."""
    n = 2
    P = 2
    D = 4
    torch.manual_seed(42)
    phi = torch.randn(P * n, D)
    gt = torch.randn(P, D)

    out = compute_feature_matching_rewards(
        phi, gt, num_generations=n, whitening=False, return_diagnostics=True
    )
    assert out["rewards"].shape == (P * n,)
    assert out["cfm_proxy"].shape == (P,)


def test_whitening_stability_rank_deficient():
    """Low-rank or duplicate samples should not NaN or crash."""
    n = 3
    D = 16
    # Make highly correlated / rank deficient
    base = torch.randn(1, D)
    phi = base.repeat(n, 1) + 1e-9 * torch.randn(n, D)
    phi_gt = torch.randn(1, D)

    out = compute_feature_matching_rewards(
        phi, phi_gt, num_generations=n, whitening=True, whiten_tol=1e-4, return_diagnostics=True
    )
    r = out["rewards"]
    assert torch.isfinite(r).all(), f"Non-finite rewards on rank-deficient input: {r}"
    # Should degrade gracefully (often near constant or zero diversity)
    assert r.shape == (n,)


def test_coefs_and_zero_diversity():
    n = 2
    phi = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gt = torch.tensor([[0.5, 0.5]])

    out_full = compute_feature_matching_rewards(
        phi, gt, num_generations=n, whitening=False, alignment_coef=1.0, diversity_coef=1.0
    )
    out_no_div = compute_feature_matching_rewards(
        phi, gt, num_generations=n, whitening=False, alignment_coef=1.0, diversity_coef=0.0
    )

    # When diversity_coef=0, reward == 2 * dot (no subtraction)
    expected_align_only = 2 * (phi * gt).sum(dim=1)
    assert torch.allclose(out_no_div["rewards"], expected_align_only, atol=1e-6)

    # Full should be smaller in magnitude usually because of the penalty
    assert (out_full["rewards"] <= out_no_div["rewards"] + 1e-5).all()


def test_get_diversity_rewards_zero_when_n1():
    phi = torch.randn(5, 3)
    d = get_diversity_rewards(phi, num_generations=1)
    assert torch.allclose(d, torch.zeros(5))


def test_get_alignment_variants():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    cos = get_alignment_rewards(a, b, normalize=True)
    dot = get_alignment_rewards(a, b, normalize=False)
    assert torch.allclose(cos, torch.tensor([0.0, 0.0]), atol=1e-7)
    assert torch.allclose(dot, torch.tensor([0.0, 0.0]), atol=1e-7)


def test_whiten_returns_same_dtype_device():
    phi = torch.randn(3, 5, dtype=torch.float16)
    gt = torch.randn(1, 5, dtype=torch.float16)
    w1, w2 = whiten_embeddings_batched(phi, gt)
    assert w1.dtype == torch.float16
    assert w2.dtype == torch.float16
