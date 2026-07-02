#!/usr/bin/env python
"""
Minimal smoke test / example for Phase 1 EBFT primitives.

Loads a tiny model (gpt2), constructs prompt + multiple "rollouts" + a GT,
extracts features using paper-style last-token + per-block norm,
computes rewards exactly following the paper formulas (with optional whitening).

This exercises:
- EBFTFeatureExtractor (full-model deepcopy path)
- extract + pooling
- compute_feature_matching_rewards
- Diagnostics (alignment, diversity, cfm_proxy)

Run:
    python examples/scripts/ebft/ebft_phase1_smoke.py

Expected: no crashes, finite positive/negative rewards, sensible CFM proxy,
and shapes (n,).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_ebft import (
    EBFTFeatureExtractor,
    compute_feature_matching_rewards,
)


def main():
    print("Loading tiny model (gpt2) for smoke test...")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Toy data: one prompt, n=4 "rollouts" (we manually create short sequences)
    # In real use the trainer will generate these on-policy.
    prompt = "Write a haiku about programming:"
    gt_completion = " Code flows like water.\nBugs hide in the silent depths.\nTests reveal the truth."

    # Fake varied rollouts (different lengths/styles)
    rollouts = [
        " Code flows like streams.",
        " Bugs in the matrix lurk.",
        " Syntax sings at dawn.",
        " Loops dance forevermore.",
    ]

    # Build full sequences: prompt + completion (no special tokens beyond tokenizer)
    def build_text(c: str) -> str:
        return prompt + " " + c.strip()

    gen_texts = [build_text(c) for c in rollouts]
    gt_text = build_text(gt_completion)

    # Tokenize together for proper padding
    all_gen = tok(gen_texts, return_tensors="pt", padding=True)
    gt_enc = tok([gt_text], return_tensors="pt", padding=True)

    gen_ids = all_gen["input_ids"]
    gen_mask = all_gen["attention_mask"]
    gt_ids = gt_enc["input_ids"]
    gt_mask = gt_enc["attention_mask"]

    print(f"Generated batch: {gen_ids.shape[0]} samples (n=4)")
    print(f"GT shape: {gt_ids.shape}")

    # === Phase 1: Feature extractor (paper defaults) ===
    extractor = EBFTFeatureExtractor(
        model,
        layer_fractions=[0.25, 0.5, 0.75],
        embed_method="last_token",  # paper + Axolotl recommended default
        normalize_blocks=True,
    )
    print(f"Extractor: {extractor}")

    # Get ϕ for all generations + GT
    with torch.no_grad():
        phi_gen = extractor.get_features(gen_ids, gen_mask)  # (4, D)
        phi_gt = extractor.get_features(gt_ids, gt_mask)  # (1, D)

    print(f"Feature dim D = {phi_gen.shape[1]}")
    print(f"phi_gen mean norm (pre any global): {phi_gen.norm(dim=-1).mean():.4f}")
    print(f"phi_gt norm: {phi_gt.norm():.4f}")

    # === Compute rewards (paper eqs) ===
    result = compute_feature_matching_rewards(
        phi_gen,
        phi_gt,
        num_generations=4,
        whitening=True,  # paper appendix + practical default
        alignment_coef=1.0,
        diversity_coef=1.0,
        return_diagnostics=True,
    )

    rewards = result["rewards"]
    print("\n=== EBFT Rewards (paper-style) ===")
    print(f"rewards: {rewards.tolist()}")
    print(f"mean reward: {rewards.mean().item():+.4f}")

    if "alignment" in result:
        print(f"alignment (scaled): {result['alignment'].tolist()}")
        print(f"diversity (scaled): {result['diversity'].tolist()}")
        print(f"cfm_proxy (||E[phi] - phi_GT||^2): {result['cfm_proxy'].item():.6f}")

    # Quick sanity
    assert torch.isfinite(rewards).all(), "Rewards must be finite"
    assert rewards.shape == (4,), "Wrong reward shape"

    # In real training these rewards (minus RLOO baseline) * logp will drive the update.
    print("\nPhase 1 primitives smoke test PASSED.")


if __name__ == "__main__":
    main()
