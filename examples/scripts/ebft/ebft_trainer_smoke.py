#!/usr/bin/env python
"""
Trainer-level smoke test for EBFT (Phase 2+).

- Loads a tiny causal LM.
- Builds a tiny synthetic dataset with "prompt" + "completion".
- Instantiates EBFTConfig + EBFTTrainer (reusing official trl generation paths).
- Runs 1-2 real training steps exercising:
  - Group sampling (num_generations)
  - On-policy generation
  - EBFTFeatureExtractor (paper last-token + block-norm)
  - compute_feature_matching_rewards + RLOO baseline inside trainer
  - PEFT detection (non-PEFT path here)
  - Logging of ebft/alignment, ebft/diversity, ebft/cfm_proxy, ebft/adv_mean

This proves that the full EBFTTrainer can be used in practice alongside `import trl`.

Run:
    python examples/scripts/ebft/ebft_trainer_smoke.py

You should see non-trivial ebft/* metrics and a completed training step without crash.
"""

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_ebft import EBFTConfig, EBFTTrainer


def main():
    print("=== EBFTTrainer smoke (full training loop) ===")
    model_id = "hf-internal-testing/tiny-random-gpt2"

    print(f"Loading tiny model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Tiny dataset with clear prompt / reference completion pairs.
    # In real use you would use much larger SFT-style data (code, math, translation, ...).
    raw = [
        {"prompt": "Write a greeting:", "completion": " Hello there!"},
        {"prompt": "Write a greeting:", "completion": " Hi friend."},
        {"prompt": "Continue: 2 + 2 =", "completion": " 4"},
        {"prompt": "Continue: sky is", "completion": " blue today."},
        {"prompt": "Say something nice:", "completion": " You are great."},
        {"prompt": "Say something nice:", "completion": " Have a good day."},
    ]
    ds = Dataset.from_list(raw)

    args = EBFTConfig(
        output_dir="/tmp/ebft-trainer-smoke",
        per_device_train_batch_size=2,
        num_generations=2,
        generation_batch_size=2,  # small for the smoke
        max_completion_length=8,
        max_steps=2,
        learning_rate=5e-6,
        report_to="none",
        logging_steps=1,
        remove_unused_columns=False,
        beta=0.0,
        disable_dropout=True,
        gradient_checkpointing=False,
        seed=123,
        # EBFT specific (already have good defaults)
        whitening=True,
        feature_layer_fractions=[0.25, 0.5, 0.75],
    )

    print("Instantiating EBFTTrainer (side-by-side with official trl)...")
    trainer = EBFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
    )
    print(f"  feature extractor: {trainer.feature_extractor}")
    print(f"  num_generations={trainer.num_generations} whitening={trainer.whitening}")

    print("\nRunning short training loop (this exercises generation + EBFT rewards)...")
    trainer.train()

    # Inspect that EBFT diagnostics were logged
    history = trainer.state.log_history
    last = history[-1] if history else {}
    print("\n=== Final logged scalars (sample) ===")
    for k in sorted(last.keys()):
        if any(x in k for x in ("ebft", "loss", "reward", "entropy", "step")):
            print(f"  {k}: {last[k]}")

    # Basic sanity: we should have seen ebft metrics
    has_ebft = any("ebft/alignment" in str(h) or "ebft" in str(h) for h in history)
    assert has_ebft or "ebft/alignment" in str(last), "Expected EBFT diagnostics in logs"

    print("\n=== EBFTTrainer smoke test PASSED ===")
    print("You can now use `from trl_ebft import EBFTTrainer` together with the real `trl` package.")


if __name__ == "__main__":
    main()


