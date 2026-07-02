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
Basic tests for EBFTConfig and EBFTTrainer construction + smoke loop.

These are intentionally lightweight (no full multi-GPU or vLLM).
They verify:
- Config defaults + EBFT fields
- Trainer can be instantiated side-by-side with real trl
- Feature extractor is wired correctly (PEFT vs full)
- A single training step runs without crashing and emits ebft/* scalars
"""

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_ebft import EBFTConfig, EBFTTrainer


def _tiny_model_and_tok():
    model = AutoModelForCausalLM.from_pretrained(
        "hf-internal-testing/tiny-random-gpt2", dtype=torch.float32
    )
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return model, tok


def _tiny_ds(n=4):
    return Dataset.from_list(
        [{"prompt": f"Test prompt {i}", "completion": f" test completion {i}."} for i in range(n)]
    )


def test_ebft_config_defaults():
    cfg = EBFTConfig(output_dir="/tmp/t")
    assert cfg.num_generations >= 2
    assert cfg.beta == 0.0
    assert cfg.whitening is True
    assert 0.2 < cfg.feature_layer_fractions[0] < 0.3
    assert cfg.reference_column == "completion"


def test_ebft_trainer_instantiation_and_feature_extractor():
    model, tok = _tiny_model_and_tok()
    ds = _tiny_ds(2)
    args = EBFTConfig(
        output_dir="/tmp/ebft-test-inst",
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        max_completion_length=4,
        max_steps=1,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False,
    )
    trainer = EBFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    assert trainer is not None
    assert trainer.feature_extractor is not None
    assert hasattr(trainer, "compute_feature_matching_rewards") or hasattr(trainer, "feature_extractor")
    # non-peft path on plain model
    assert trainer.feature_extractor.is_peft_shared is False
    # cleanup
    del trainer, model


@pytest.mark.slow
def test_ebft_trainer_single_training_step_emits_metrics():
    """Full but tiny step: verifies generation + EBFT reward path + logging."""
    model, tok = _tiny_model_and_tok()
    ds = _tiny_ds(4)
    args = EBFTConfig(
        output_dir="/tmp/ebft-test-step",
        per_device_train_batch_size=1,
        num_generations=2,
        generation_batch_size=2,
        max_completion_length=4,
        max_steps=1,
        report_to="none",
        logging_steps=1,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        beta=0.0,
    )
    trainer = EBFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    trainer.train()

    # Check that EBFT-specific metrics made it into the log
    logs = trainer.state.log_history
    assert len(logs) > 0
    # last or penultimate should contain ebft keys from our injection
    joined = " ".join(str(l) for l in logs)
    assert "ebft/alignment" in joined or any("ebft" in str(k) for k in logs[-1].keys())
    del trainer, model


def test_ebft_strided_generation_helper_and_trainer_path():
    """Lightweight check that strided path can be invoked and produces correct output shapes.

    Uses stride=0 + context to keep position semantics close to naive.
    Does not require identical samples (tiny random model + sampling variance).
    """
    from trl_ebft.generation_strided import generate_completions_with_strided_blocks

    model, tok = _tiny_model_and_tok()
    prompts = ["Hello there", "Test case"]
    n = 2
    max_new = 5

    # Direct helper test
    p_list, c_list, texts = generate_completions_with_strided_blocks(
        model=model,
        tokenizer=tok,
        prompts=prompts * n,  # simulate expanded RepeatSampler batch
        num_generations=n,
        max_new_tokens=max_new,
        stride=0,
        context_length=128,
        temperature=0.0,  # greedy for determinism in helper
        do_sample=False,
        seed=123,
        device=next(model.parameters()).device,
    )
    assert len(p_list) == len(prompts) * n
    assert len(c_list) == len(p_list)
    assert len(texts) == len(c_list)
    # Each completion <= max_new (may be shorter due to EOS on tiny model)
    for c in c_list:
        assert isinstance(c, list)
        assert len(c) <= max_new + 1  # +1 for possible EOS

    # Trainer path with flag (uses _generate override)
    ds = _tiny_ds(4)
    args = EBFTConfig(
        output_dir="/tmp/ebft-strided-test",
        per_device_train_batch_size=2,
        num_generations=2,
        generation_batch_size=2,
        max_completion_length=5,
        max_steps=1,
        report_to="none",
        logging_steps=1,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        seed=123,
        use_strided_sampling=True,
        strided_stride=0,
        strided_context_length=128,
    )
    trainer = EBFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tok)
    assert trainer.use_strided_sampling is True
    trainer.train()
    # If we got here, the full _generate_and_score path (features, rewards, loss) worked with strided gens
    assert trainer.state.log_history is not None
    del trainer, model
    print("strided helper + trainer path: OK")
