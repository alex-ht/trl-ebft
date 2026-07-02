# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless otherwise indicated by copyright statute that is
# or was first published by the United States Government.
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
EBFTConfig for Energy-Based Fine-Tuning.

Inherits generation, sampling, optimizer, and distributed settings from RLOOConfig
(itself based on TRL _BaseConfig / transformers TrainingArguments) so that all the
standard TRL training machinery (RepeatSampler, generation batching, vLLM paths,
gradient accumulation, DeepSpeed/FSDP, PEFT, logging, etc.) is available out of the box.

EBFT-specific fields control the feature extractor and the feature-matching reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from trl.trainer.rloo_config import RLOOConfig


@dataclass
class EBFTConfig(RLOOConfig):
    # docstyle-ignore
    r"""
    Configuration class for the [`EBFTTrainer`].

    This is a standalone configuration for Energy-Based Fine-Tuning that lives in `trl_ebft`
    and is used **alongside** (never inside) the official `trl` package.

    It inherits all generation, optimization, distributed, vLLM, and sampling controls
    from [`~trl.RLOOConfig`]. Only EBFT-specific parameters are documented here.

    Example:
        ```python
        from trl_ebft import EBFTConfig, EBFTTrainer

        args = EBFTConfig(
            num_generations=4,
            max_completion_length=128,
            feature_layer_fractions=[0.25, 0.5, 0.75],
            whitening=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            # ... other RLOO/TrainingArguments fields
        )
        trainer = EBFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
        ```

    Parameters:
        > EBFT feature matching

        feature_layer_fractions (`list[float]`, *optional*, defaults to `[0.25, 0.5, 0.75]`):
            Fractional depths in the transformer at which to extract hidden states for the feature map ϕ.
            Paper and ablations recommend 3 layers spread across depth.
        embed_method (`str`, *optional*, defaults to `"last_token"`):
            Pooling method to turn per-layer hidden states into a sequence embedding ϕ(c : y).
            Options: `"last_token"` (recommended), `"mean_pooling"`, `"completion_mean"`, `"concat"`.
        whitening (`bool`, *optional*, defaults to `True`):
            Whether to apply per-group whitening (paper eq. 8/9). Strongly recommended for stability.
        alignment_coef (`float`, *optional*, defaults to `1.0`):
            Scaling coefficient for the alignment term in the EBFT reward.
        diversity_coef (`float`, *optional*, defaults to `1.0`):
            Scaling coefficient for the (negative) diversity term.
        ce_reg_weight (`float`, *optional*, defaults to `0.0`):
            If > 0, adds γ * mean CE loss (on reference completions) to the policy gradient loss.
            Small values (0.03-0.1) often help validation CE.
        reference_column (`str`, *optional*, defaults to `"completion"`):
            Dataset column name that holds the ground-truth reference completion / response text.
            Also tries `"response"`, `"chosen"`, and `"ground_truth"` as fallbacks.
        feature_batch_size (`int` or `None`, *optional*):
            Micro-batch size for feature network forward passes. None = full batch.

        > Misc

        beta (`float`, *optional*, defaults to `0.0`):
            KL regularization strength w.r.t. reference (disabled by default for EBFT).
            Set >0 only if you explicitly want an extra KL term.
    """

    # --- EBFT specific ---
    feature_layer_fractions: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75],
        metadata={"help": "Layer fractions for feature extraction (paper default 25/50/75%)."},
    )
    embed_method: str = field(
        default="last_token",
        metadata={"help": 'Pooling for ϕ: "last_token" (best), "mean_pooling", "completion_mean", or "concat".'},
    )
    whitening: bool = field(
        default=True,
        metadata={"help": "Apply per-group whitening (recommended per paper appendix)."},
    )
    alignment_coef: float = field(
        default=1.0,
        metadata={"help": "Multiplier for alignment (feature match) term in reward."},
    )
    diversity_coef: float = field(
        default=1.0,
        metadata={"help": "Multiplier for diversity penalty term in reward."},
    )
    ce_reg_weight: float = field(
        default=0.0,
        metadata={"help": "γ for optional CE regularization term on reference completions (0 disables)."},
    )
    reference_column: str = field(
        default="completion",
        metadata={
            "help": 'Column in dataset containing the reference completion (falls back to "response", "chosen", "ground_truth").'
        },
    )
    feature_batch_size: int | None = field(
        default=None,
        metadata={"help": "Micro batch size for feature extractor forwards. None uses full batch."},
    )

    # --- Strided block-parallel sampling (Phase 3, optional efficiency for longer generations) ---
    # When enabled, multiple rollouts share prefix computation via custom attention masks
    # (interleaved block generation + strided position/mask). Falls back to naive when False.
    use_strided_sampling: bool = field(
        default=False,
        metadata={"help": "Enable strided block-parallel generation for sharing prompt prefixes across rollouts (more efficient for long max_completion_length)."},
    )
    strided_stride: int = field(
        default=64,
        metadata={"help": "Stride between block anchors (for position offsets and context windows). Use 0 for identical position ids to naive generation when doing parallel rollouts from same prompt."},
    )
    strided_context_length: int = field(
        default=128,
        metadata={"help": "Base context window length for each block's attention to prefix. For prompt+completion, typically max(prompt_len, this)."},
    )
    strided_num_blocks: int | None = field(
        default=None,
        metadata={"help": "Number of parallel blocks (rollouts) in strided mode. If None, uses num_generations. Derived in most cases."},
    )

    # Sensible EBFT defaults (override some RLOO ones)
    def __post_init__(self):
        super().__post_init__()
        # EBFT paper / default: no extra KL ref term (set explicitly; users can override)
        self.beta = 0.0
        # Good defaults for group sampling + feature work (EBFT paper uses n=4-8)
        # Only bump if the user did not explicitly choose a higher value and no generation_batch forces small n.
        if getattr(self, "num_generations", None) is None or self.num_generations < 2:
            self.num_generations = 4
        if getattr(self, "max_completion_length", None) in (None, 0):
            self.max_completion_length = 128
        # Logging a bit more often by default is helpful when watching alignment/diversity
        if getattr(self, "logging_steps", 10) == 10:
            self.logging_steps = 5
        # Strided: if num_blocks not set, it will be resolved to num_generations at use time.
        # Keep use_strided_sampling=False by default to preserve exact Phase 2 naive behavior.
        if getattr(self, "strided_stride", None) is None:
            self.strided_stride = 64
        if getattr(self, "strided_context_length", None) is None:
            self.strided_context_length = 128


# Backwards / convenience alias (some people will try EBFTConfig directly)
__all__ = ["EBFTConfig"]
