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
TRL-EBFT: Energy-Based Fine-Tuning for use alongside the official `trl` package.

This standalone package implements the full EBFT algorithm from the paper
"Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models"
(Jelassi et al., 2026, arXiv:2603.12248).

Core idea
---------
Instead of next-token prediction or scalar rewards, match the feature statistics
(hidden states at selected layers) of on-policy rollouts to reference completions.

Main exports
------------
EBFTConfig
    Full configuration. Inherits generation, sampling, PEFT, vLLM, accelerate,
    and optimizer controls from TRL's RLOOConfig. Only EBFT-specific fields are
    documented here.

EBFTTrainer
    Trainer that subclasses TRL's RLOOTrainer, reuses its generation machinery,
    and injects the paper's feature-matching reward (with RLOO baseline).

EBFTFeatureExtractor
    Frozen feature extractor ϕ (multi-layer, last-token/mean pooling, block-norm).

compute_feature_matching_rewards
    Exact implementation of the paper reward (alignment - diversity ± whitening).

Strided generation (Phase 3)
    generate_completions_with_strided_blocks — optional efficient block-parallel
    generation that shares prompt prefixes across rollouts.

Typical usage (alongside official trl)
--------------------------------------
    import trl
    from trl_ebft import EBFTConfig, EBFTTrainer

    args = EBFTConfig(
        num_generations=4,
        whitening=True,
        feature_layer_fractions=[0.25, 0.5, 0.75],
        # plus all normal TRL/TrainingArguments fields
    )
    trainer = EBFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    trainer.train()

See README.md for full installation instructions, dataset format, strided
sampling details, and a complete config reference.

Paper: https://arxiv.org/abs/2603.12248
"""

from .ebft_config import EBFTConfig
from .ebft_trainer import EBFTTrainer
from .feature_extractor import (
    EBFTFeatureExtractor,
    apply_embed_method,
    extract_hidden_states,
    get_layer_indices_from_fractions,
)
from .rewards import (
    compute_feature_matching_rewards,
    compute_ebft_rewards_from_ids,
    get_alignment_rewards,
    get_diversity_rewards,
    whiten_embeddings_batched,
)

# Optional strided generation (Phase 3). Safe to import even if internals change.
try:
    from .generation_strided import (
        generate_completions_with_strided_blocks,
        build_strided_dense_mask_and_positions,
    )
except Exception:  # pragma: no cover
    generate_completions_with_strided_blocks = None  # type: ignore
    build_strided_dense_mask_and_positions = None  # type: ignore

__all__ = [
    # Phase 2 main API
    "EBFTConfig",
    "EBFTTrainer",
    # Phase 1 primitives
    "EBFTFeatureExtractor",
    "compute_feature_matching_rewards",
    "compute_ebft_rewards_from_ids",
    "whiten_embeddings_batched",
    "get_alignment_rewards",
    "get_diversity_rewards",
    # lower-level helpers
    "extract_hidden_states",
    "apply_embed_method",
    "get_layer_indices_from_fractions",
]

__version__ = "0.3.0"