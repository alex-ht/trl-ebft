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
Feature extraction primitives for Energy-Based Fine-Tuning (EBFT).

This module implements the feature map ϕ described in the paper:
"Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models"
(Jelassi et al., 2026, https://huggingface.co/papers/2603.12248).

Key design from the paper:
- Frozen feature network (copy of generator at init).
- Hidden states from fractional depths (default 25%, 50%, 75%).
- Per-block L2 normalization of each layer's activations, then concat.
- Last-token pooling (default, recommended by paper ablations).

The primitives are self-contained and reusable. Trainer code (Phase 2+)
will duplicate relevant logic per TRL AGENTS.md conventions for isolation.

Supports both full-parameter models (via deepcopy of frozen copy) and
PEFT models (via weight sharing + disable_adapter() trick to avoid ~2x memory).
"""

from __future__ import annotations

import contextlib
import copy
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from transformers import PreTrainedModel

if TYPE_CHECKING:
    from peft import PeftModel


def get_layer_indices_from_fractions(
    model: PreTrainedModel | torch.nn.Module,
    fractions: list[float] | tuple[float, ...],
) -> list[int]:
    """
    Convert fractional layer depths to concrete layer indices.

    Paper recommendation: layers at 25%, 50%, 75% depth capture a good
    mix of low-level, semantic, and higher-level features.

    Args:
        model (`PreTrainedModel` or `nn.Module`):
            The model (or its unwrapped version). Reads `config.num_hidden_layers`
            (or `config.text_config.num_hidden_layers` for some VLMs).
        fractions (`list[float]`):
            List of fractions in [0, 1), e.g. [0.25, 0.5, 0.75].

    Returns:
        `list[int]`: Absolute layer indices (0 = embeddings output, up to num_layers).
            Hidden states tuple length = num_layers + 1.
    """
    # Handle VLM-style nested config
    config = getattr(model, "config", model)
    if hasattr(config, "text_config") and hasattr(config.text_config, "num_hidden_layers"):
        config = config.text_config
    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        # Fallback: try common attribute names
        num_layers = getattr(config, "n_layer", None) or getattr(config, "num_layers", None)
    if num_layers is None:
        raise ValueError(
            f"Could not determine num_hidden_layers from model config. "
            f"Got config type {type(config)}"
        )

    indices = []
    for frac in fractions:
        if not 0 <= frac < 1:
            raise ValueError(f"Layer fraction must be in [0, 1), got {frac}")
        idx = int(frac * num_layers)
        # Ensure at least layer 0 and at most num_layers (hidden_states[0] is embed)
        idx = max(0, min(idx, num_layers))
        if idx not in indices:  # dedup
            indices.append(idx)
    return sorted(indices)


@torch.no_grad()
def extract_hidden_states(
    model: PreTrainedModel | torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_indices: list[int],
    batch_size: int | None = None,
    normalize_blocks: bool = True,
) -> torch.Tensor:
    """
    Forward through the transformer body and extract + concatenate hidden states.

    Uses only the inner `model` body (avoids lm_head vocab projection).

    Per paper: we optionally L2-normalize **each layer block** along the hidden
    dimension before concatenation. This ensures no single layer dominates.

    Args:
        model (`PreTrainedModel` or `nn.Module`):
            Feature network (frozen). Can be the raw model or body.
        input_ids (`torch.Tensor` of shape `(B, S)`):
            Token ids.
        attention_mask (`torch.Tensor` of shape `(B, S)`):
            Attention mask (1 for real tokens).
        layer_indices (`list[int]`):
            Which hidden_states indices to keep (from `output_hidden_states`).
        batch_size (`int`, *optional*):
            If set, process in micro-batches to control peak memory.
        normalize_blocks (`bool`, defaults to `True`):
            If True, apply F.normalize(..., dim=-1) to each selected layer's
            hidden states independently before concatenating (paper-style).

    Returns:
        `torch.Tensor` of shape `(B, S, D)` where D = len(layer_indices) * hidden_size.
            If normalize_blocks=True, each block slice of size hidden_size is unit-normed.
    """
    if batch_size is None:
        batch_size = input_ids.shape[0]

    # Prefer the inner transformer body (skips lm_head entirely).
    # Only hidden_states are needed for features.
    body = getattr(model, "model", None)
    if body is not None and hasattr(body, "forward"):
        forward_model = body
    else:
        forward_model = model

    all_features: list[torch.Tensor] = []
    for i in range(0, input_ids.shape[0], batch_size):
        chunk_ids = input_ids[i : i + batch_size]
        chunk_mask = attention_mask[i : i + batch_size]

        outputs = forward_model(
            chunk_ids,
            attention_mask=chunk_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # hidden_states: tuple of (num_layers + 1) tensors, each (B_chunk, S, H)
        # index 0 = embedding layer output; 1.. = after layer 0, etc.
        hidden_states = outputs.hidden_states

        selected: list[torch.Tensor] = []
        for idx in layer_indices:
            if idx >= len(hidden_states):
                raise IndexError(
                    f"layer index {idx} out of range for hidden_states of length {len(hidden_states)}"
                )
            h = hidden_states[idx]  # (B_chunk, S, H)
            if normalize_blocks:
                h = F.normalize(h, p=2, dim=-1)
            selected.append(h)

        # Concatenate selected (normalized) blocks along feature dim: (B_chunk, S, k*H)
        all_features.append(torch.cat(selected, dim=-1))

    return torch.cat(all_features, dim=0)


def apply_embed_method(
    hidden_states: torch.Tensor,
    method: str = "last_token",
    attention_mask: torch.Tensor | None = None,
    prompt_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pool token-level hidden states (already per-block normalized + concatenated)
    into sequence-level embeddings ϕ(c : y).

    Defaults to "last_token" per paper recommendation and ablations
    (last-token > mean pooling for this objective).

    Args:
        hidden_states (`torch.Tensor` of shape `(B, S, D)`):
            Features after extract + per-block norm + concat.
        method (`str`, defaults to `"last_token"`):
            One of:
            - `"last_token"`: hidden state at the last real (non-pad) position.
            - `"mean_pooling"`: average over all real tokens (masked).
            - `"completion_mean"`: average only over completion tokens (requires `prompt_lengths`).
            - `"concat"`: quartile positions (q1, q2, q3) concatenated (experimental).
        attention_mask (`torch.Tensor` of shape `(B, S)`, *optional*):
            Required for correct last_token / mean / completion handling with padding.
        prompt_lengths (`torch.Tensor` of shape `(B,)`, *optional*):
            Number of prompt tokens (for "completion_mean").

    Returns:
        `torch.Tensor` of shape `(B, D)` (or `(B, 3*D)` for concat).
    """
    method = method.lower()
    B, S, D = hidden_states.shape

    if method == "last_token":
        if attention_mask is not None:
            # Right-padding assumed (standard for generation in causal LMs)
            last_idx = attention_mask.sum(dim=1).long() - 1  # (B,)
            last_idx = last_idx.clamp(min=0, max=S - 1)
            return hidden_states[torch.arange(B, device=hidden_states.device), last_idx]
        return hidden_states[:, -1, :]

    if method == "mean_pooling":
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # (B, S, 1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (hidden_states * mask).sum(dim=1) / denom
        return hidden_states.mean(dim=1)

    if method == "completion_mean":
        if prompt_lengths is None:
            raise ValueError("completion_mean pooling requires prompt_lengths (B,)")
        positions = torch.arange(S, device=hidden_states.device).unsqueeze(0)  # (1, S)
        comp_mask = positions >= prompt_lengths.unsqueeze(1)  # (B, S)
        if attention_mask is not None:
            comp_mask = comp_mask & attention_mask.bool()
        mask = comp_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    if method == "concat":
        if attention_mask is not None:
            valid_lens = attention_mask.sum(dim=1).long()
        else:
            valid_lens = torch.full((B,), S, device=hidden_states.device, dtype=torch.long)
        q1 = (valid_lens // 4).clamp(min=0, max=S - 1)
        q2 = (valid_lens // 2).clamp(min=0, max=S - 1)
        q3 = (3 * valid_lens // 4).clamp(min=0, max=S - 1)
        batch_idx = torch.arange(B, device=hidden_states.device)
        return torch.cat(
            [
                hidden_states[batch_idx, q1],
                hidden_states[batch_idx, q2],
                hidden_states[batch_idx, q3],
            ],
            dim=-1,
        )

    raise ValueError(f"Unknown embed_method: {method}. Choose from last_token, mean_pooling, completion_mean, concat.")


class EBFTFeatureExtractor:
    """
    Self-contained feature extractor for EBFT.

    Handles:
    - Creating / referencing the frozen feature network.
    - PEFT weight-sharing optimization (no full copy when using LoRA).
    - Correct extraction + pooling to ϕ vectors matching the paper.

    Typical usage (full model):
        extractor = EBFTFeatureExtractor(model, layer_fractions=[0.25, 0.5, 0.75])
        phi = extractor.get_features(input_ids, attention_mask)  # (B, D)

    PEFT case (call before wrapping or on unwrapped):
        # extractor will share base weights and use disable_adapter at extract time.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        layer_fractions: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
        embed_method: str = "last_token",
        normalize_blocks: bool = True,
        batch_size: int | None = None,
        is_peft: bool | None = None,
    ):
        """
        Args:
            model (`PreTrainedModel`):
                The generator model. For non-PEFT, a deepcopy will be made and frozen.
                For PEFT, we store a reference and rely on disable_adapter().
            layer_fractions (`list[float]`, defaults to `[0.25, 0.5, 0.75]`):
                Depths for feature extraction.
            embed_method (`str`, defaults to `"last_token"`):
                Pooling strategy.
            normalize_blocks (`bool`, defaults to `True`):
                Per-block L2 norm (paper).
            batch_size (`int`, *optional*):
                Micro-batch size for feature forwards.
            is_peft (`bool`, *optional*):
                Force PEFT detection. If None, auto-detects via PeftModel or `disable_adapter`.
        """
        self.layer_fractions = tuple(layer_fractions)
        self.embed_method = embed_method
        self.normalize_blocks = normalize_blocks
        self.batch_size = batch_size

        # Auto-detect PEFT if not forced.
        # We prefer hasattr(disable_adapter) because it works even after accelerate/DeepSpeed wrapping
        # (isinstance(PeftModel) can fail on wrapped modules).
        if is_peft is None:
            try:
                from peft import PeftModel

                self._is_peft_shared = isinstance(model, PeftModel) or hasattr(
                    model, "disable_adapter"
                )
            except Exception:
                self._is_peft_shared = hasattr(model, "disable_adapter")
        else:
            self._is_peft_shared = bool(is_peft)

        if self._is_peft_shared:
            # Memory-saving path: share the base model weights.
            # Extraction must be done under `disable_adapter()` context.
            self.feature_network: PreTrainedModel | None = None
            self._source_model_for_peft = model
            # Compute indices on the source
            self.layer_indices = get_layer_indices_from_fractions(model, self.layer_fractions)
        else:
            self._source_model_for_peft = None
            # Deepcopy + freeze (standard path for non-PEFT)
            # Note: caller is responsible for ensuring this is only done once.
            unwrapped = model
            self.feature_network = copy.deepcopy(unwrapped)
            for p in self.feature_network.parameters():
                p.requires_grad = False
            self.feature_network.eval()
            self.layer_indices = get_layer_indices_from_fractions(
                self.feature_network, self.layer_fractions
            )

    @property
    def is_peft_shared(self) -> bool:
        return self._is_peft_shared

    def _get_forward_model(self):
        if self._is_peft_shared:
            return self._source_model_for_peft
        return self.feature_network

    @contextlib.contextmanager
    def _feature_forward_context(self):
        """Context that yields the model to use for feature forward.

        For PEFT: temporarily disables adapters so we see base weights.
        """
        if self._is_peft_shared and self._source_model_for_peft is not None:
            # Robust disable (works under DDP/FSDP wrappers in practice)
            try:
                ctx = self._source_model_for_peft.disable_adapter()
            except Exception:
                ctx = contextlib.nullcontext()
            with ctx:
                was_training = self._source_model_for_peft.training
                self._source_model_for_peft.eval()
                try:
                    yield self._source_model_for_peft
                finally:
                    if was_training:
                        self._source_model_for_peft.train()
        else:
            with contextlib.nullcontext():
                yield self.feature_network

    @torch.no_grad()
    def get_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute sequence embeddings ϕ(c : y) for the provided full sequences.

        Args:
            input_ids (`torch.Tensor`): (B, S)
            attention_mask (`torch.Tensor`): (B, S)
            prompt_lengths (`torch.Tensor`, *optional*): For completion_mean.

        Returns:
            `torch.Tensor` of shape `(B, D_phi)`
        """
        model_to_use = self._get_forward_model()
        if model_to_use is None and not self._is_peft_shared:
            raise RuntimeError("No feature network available.")

        with self._feature_forward_context() as fwd_model:
            hidden = extract_hidden_states(
                fwd_model,
                input_ids,
                attention_mask,
                self.layer_indices,
                batch_size=self.batch_size,
                normalize_blocks=self.normalize_blocks,
            )

        phi = apply_embed_method(
            hidden,
            method=self.embed_method,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )
        return phi

    def __repr__(self) -> str:
        return (
            f"EBFTFeatureExtractor(layers={self.layer_indices}, "
            f"method={self.embed_method}, peft_shared={self._is_peft_shared})"
        )
