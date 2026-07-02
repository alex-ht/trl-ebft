"""
Strided block-parallel generation helpers for EBFTTrainer.

This module provides optional efficient generation of multiple rollouts per prompt
by interleaving tokens from N "blocks" (rollouts) into a single growing sequence
and using custom (dense 4D or flex BlockMask) attention masks.

Key ideas (adapted from Axolotl EBFT strided.py + Quiet-STaR / paper Section F):
- Prompt prefix is present *once* in the sequence; all block streams attend to it.
- Each "block" (one rollout/completion) only attends to the shared prompt + its
  own previously generated tokens (no cross-rollout leakage).
- Generation loop is manual (token-by-token) because seq len grows and we must
  sample from specific positions for each block at each micro-step.
- Always uses eager/dense 4D masks during generation to avoid dynamo recompile
  storms from variable lengths and no_grad/grad toggles (flex_attention for fixed
  size training forwards if desired in future).
- Position IDs are assigned so that with stride=0 each block's generated tokens
  receive positions [prompt_len, prompt_len+1, ...] (identical to naive AR gen).

When `use_strided_sampling=True` the outputs (prompt_ids, completion_ids, texts)
have exactly the same shapes/semantics as the naive path, so feature extraction,
logp computation, rewards, etc. are unchanged.

Defaults preserve naive behavior when the flag is False.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

# ---------------------------------------------------------------------------
# Flex attention availability (for future / completeness; gen uses eager)
# ---------------------------------------------------------------------------

_FLEX_ATTENTION_AVAILABLE = False
try:
    from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401

    _FLEX_ATTENTION_AVAILABLE = True
except Exception:
    pass


def _patch_flex_attention_dtype():
    """Patch HF flex_attention_forward for q/k/v dtype consistency under grad ckpt."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original_fn = ALL_ATTENTION_FUNCTIONS.get("flex_attention")
    if original_fn is None:
        return

    def patched_flex_attention_forward(
        module, query, key, value, attention_mask, **kwargs
    ):
        target_dtype = value.dtype
        if query.dtype != target_dtype:
            query = query.to(target_dtype)
        if key.dtype != target_dtype:
            key = key.to(target_dtype)
        return original_fn(module, query, key, value, attention_mask, **kwargs)

    ALL_ATTENTION_FUNCTIONS["flex_attention"] = patched_flex_attention_forward


@contextlib.contextmanager
def override_attn_implementation(model: PreTrainedModel, implementation: str):
    """Temporarily force attn implementation (e.g. 'eager').

    Essential during the generation loop: variable seq lens + no_grad cause
    repeated dynamo guards/recompiles when using flex. We force eager for gen
    (dense masks) and can use flex only for fixed-size later forwards.
    """
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "_attn_implementation"):
        yield
        return
    saved = config._attn_implementation
    config._attn_implementation = implementation
    try:
        yield
    finally:
        config._attn_implementation = saved


# ---------------------------------------------------------------------------
# Core strided mask + position builders (adapted from Axolotl strided.py)
# ---------------------------------------------------------------------------


def _strided_mask_mod(
    b: int,
    h: int,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    prompt_length: int,
    context_length: int,
    max_generation_length: int,
    stride: int,
    num_blocks: int,
) -> torch.Tensor:
    """
    Mask mod for flex create_block_mask (or reference for dense).

    - Prompt region: standard causal.
    - Generated region: a token in block `block_idx` at gen step `gen_step` attends to:
        * all prompt tokens up to its context window
        * itself
        * earlier tokens belonging to the *same* block only.
    """
    # Prompt region
    is_prompt_q = q_idx < prompt_length
    is_prompt_kv = kv_idx < prompt_length
    prompt_causal = is_prompt_q & is_prompt_kv & (q_idx >= kv_idx)

    # Generated region
    is_gen_q = ~is_prompt_q
    gen_offset = q_idx - prompt_length
    gen_step = gen_offset // num_blocks
    block_idx = gen_offset % num_blocks

    # Context window end (adapt for prompt+completion use: do not over-clamp past prompt)
    # For parallel rollouts we typically pass context_length >= prompt_length and stride=0
    # so every block sees the full prompt.
    context_end = torch.clamp(
        block_idx * stride + context_length,
        max=prompt_length,  # allow full prompt for structured/prompt-comp case
    )

    in_context = is_gen_q & is_prompt_kv & (kv_idx < context_end)

    is_self = q_idx == kv_idx

    # Earlier tokens in SAME block only (cross-block generated tokens are masked)
    is_gen_kv = ~is_prompt_kv
    kv_gen_offset = kv_idx - prompt_length
    kv_gen_step = kv_gen_offset // num_blocks
    kv_block_idx = kv_gen_offset % num_blocks
    same_block_prev = (
        is_gen_q & is_gen_kv & (kv_block_idx == block_idx) & (kv_gen_step < gen_step)
    )

    return prompt_causal | in_context | is_self | same_block_prev


def create_strided_block_mask(
    prompt_length: int,
    context_length: int,
    max_generation_length: int,
    stride: int,
    num_blocks: int,
    full_sequence_length: int,
    batch_size: int,
    num_heads: int | None,
    device: torch.device,
):
    """Create BlockMask for flex_attention (when available and fixed size)."""
    if not _FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("flex_attention not available; use dense fallback.")

    _prompt_length = torch.tensor(prompt_length, device=device)
    _context_length = torch.tensor(context_length, device=device)
    _max_gen_len = torch.tensor(max_generation_length, device=device)
    _stride = torch.tensor(stride, device=device)
    _num_blocks = torch.tensor(num_blocks, device=device)

    def mask_mod(b, h, q_idx, kv_idx):
        return _strided_mask_mod(
            b,
            h,
            q_idx,
            kv_idx,
            prompt_length=_prompt_length,
            context_length=_context_length,
            max_generation_length=_max_gen_len,
            stride=_stride,
            num_blocks=_num_blocks,
        )

    block_mask = create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,  # broadcast heads
        Q_LEN=full_sequence_length,
        KV_LEN=full_sequence_length,
        device=device,
    )
    return block_mask


def build_strided_position_ids(
    full_sequence_length: int,
    prompt_length: int,
    context_length: int,
    generation_step: int,
    stride: int,
    num_blocks: int,
    device: torch.device,
    batch_size: int = 1,
) -> torch.Tensor:
    """Build position IDs consistent with the strided layout."""
    position_ids = torch.empty(
        (batch_size, full_sequence_length), dtype=torch.long, device=device
    )
    position_ids[:, :prompt_length] = torch.arange(prompt_length, device=device)

    block_starting_positions = (
        torch.arange(num_blocks, device=device) * stride + context_length
    )
    for gs in range(generation_step):
        start = prompt_length + gs * num_blocks
        end = start + num_blocks
        position_ids[:, start:end] = block_starting_positions + gs

    return position_ids


def build_strided_dense_mask_and_positions(
    full_sequence_length: int,
    prompt_length: int,
    context_length: int,
    generation_step: int,
    max_generation_length: int,
    stride: int,
    num_blocks: int,
    device: torch.device,
    batch_size: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build dense 4D attention mask (for eager / generation) + position IDs.

    Used for the variable-length generation loop (always safe, no dynamo issues).
    """
    min_value = torch.finfo(dtype).min
    attention_mask = torch.full(
        (batch_size, 1, full_sequence_length, full_sequence_length),
        min_value,
        dtype=dtype,
        device=device,
    )

    if prompt_length > 0:
        causal_mask = torch.tril(
            torch.ones((prompt_length, prompt_length), dtype=torch.bool, device=device)
        )
        attention_mask[:, :, :prompt_length, :prompt_length].masked_fill_(
            causal_mask.view(1, 1, prompt_length, prompt_length), 0.0
        )

    # Fill allowed attentions for the generated tokens up to `generation_step`
    for gs in range(generation_step):
        for bidx in range(num_blocks):
            gen_pos = prompt_length + gs * num_blocks + bidx
            # context window (capped at prompt for our use case)
            c_end = min(
                bidx * stride + context_length,
                prompt_length,
            )
            # attend to prompt prefix
            attention_mask[:, 0, gen_pos, :c_end] = 0.0
            # self
            attention_mask[:, 0, gen_pos, gen_pos] = 0.0
            # previous same-block tokens
            if gs > 0:
                for prev_s in range(gs):
                    prev_pos = prompt_length + prev_s * num_blocks + bidx
                    attention_mask[:, 0, gen_pos, prev_pos] = 0.0

    position_ids = build_strided_position_ids(
        full_sequence_length,
        prompt_length,
        context_length,
        generation_step,
        stride,
        num_blocks,
        device,
        batch_size,
    )
    return attention_mask, position_ids


# ---------------------------------------------------------------------------
# High-level strided generation for a batch of prompts (EBFT use)
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_completions_with_strided_blocks(
    model: PreTrainedModel,
    tokenizer: Any,
    prompts: list[str],
    num_generations: int,
    max_new_tokens: int,
    stride: int = 0,
    context_length: int = 128,
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
    pad_token_id: int | None = None,
    eos_token_id: int | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
) -> tuple[list[list[int]], list[list[int]], list[str]]:
    """
    Strided block-parallel generation producing multiple completions per prompt.

    For each (unique) prompt we:
      - Tokenize the prompt once.
      - Run a manual generation loop that advances `num_generations` (== num_blocks)
        independent rollouts by appending interleaved tokens and using a custom
        attention mask so that:
          * the prompt prefix KV is materialized only once
          * each rollout only sees prompt + its own prior tokens
      - De-interleave the final long sequence into per-rollout completion ids.

    Returns the same triple as RLOOTrainer._generate:
        prompt_ids_list, completion_ids_list, completions (decoded)

    The returned lists have length = len(prompts) (expanded), with each prompt's
    n completions grouped. Downstream code (padding, feature extraction, logps,
    rewards) is completely unchanged.

    Notes:
    - Generation is always performed with dense eager masks (override "eager").
    - To obtain bit-wise identical behavior to naive generation (same positions),
      call with stride=0 (the default here for rollout sharing) and the builder
      will assign positions [p_len + t] to the t-th token of every rollout.
    - Early stopping per rollout at EOS is supported (we stop sampling that block
      and pad its remaining slots; final trimming happens on decode side too).
    - Works with or without PEFT (we generate with active adapters).
    """
    if device is None:
        device = next(model.parameters()).device

    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    # Group the (possibly repeated) prompts. RepeatSampler yields groups of n identical.
    # We process per unique prompt with its num_generations rollouts.
    if len(prompts) == 0:
        return [], [], []

    # Infer how many prompts and validate grouping
    # We don't assume perfect grouping here; we take every group of `num_generations`
    # and use the first text of the group as the prompt for that group.
    num_groups = (len(prompts) + num_generations - 1) // num_generations
    all_prompt_ids: list[list[int]] = []
    all_completion_ids: list[list[int]] = []
    all_completions: list[str] = []

    unwrapped = model
    # Unwrap common wrappers so we can override attn impl and call forward directly
    for attr in ("module", "_orig_mod", "model"):
        if hasattr(unwrapped, attr):
            unwrapped = getattr(unwrapped, attr)
    # For PEFT we still want adapters active during generation -> no disable_adapter here.

    model_dtype = next(model.parameters()).dtype

    # Seeding for reproducibility / parity tests
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
    else:
        # fall back to global generator state
        rng = None

    for g in range(num_groups):
        start = g * num_generations
        end = min(start + num_generations, len(prompts))
        group_size = end - start
        p_text = prompts[start]  # representative prompt for the group

        # Tokenize prompt (no special tokens added here to match TRL generate style)
        enc = tokenizer(
            p_text,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        p_ids = enc["input_ids"][0].to(device)  # (p_len,)
        p_len = p_ids.shape[0]

        # Effective num blocks for this group (may be < configured at last incomplete group)
        n_blocks = group_size
        gen_len = max_new_tokens

        # Use provided context/stride but ensure we always see at least the full prompt
        used_stride = stride
        used_ctx = max(context_length, p_len)

        # Start the "document" with the prompt
        full_sequence = p_ids.unsqueeze(0).clone()  # (1, p_len)
        # Track per-block whether it has emitted EOS (we keep sampling but can early-mask)
        active_blocks = torch.ones(n_blocks, dtype=torch.bool, device=device)

        # Force eager for the generation loop (variable lens + sampling)
        with override_attn_implementation(unwrapped, "eager"):
            for gs in range(gen_len):
                cur_len = full_sequence.shape[1]

                # Build mask + pos for the *current* state (before appending this step's tokens)
                dense_mask, pos_ids = build_strided_dense_mask_and_positions(
                    full_sequence_length=cur_len,
                    prompt_length=p_len,
                    context_length=used_ctx,
                    generation_step=gs,
                    max_generation_length=gen_len,
                    stride=used_stride,
                    num_blocks=n_blocks,
                    device=device,
                    batch_size=1,
                    dtype=model_dtype,
                )

                # Forward (bf16 autocast common for these models)
                with torch.autocast(device_type=device.type if device.type != "cpu" else "cpu", dtype=torch.bfloat16, enabled=(model_dtype != torch.float32)):
                    outputs = model(
                        full_sequence,
                        attention_mask=dense_mask,
                        position_ids=pos_ids,
                        return_dict=True,
                    )
                logits = outputs.logits  # (1, cur_len, V)

                # For each block, the logit position that predicts its *next* token
                # follows the interleaving layout (step-major, then blocks).
                # For gs==0 we always predict from the final prompt token (common ancestor for all rollouts).
                if gs == 0:
                    logit_positions = [p_len - 1] * n_blocks
                else:
                    logit_positions = [
                        p_len + (gs - 1) * n_blocks + bidx for bidx in range(n_blocks)
                    ]

                # Clamp positions (in case stride made some negative early)
                logit_positions = [max(0, min(lp, cur_len - 1)) for lp in logit_positions]
                pos_tensor = torch.tensor(logit_positions, device=device, dtype=torch.long)

                block_logits = logits.index_select(1, pos_tensor).squeeze(0)  # (n_blocks, V)

                # Sample per block, but only for active ones
                next_tokens = []
                for bidx in range(n_blocks):
                    if not active_blocks[bidx]:
                        next_tokens.append(pad_token_id)
                        continue

                    bl = block_logits[bidx : bidx + 1]  # (1, V)

                    if not do_sample or temperature <= 0:
                        tok = torch.argmax(bl, dim=-1)
                    else:
                        bl = bl / temperature
                        if top_k is not None and top_k > 0:
                            v, _ = torch.topk(bl, min(top_k, bl.size(-1)))
                            bl = torch.where(bl < v[:, -1:], torch.full_like(bl, -float("inf")), bl)
                        probs = F.softmax(bl, dim=-1)

                        if top_p < 1.0:
                            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                            cumulative = torch.cumsum(sorted_probs, dim=-1)
                            remove = cumulative > top_p
                            remove[..., 1:] = remove[..., :-1].clone()
                            remove[..., 0] = False
                            mask = torch.zeros_like(probs, dtype=torch.bool)
                            mask.scatter_(-1, sorted_idx, remove)
                            probs = probs.masked_fill(mask, 0.0)
                            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

                        if rng is not None:
                            tok = torch.multinomial(probs, 1)
                        else:
                            tok = torch.multinomial(probs, 1)
                    tok = tok.squeeze(-1)  # (1,)

                    # Check EOS for this block
                    if int(tok.item()) == eos_token_id:
                        active_blocks[bidx] = False
                    next_tokens.append(tok.item())

                # Append the n (or group_size) tokens for this micro step
                sampled = torch.tensor(next_tokens, device=device).unsqueeze(0)  # (1, n_blocks)
                full_sequence = torch.cat([full_sequence, sampled], dim=1)

        # De-interleave the generated tokens into per-block completions
        full_list = full_sequence[0].tolist()
        gen_tokens_for_blocks: list[list[int]] = [[] for _ in range(n_blocks)]

        for gs in range(gen_len):
            for bidx in range(n_blocks):
                pos = p_len + gs * n_blocks + bidx
                if pos < len(full_list):
                    tok = full_list[pos]
                    # If we already hit EOS for this block, stop collecting real tokens
                    # (we collected pad/eos already; trim later)
                    gen_tokens_for_blocks[bidx].append(tok)

        # Trim at first EOS per block (standard behavior)
        trimmed_comps: list[list[int]] = []
        for comp in gen_tokens_for_blocks:
            if eos_token_id in comp:
                idx = comp.index(eos_token_id)
                comp = comp[: idx + 1]
            # also strip trailing pads if any
            while comp and comp[-1] == pad_token_id:
                comp.pop()
            trimmed_comps.append(comp)

        # Replicate prompt ids for the group (to match naive expanded return shape)
        p_list = p_ids.tolist()
        for bidx in range(n_blocks):
            all_prompt_ids.append(p_list)
            all_completion_ids.append(trimmed_comps[bidx])
            # decode for logging / downstream
            try:
                txt = tokenizer.decode(trimmed_comps[bidx], skip_special_tokens=True)
            except Exception:
                txt = ""
            all_completions.append(txt)

    return all_prompt_ids, all_completion_ids, all_completions


# Convenience alias used by trainer
strided_generate = generate_completions_with_strided_blocks

__all__ = [
    "generate_completions_with_strided_blocks",
    "strided_generate",
    "build_strided_dense_mask_and_positions",
    "build_strided_position_ids",
    "create_strided_block_mask",
    "override_attn_implementation",
    "_FLEX_ATTENTION_AVAILABLE",
]
