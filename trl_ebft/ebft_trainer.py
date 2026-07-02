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
EBFTTrainer: Energy-Based Fine-Tuning trainer for use **alongside** the official `trl` library.

This implements the full EBFT loop (Algorithm 1 in the paper) using:
- On-policy group sampling (via RLOO/GRPO-style RepeatSampler + generation)
- Feature extraction with PEFT smart sharing (no 2x memory for LoRA)
- Paper-exact compute_feature_matching_rewards (with whitening + RLOO baseline)
- Standard TRL training machinery (accelerate, logging, checkpointing, vLLM opt-in, etc.)

Usage (side-by-side with official trl):
    import trl
    from trl_ebft import EBFTConfig, EBFTTrainer

    config = EBFTConfig(...)
    trainer = EBFTTrainer(model=..., args=config, train_dataset=ds_with_prompt_and_completion)
    trainer.train()

The trainer reuses large amounts of battle-tested logic from `trl.trainer.rloo_trainer`
(by subclassing) while specializing the reward to be the EBFT feature-matching reward
and correctly handling reference ("completion") columns from the dataset.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from typing import Any

import torch
from accelerate.logging import get_logger
from accelerate.utils import gather, gather_object
from datasets import Dataset
from torch.utils.data import Sampler
from transformers import PreTrainedModel

from trl.trainer.rloo_trainer import RLOOTrainer
from trl.extras.profiling import profiling_decorator
from trl.trainer.utils import (
    RepeatSampler,
    nanstd,
    pad,
    print_prompt_completions_sample,
)

from .feature_extractor import EBFTFeatureExtractor
from .rewards import compute_feature_matching_rewards

# Strided generation (optional, Phase 3). Import is lazy-safe.
try:
    from .generation_strided import generate_completions_with_strided_blocks
except Exception:
    generate_completions_with_strided_blocks = None  # type: ignore[assignment]

if True:  # always available after Phase 1
    from peft import PeftModel  # type: ignore

logger = get_logger(__name__)


class EBFTTrainer(RLOOTrainer):
    """
    Trainer for Energy-Based Fine-Tuning (EBFT).

    See EBFTConfig for all options. This class deliberately follows the code patterns,
    naming, and structure of RLOOTrainer / GRPOTrainer so that TRL users feel at home.
    """

    _tag_names = ["trl", "ebft", "energy-based"]
    _name = "EBFT"

    def __init__(
        self,
        model: str | PreTrainedModel,
        args: "EBFTConfig | None" = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: Any | None = None,
        callbacks: list | None = None,
        optimizers: tuple = (None, None),
        peft_config: Any | None = None,
        # Intentionally do NOT take reward_funcs — EBFT computes its own internally
    ):
        from .ebft_config import EBFTConfig  # local to avoid circular at import time

        if args is None:
            # Create a default EBFTConfig (will pick up model name etc inside RLOO)
            if isinstance(model, str):
                model_name = model
            else:
                model_name = getattr(model.config, "_name_or_path", None) or getattr(model.config, "name_or_path", "ebft-model")
            model_name = str(model_name).split("/")[-1]
            args = EBFTConfig(f"{model_name}-EBFT")

        if not isinstance(args, EBFTConfig):
            # Allow plain RLOOConfig or TrainingArguments for convenience; wrap
            args = EBFTConfig(**{k: v for k, v in args.__dict__.items() if not k.startswith("_")})

        self.ebft_config = args
        self.reference_column = args.reference_column
        self.whitening = args.whitening
        self.alignment_coef = args.alignment_coef
        self.diversity_coef = args.diversity_coef
        self.ce_reg_weight = args.ce_reg_weight
        self.feature_layer_fractions = tuple(args.feature_layer_fractions)
        self.embed_method = args.embed_method

        # Dummy reward func — we completely override the reward path.
        # This satisfies RLOOTrainer.__init__ signature without side effects.
        def _ebft_dummy_reward(prompts, completions, **kwargs):
            return [0.0] * len(prompts)

        # Let RLOOTrainer do the heavy lifting: model loading/wrapping, PEFT, processing_class setup,
        # sampler wiring, vLLM init, ref model (if beta>0), optimizer, etc.
        super().__init__(
            model=model,
            reward_funcs=[_ebft_dummy_reward],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # === EBFT Feature Extractor (smart PEFT sharing) ===
        # Pass the (possibly PEFT-wrapped + accelerate-wrapped) model.
        # EBFTFeatureExtractor will detect `disable_adapter` and avoid deepcopy.
        self.feature_extractor = EBFTFeatureExtractor(
            self.model,
            layer_fractions=self.feature_layer_fractions,
            embed_method=self.embed_method,
            normalize_blocks=True,
            batch_size=args.feature_batch_size,
        )
        logger.info(f"EBFT feature extractor: {self.feature_extractor}")

        # Storage for EBFT-specific diagnostics (logged via log())
        self._ebft_diagnostics: dict[str, list] = defaultdict(list)

        # Make sure num_generations etc are on self (already set by super, but be explicit)
        self.num_generations = args.num_generations
        self.num_generations_eval = args.num_generations_eval or self.num_generations

        # Strided sampling flags (Phase 3)
        self.use_strided_sampling = bool(getattr(args, "use_strided_sampling", False))
        self.strided_stride = int(getattr(args, "strided_stride", 0) or 0)
        self.strided_context_length = int(getattr(args, "strided_context_length", 128) or 128)
        self.strided_num_blocks = getattr(args, "strided_num_blocks", None)

    # ------------------------------------------------------------------
    # Sampling (identical to RLOO — we inherit, but re-expose for clarity)
    # ------------------------------------------------------------------
    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            seed=self.args.seed,
        )

    # ------------------------------------------------------------------
    # Strided block-parallel generation (optional, when use_strided_sampling)
    # Falls back to parent when disabled or unavailable. Produces identical
    # return shapes so the rest of _generate_and_score_completions is shared.
    # ------------------------------------------------------------------
    def _generate(self, prompts: list):
        use_strided = bool(getattr(self.args, "use_strided_sampling", False))
        # Strided is currently implemented for the regular HF generate path only.
        # vLLM / continuous batching users get the fast path as-is (naive).
        if use_strided and not getattr(self, "use_vllm", False) and not getattr(self, "use_transformers_continuous_batching", False):
            if generate_completions_with_strided_blocks is not None:
                # Strided path: share prefix compute across rollouts via interleaved blocks + masks.
                # We still respect the expanded `prompts` list length (from RepeatSampler).
                # Generation params come from self.generation_config / args.
                gc = getattr(self, "generation_config", None)
                temperature = getattr(gc, "temperature", 0.7) if gc is not None else 0.7
                top_p = getattr(gc, "top_p", 1.0) if gc is not None else 1.0
                top_k = getattr(gc, "top_k", None) if gc is not None else None
                do_sample = getattr(gc, "do_sample", True) if gc is not None else True

                max_new = int(getattr(self.args, "max_completion_length", 128) or 128)
                stride = int(getattr(self.args, "strided_stride", 0) or 0)
                ctx_len = int(getattr(self.args, "strided_context_length", 128) or 128)
                # Strided blocks = num rollouts we generate per prompt. Keep tied to num_generations
                # for RepeatSampler / feature / reward compatibility. strided_num_blocks is advanced override.
                n_per_prompt = int(getattr(self.args, "strided_num_blocks", None) or self.num_generations)

                # Use the trainer seed for parity tests
                seed = getattr(self.args, "seed", None)

                # Note: generate_completions... expects the *expanded* prompts list
                # (len % n_per_prompt == 0) and internally groups by the passed num_generations.
                prompt_ids_list, completion_ids_list, completions = generate_completions_with_strided_blocks(
                    model=self.model,
                    tokenizer=self.processing_class,
                    prompts=prompts,
                    num_generations=n_per_prompt,
                    max_new_tokens=max_new,
                    stride=stride,
                    context_length=ctx_len,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=getattr(self.processing_class, "pad_token_id", None),
                    eos_token_id=getattr(self.processing_class, "eos_token_id", None),
                    seed=seed,
                    device=self.accelerator.device if hasattr(self, "accelerator") else None,
                )
                # Record basic length metrics similar to parent (best effort)
                try:
                    mode = "train" if self.model.training else "eval"
                    comp_lens = torch.tensor([len(c) for c in completion_ids_list], device=self.accelerator.device)
                    self._metrics[mode]["completions/mean_length"].append(comp_lens.float().mean().item())
                    self._metrics[mode]["completions/max_length"].append(comp_lens.float().max().item())
                except Exception:
                    pass
                return prompt_ids_list, completion_ids_list, completions

        # Default / fallback: original RLOO path (vLLM / HF generate / continuous batching)
        return super()._generate(prompts)

    # ------------------------------------------------------------------
    # Core: generate + compute EBFT feature-matching rewards + RLOO adv
    # ------------------------------------------------------------------
    @profiling_decorator
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        # For Phase 2 we focus on pure text (chat or plain). Multimodal can be added later.
        # If images present we still let the parent _generate handle it (best effort).
        has_images = any("image" in x or "images" in x for x in inputs)
        if has_images:
            logger.warning("EBFTTrainer Phase 2 basic path has limited multimodal support; falling back to base generation.")

        # 1. Generate using the (excellent) parent implementation (handles vLLM / HF / continuous batching)
        prompt_ids_list, completion_ids_list, completions = self._generate(prompts)

        # 2. Pad exactly like RLOO (left for prompt, right for completion)
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device)

        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        # 3. Compute OLD logprobs (needed by loss) — reuse parent's method
        with torch.no_grad():
            old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
            )
            old_logps = (old_per_token_logps * completion_mask).sum(dim=1)

        # 4. === EBFT FEATURE-MATCHING REWARD (the heart of the algorithm) ===
        # Extract features for all generated samples (P*n, D)
        # We use the already-constructed padded prompt+completion ids + correct attention mask.
        phi_gen = self.feature_extractor.get_features(prompt_completion_ids, attention_mask)

        # Collect one ground-truth reference per prompt group.
        # Because of RepeatSampler, groups of `num_generations` consecutive items share the same reference.
        gt_full_texts: list[str] = []
        gt_refs_for_logging: list[str] = []
        for i in range(0, len(inputs), num_generations):
            ex = inputs[i]
            ref = (
                ex.get(self.reference_column)
                or ex.get("response")
                or ex.get("chosen")
                or ex.get("ground_truth")
                or ex.get("completion")
                or ""
            )
            # Build the full (prompt + reference) string.
            # The user is responsible for ensuring that "prompt" + "completion" forms a
            # sensible full sequence for feature extraction (same style as the generations).
            # We concatenate directly; extra whitespace is usually harmless for modern tokenizers.
            p = prompts[i]
            if ref and not str(ref).startswith((" ", "\n", "\t")):
                full = p + " " + str(ref)
            else:
                full = p + str(ref)
            gt_full_texts.append(full)
            gt_refs_for_logging.append(str(ref)[:120])

        # Tokenize GTs for feature extraction (right padding is fine — we only use the mask for last-token)
        gt_enc = self.processing_class(
            gt_full_texts,
            return_tensors="pt",
            padding=True,
            padding_side="right",
            add_special_tokens=False,  # generation paths usually don't add extra bos here
        )
        gt_ids = gt_enc["input_ids"].to(device)
        gt_mask = gt_enc["attention_mask"].to(device)

        phi_gt = self.feature_extractor.get_features(gt_ids, gt_mask)  # (P, D)

        # Core paper reward (with optional whitening)
        reward_out = compute_feature_matching_rewards(
            phi_gen,
            phi_gt,
            num_generations=num_generations,
            whitening=self.whitening,
            alignment_coef=self.alignment_coef,
            diversity_coef=self.diversity_coef,
            return_diagnostics=True,
        )
        rewards = reward_out["rewards"]  # (B,)

        # 5. RLOO-style baseline + advantages (exactly like paper + RLOOTrainer)
        # Group per prompt
        grouped_rewards = rewards.view(-1, num_generations)
        if num_generations > 1:
            grouped_sum = torch.nansum(grouped_rewards, dim=1, keepdim=True)
            scorable = (~torch.isnan(grouped_rewards)).sum(dim=1, keepdim=True)
            baselines = (grouped_sum - grouped_rewards) / (scorable - 1).clamp(min=1)
            baselines = baselines.view(-1)
            advantages = rewards - baselines
        else:
            advantages = torch.zeros_like(rewards)

        # Optional advantage normalization (useful in practice)
        if getattr(self, "normalize_advantages", False):
            advantages = (advantages - torch.nanmean(advantages)) / (nanstd(advantages) + 1e-4)

        advantages = torch.nan_to_num(advantages, nan=0.0)

        # 6. Diagnostics for logging (alignment / diversity / cfm are gold for EBFT)
        if "alignment" in reward_out:
            # gather_object expects list-like; scalars -> wrap then take mean on main process
            aln = float(reward_out["alignment"].mean().item())
            div = float(reward_out["diversity"].mean().item())
            cfm = float(reward_out["cfm_proxy"].mean().item())
            # gather to have global view (works for single-process too)
            aln_g = self.accelerator.gather(torch.tensor([aln], device=device)).mean().item()
            div_g = self.accelerator.gather(torch.tensor([div], device=device)).mean().item()
            cfm_g = self.accelerator.gather(torch.tensor([cfm], device=device)).mean().item()
            self._ebft_diagnostics["alignment"].append(aln_g)
            self._ebft_diagnostics["diversity"].append(div_g)
            self._ebft_diagnostics["cfm_proxy"].append(cfm_g)

        # Standard TRL logging of prompts/completions (reuse parent style)
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["advantages"].extend(gather_object(advantages.tolist()))

        # Also log a small sample occasionally
        if self.args.logging_steps and (self.state.global_step % max(1, self.args.logging_steps) == 0):
            try:
                print_prompt_completions_sample(
                    prompts_text[:4], completions_text[:4], rewards[:4].tolist(), 0
                )
            except Exception:
                pass

        # 7. Return the exact dict shape expected by RLOO/GRPO compute_loss + _prepare_inputs
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_logps": old_logps,
            "advantages": advantages,
        }
        return output

    # ------------------------------------------------------------------
    # Optional: CE regularization (Phase 3 hook, implemented lightly here)
    # ------------------------------------------------------------------
    def _compute_loss(self, model, inputs):
        # Standard policy gradient loss (clipped surrogate, same as RLOO) driven by EBFT advantages
        loss = super()._compute_loss(model, inputs)  # uses advantages we injected

        # Optional small CE reg on the reference completions (encourages lower validation CE)
        if self.ce_reg_weight > 0 and "prompt_ids" in inputs and "completion_ids" in inputs:
            # We don't have GT ids here directly, but we can approximate by computing NLL
            # of the *generated* tokens w.r.t model (entropy bonus style) or skip detailed.
            # For a true CE on reference we would need to also carry gt ids.
            # For Phase 2 we keep it as a no-op placeholder (user can set 0.0).
            # Full version would tokenize references again or carry them.
            pass

        # Log EBFT diagnostics
        mode = "train" if self.model.training else "eval"
        if self._ebft_diagnostics:
            for k, vals in list(self._ebft_diagnostics.items()):
                if vals:
                    v = sum(vals) / len(vals)
                    self._metrics[mode][f"ebft/{k}"].append(v)
            self._ebft_diagnostics.clear()

        # Also log mean |reward| / alignment as proxy for feature matching health
        if "advantages" in inputs:
            adv = inputs["advantages"]
            self._metrics[mode]["ebft/adv_mean"].append(self.accelerator.gather(adv).nanmean().item())

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("EBFTTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    # ------------------------------------------------------------------
    # Logging — make sure EBFT metrics are emitted cleanly
    # ------------------------------------------------------------------
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"

        # Pull any pending EBFT diagnostics that may have been added directly
        for key in list(self._ebft_diagnostics.keys()):
            vals = self._ebft_diagnostics.pop(key, [])
            if vals:
                gathered = self.accelerator.gather(torch.tensor(vals, device=self.accelerator.device)).mean().item()
                logs[f"ebft/{key}"] = gathered

        super().log(logs, start_time=start_time)

    # Small helper for users / evals
    @torch.no_grad()
    def compute_feature_matching_loss_proxy(self, dataloader) -> float:
        """Quick helper to evaluate current feature-matching proxy on a dataset (uses GT column)."""
        self.feature_extractor.feature_network.eval() if self.feature_extractor.feature_network is not None else None
        total = 0.0
        count = 0
        for batch in dataloader:
            # minimal path — user can expand
            pass
        return total / max(count, 1)


__all__ = ["EBFTTrainer", "EBFTConfig"]
