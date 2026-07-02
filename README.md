# TRL-EBFT

**Energy-Based Fine-Tuning (EBFT)** for use **alongside** the official Hugging Face `trl` package.

> Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models  
> Jelassi et al., arXiv:2603.12248 (2026)

EBFT fine-tunes language models by matching **feature statistics** of on-policy rollouts to reference completions, instead of relying on external reward models or verifiers.

This package (`trl_ebft`) is a **standalone** implementation. You continue to use the official `trl` for everything else (`import trl`), and import EBFT components from here.

```bash
pip install trl
pip install -e .   # this package (trl-ebft)
```

## Key Idea (Paper)

Instead of next-token prediction or scalar rewards, EBFT optimizes:

```
L_FM = || E[ϕ(c : ŷ)] - ϕ(c : y) ||²
```

where `ϕ` is a frozen feature map (hidden states at selected layers, last-token pooled, optionally whitened).

The resulting reward is:

- Alignment term (match reference features)
- Diversity term (push rollouts apart)
- RLOO baseline for low-variance policy gradient

Optional CE regularization further improves validation cross-entropy.

**Advantages**:
- No verifier or reward model needed (uses reference completions from SFT data)
- Works on non-verifiable tasks (raw code, translation, prose)
- Improves both downstream metrics and calibration

## Quick Start

```python
import trl
from trl_ebft import EBFTConfig, EBFTTrainer
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# Dataset must contain "prompt" + reference column (completion / response / ground_truth)
ds = Dataset.from_list([
    {"prompt": "Write a Python function to...", "completion": "def add(a, b): return a + b"},
    ...
])

args = EBFTConfig(
    num_generations=4,
    max_completion_length=256,
    whitening=True,                 # strongly recommended
    feature_layer_fractions=[0.25, 0.5, 0.75],
    ce_reg_weight=0.05,             # optional, helps val CE
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-6,
    max_steps=500,
    report_to="wandb",
)

trainer = EBFTTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    processing_class=tokenizer,
)
trainer.train()
```

## Installation

```bash
pip install trl
git clone https://github.com/your-org/trl-ebft
cd trl-ebft
pip install -e .
```

The package deliberately lives outside the `trl` namespace so you can use the official TRL trainers and this one in the same project.

## Dataset Format

Required columns (flexible naming):
- `prompt`: the input prompt
- Reference column (tried in order):
  - `completion`
  - `response`
  - `chosen`
  - `ground_truth`

The reference text is used both for feature extraction (as the target) and optionally for CE regularization.

## Important Config Options

| Parameter                    | Default          | Description |
|-----------------------------|------------------|-----------|
| `feature_layer_fractions`   | `[0.25, 0.5, 0.75]` | Depths at which to extract hidden states |
| `embed_method`              | `"last_token"`   | Pooling: last_token (recommended), mean_pooling, completion_mean, concat |
| `whitening`                 | `True`           | Per-group whitening (paper eq. 8/9). Keep on for stability |
| `alignment_coef` / `diversity_coef` | `1.0`     | Scaling of the two terms in the reward |
| `ce_reg_weight`             | `0.0`            | γ for optional CE regularization on references |
| `reference_column`          | `"completion"`   | Column name for ground-truth text |
| `use_strided_sampling`      | `False`          | Enable strided block-parallel generation (see below) |

All other fields (generation, PEFT, vLLM, DeepSpeed, etc.) are inherited from `RLOOConfig` / `TrainingArguments`.

## Strided Block-Parallel Sampling (Efficiency)

When `use_strided_sampling=True`, multiple rollouts share prefix computation using a custom attention mask. This is a major win for long generations or larger `num_generations`.

```python
args = EBFTConfig(
    use_strided_sampling=True,
    strided_stride=64,
    strided_context_length=128,
    ...
)
```

- Default (`False`) = naive independent generations (exact same behavior as before).
- Outputs are **identical in shape** to the naive path, so everything downstream (features, rewards, logging) works unchanged.
- Requires `torch >= 2.0`; falls back gracefully.

## Direct Use of Primitives

You can also use just the reward machinery with any trainer:

```python
from trl_ebft import EBFTFeatureExtractor, compute_feature_matching_rewards

extractor = EBFTFeatureExtractor(model, layer_fractions=[0.25, 0.5, 0.75])
phi = extractor.get_features(input_ids, attention_mask)

rewards = compute_feature_matching_rewards(
    phi_gen, phi_gt,
    num_generations=4,
    whitening=True,
)
```

## Citation

If you use this in research, please cite the original paper:

```bibtex
@article{jelassi2026matching,
  title   = {Matching Features, Not Tokens: Energy-Based Fine-Tuning of Language Models},
  author  = {Samy Jelassi and others},
  journal = {arXiv preprint arXiv:2603.12248},
  year    = {2026}
}
```

## Development / Testing

```bash
python -m pytest tests/ -q
python examples/scripts/ebft/ebft_trainer_smoke.py
```

All code lives under `trl_ebft/`. The official `trl` package is never modified.

## Status

- Core EBFTTrainer + feature-matching rewards: complete
- Strided sampling: complete (optional)
- Works alongside official `trl` (RLOO/GRPO patterns)
- Tested with PEFT and small models

For the full training loop on coding/translation tasks, see the smoke scripts and adapt them to your data.