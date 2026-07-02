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
Unit tests for EBFT feature extraction primitives.

These tests verify:
- Layer index computation
- Hidden state extraction shapes and device
- Per-block L2 normalization (paper requirement)
- Pooling methods (last_token is default and correct w/ padding)
- Correct handling of attention masks
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from trl_ebft.feature_extractor import (
    EBFTFeatureExtractor,
    apply_embed_method,
    extract_hidden_states,
    get_layer_indices_from_fractions,
)


def _get_tiny_model():
    """Use a very small public model for fast tests. gpt2 is ~124M but loads quick."""
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model.eval()
    return model


def _get_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


class TestLayerIndices:
    def test_basic_fractions(self):
        # Simulate a 12-layer model (gpt2)
        class FakeConfig:
            num_hidden_layers = 12

        class FakeModel:
            config = FakeConfig()

        idxs = get_layer_indices_from_fractions(FakeModel(), [0.25, 0.5, 0.75])
        assert idxs == [3, 6, 9], f"Expected [3,6,9] for 12L, got {idxs}"

    def test_edge_and_dedup(self):
        class FakeConfig:
            num_hidden_layers = 32

        class FakeModel:
            config = FakeConfig()

        idxs = get_layer_indices_from_fractions(FakeModel(), [0.0, 0.1, 0.999, 0.5])
        # 0.0->0, 0.1->3, 0.5->16, 0.999->31
        assert 0 in idxs and 31 in idxs
        assert len(idxs) == len(set(idxs))


class TestExtractAndPool:
    def test_extract_shapes_and_norm(self):
        model = _get_tiny_model()
        tok = _get_tokenizer()

        texts = ["Hello world", "This is a test sentence for pooling."]
        enc = tok(texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        mask = enc["attention_mask"]

        # 12 layer GPT-2
        layer_idxs = [3, 6, 9]
        hs = extract_hidden_states(
            model, input_ids, mask, layer_idxs, batch_size=None, normalize_blocks=True
        )

        B, S, D = hs.shape
        assert B == 2
        assert D == 3 * model.config.n_embd  # 3 blocks * 768

        # Check per-block normalization (take first block slice)
        H = model.config.n_embd
        block0 = hs[0, :, :H]
        norms = block0.norm(dim=-1)
        # Norms should be ~1.0 where mask==1 (allowing tiny float error)
        valid = mask[0].bool()
        assert torch.allclose(norms[valid], torch.ones_like(norms[valid]), atol=1e-5)

    def test_last_token_pooling_with_padding(self):
        # Hand-crafted small case
        torch.manual_seed(0)
        B, S, H = 2, 5, 4
        hs = torch.randn(B, S, H)
        # mask: sample0 has len 3, sample1 has len 5
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.long)

        pooled = apply_embed_method(hs, "last_token", mask)
        assert pooled.shape == (2, H)

        # Manually verify last real token
        assert torch.allclose(pooled[0], hs[0, 2])
        assert torch.allclose(pooled[1], hs[1, 4])

    def test_mean_pooling(self):
        hs = torch.ones(1, 4, 2)
        mask = torch.tensor([[1, 1, 0, 0]])
        pooled = apply_embed_method(hs, "mean_pooling", mask)
        # mean of first two ones
        assert torch.allclose(pooled, torch.tensor([[1.0, 1.0]]))


@pytest.mark.slow
def test_ebft_extractor_class_end_to_end():
    """Integration smoke with real (tiny) model and extractor."""
    model = _get_tiny_model()
    tok = _get_tokenizer()

    extractor = EBFTFeatureExtractor(
        model,
        layer_fractions=[0.25, 0.5, 0.75],
        embed_method="last_token",
        normalize_blocks=True,
    )

    texts = ["The quick brown fox", "jumps over the lazy dog and more text"]
    enc = tok(texts, return_tensors="pt", padding=True, return_attention_mask=True)
    ids = enc["input_ids"]
    mask = enc["attention_mask"]

    phi = extractor.get_features(ids, mask)
    assert phi.shape[0] == 2
    # D = 3 blocks * 768
    assert phi.shape[1] == 3 * model.config.n_embd

    # Last token should have been used; values not all zero
    assert phi.abs().mean() > 1e-6


def test_completion_mean_requires_lengths():
    hs = torch.randn(1, 6, 8)
    with pytest.raises(ValueError):
        apply_embed_method(hs, "completion_mean")
