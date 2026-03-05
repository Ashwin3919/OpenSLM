"""Tests for the BitNet 1.58b SLM."""

import torch
import pytest

from src.models.bitnet.model import BitNetSLM, BitLinear
from src.models.bitnet.config import BitNetConfig
from src.core.generation import generate


@pytest.fixture
def tiny_bitnet_config():
    """Minimal BitNetConfig for fast CPU tests."""
    return BitNetConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        intermediate_size=64,
        dropout=0.0,
    )


def test_forward_with_targets_shapes(tiny_bitnet_config):
    """Training forward: logits (B,T,V) and scalar loss."""
    model = BitNetSLM(tiny_bitnet_config)
    B, T = 2, tiny_bitnet_config.block_size
    idx = torch.randint(0, tiny_bitnet_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_bitnet_config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, tiny_bitnet_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_forward_without_targets_shapes(tiny_bitnet_config):
    """Generation forward: logits (B,1,V) and loss is None."""
    model = BitNetSLM(tiny_bitnet_config)
    idx = torch.randint(0, tiny_bitnet_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, tiny_bitnet_config.vocab_size)
    assert loss is None


def test_forward_exceeds_block_size(tiny_bitnet_config):
    """Passing T > block_size must raise AssertionError."""
    model = BitNetSLM(tiny_bitnet_config)
    idx = torch.randint(0, tiny_bitnet_config.vocab_size,
                        (1, tiny_bitnet_config.block_size + 1))
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_length(tiny_bitnet_config):
    """generate() must return (B, T + max_new_tokens) tokens."""
    model = BitNetSLM(tiny_bitnet_config)
    idx = torch.randint(0, tiny_bitnet_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_count_parameters(tiny_bitnet_config):
    """count_parameters() must return a positive integer."""
    model = BitNetSLM(tiny_bitnet_config)
    n = model.count_parameters()
    assert isinstance(n, int) and n > 0


def test_bitlinear_weight_quantized_range():
    """BitLinear ternary-quantized weight values must be in {-1, 0, +1}."""
    layer = BitLinear(16, 32)
    w = layer._ternary_weight(layer.weight)
    # After detach so we can check values without STE
    w_vals = (w - w.detach()) + w.detach()
    unique = w_vals.detach().round().unique()
    assert all(v.item() in (-1.0, 0.0, 1.0) for v in unique)


def test_bitlinear_output_shape():
    """BitLinear forward must produce correct output shape."""
    layer = BitLinear(32, 64)
    x = torch.randn(2, 8, 32)
    assert layer(x).shape == (2, 8, 64)


def test_bitlinear_gradient_flows():
    """Backward through BitLinear (STE) must not error."""
    layer = BitLinear(16, 32)
    x = torch.randn(1, 4, 16, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None


def test_embeddings_not_quantized(tiny_bitnet_config):
    """Token embedding and lm_head must be standard nn.Linear/nn.Embedding."""
    import torch.nn as nn
    model = BitNetSLM(tiny_bitnet_config)
    assert isinstance(model.wte, nn.Embedding)
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, BitLinear)
