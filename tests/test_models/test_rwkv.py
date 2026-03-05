"""Tests for the RWKV SLM."""

import torch
import pytest

from src.models.rwkv.model import RWKVSLM, RWKV_TimeMix, RWKV_ChannelMix
from src.models.rwkv.config import RWKVConfig
from src.core.generation import generate


@pytest.fixture
def tiny_rwkv_config():
    """Minimal RWKVConfig for fast CPU tests."""
    return RWKVConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_embd=32,
        n_head=4,
        ffn_mult=2,
        dropout=0.0,
    )


def test_forward_with_targets_shapes(tiny_rwkv_config):
    """Training forward: logits (B,T,V) and scalar loss."""
    model = RWKVSLM(tiny_rwkv_config)
    B, T = 2, tiny_rwkv_config.block_size
    idx = torch.randint(0, tiny_rwkv_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_rwkv_config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, tiny_rwkv_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_forward_without_targets_shapes(tiny_rwkv_config):
    """Generation forward: logits (B,1,V) and loss is None."""
    model = RWKVSLM(tiny_rwkv_config)
    idx = torch.randint(0, tiny_rwkv_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, tiny_rwkv_config.vocab_size)
    assert loss is None


def test_forward_exceeds_block_size(tiny_rwkv_config):
    """Passing T > block_size must raise AssertionError."""
    model = RWKVSLM(tiny_rwkv_config)
    idx = torch.randint(0, tiny_rwkv_config.vocab_size,
                        (1, tiny_rwkv_config.block_size + 1))
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_length(tiny_rwkv_config):
    """generate() must return (B, T + max_new_tokens) tokens."""
    model = RWKVSLM(tiny_rwkv_config)
    idx = torch.randint(0, tiny_rwkv_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_count_parameters(tiny_rwkv_config):
    """count_parameters() must return a positive integer."""
    model = RWKVSLM(tiny_rwkv_config)
    n = model.count_parameters()
    assert isinstance(n, int) and n > 0


def test_time_mix_output_shape(tiny_rwkv_config):
    """RWKV_TimeMix output must match input shape."""
    tm = RWKV_TimeMix(tiny_rwkv_config)
    x = torch.randn(1, 8, tiny_rwkv_config.n_embd)
    assert tm(x).shape == x.shape


def test_channel_mix_output_shape(tiny_rwkv_config):
    """RWKV_ChannelMix output must match input shape."""
    cm = RWKV_ChannelMix(tiny_rwkv_config)
    x = torch.randn(1, 8, tiny_rwkv_config.n_embd)
    assert cm(x).shape == x.shape


def test_weight_tying(tiny_rwkv_config):
    """Token embedding and LM head must share the same weight tensor."""
    model = RWKVSLM(tiny_rwkv_config)
    assert model.wte.weight is model.lm_head.weight
