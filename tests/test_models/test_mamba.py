"""Tests for the Mamba SSM SLM."""

import torch
import pytest

from src.models.mamba.model import MambaSLM
from src.models.mamba.config import MambaConfig
from src.core.generation import generate


@pytest.fixture
def tiny_mamba_config():
    """Minimal MambaConfig for fast CPU tests."""
    return MambaConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        d_model=32,
        d_state=4,
        d_conv=4,
        expand=2,
        dropout=0.0,
    )


def test_forward_with_targets_shapes(tiny_mamba_config):
    """Training forward: logits (B,T,V) and scalar loss."""
    model = MambaSLM(tiny_mamba_config)
    B, T = 2, tiny_mamba_config.block_size
    idx = torch.randint(0, tiny_mamba_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_mamba_config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, tiny_mamba_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_forward_without_targets_shapes(tiny_mamba_config):
    """Generation forward: logits (B,1,V) and loss is None."""
    model = MambaSLM(tiny_mamba_config)
    idx = torch.randint(0, tiny_mamba_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, tiny_mamba_config.vocab_size)
    assert loss is None


def test_forward_exceeds_block_size(tiny_mamba_config):
    """Passing T > block_size must raise AssertionError."""
    model = MambaSLM(tiny_mamba_config)
    idx = torch.randint(0, tiny_mamba_config.vocab_size,
                        (1, tiny_mamba_config.block_size + 1))
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_length(tiny_mamba_config):
    """generate() must return (B, T + max_new_tokens) tokens."""
    model = MambaSLM(tiny_mamba_config)
    idx = torch.randint(0, tiny_mamba_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_count_parameters(tiny_mamba_config):
    """count_parameters() must return a positive integer."""
    model = MambaSLM(tiny_mamba_config)
    n = model.count_parameters()
    assert isinstance(n, int) and n > 0


def test_ssm_state_shapes(tiny_mamba_config):
    """The MambaBlock SSM scan must produce the correct inner shape."""
    from src.core.mamba_block import MambaBlock
    block = MambaBlock(d_model=32, d_state=4, d_conv=4, expand=2)
    x = torch.randn(1, 8, 32)
    out = block(x)
    assert out.shape == (1, 8, 32)


def test_weight_tying(tiny_mamba_config):
    """Token embedding and LM head must share the same weight tensor."""
    model = MambaSLM(tiny_mamba_config)
    assert model.wte.weight is model.lm_head.weight
