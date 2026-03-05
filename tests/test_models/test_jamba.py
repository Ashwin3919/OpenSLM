"""Tests for the Jamba Hybrid SLM."""

import torch
import pytest

from src.models.jamba.model import JambaSLM, HybridBlock
from src.models.jamba.config import JambaConfig
from src.core.mamba_block import MambaBlock
from src.core.generation import generate


@pytest.fixture
def tiny_jamba_config():
    """Minimal JambaConfig for fast CPU tests (4 layers: 2 Mamba + 2 Attn)."""
    return JambaConfig(
        vocab_size=100,
        block_size=16,
        n_layer=4,
        n_embd=32,
        n_head=4,
        mamba_d_state=4,
        mamba_d_conv=4,
        mamba_expand=2,
        intermediate_size=64,
        dropout=0.0,
    )


def test_forward_with_targets_shapes(tiny_jamba_config):
    """Training forward: logits (B,T,V) and scalar loss."""
    model = JambaSLM(tiny_jamba_config)
    B, T = 2, tiny_jamba_config.block_size
    idx = torch.randint(0, tiny_jamba_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_jamba_config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, tiny_jamba_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_forward_without_targets_shapes(tiny_jamba_config):
    """Generation forward: logits (B,1,V) and loss is None."""
    model = JambaSLM(tiny_jamba_config)
    idx = torch.randint(0, tiny_jamba_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, tiny_jamba_config.vocab_size)
    assert loss is None


def test_forward_exceeds_block_size(tiny_jamba_config):
    """Passing T > block_size must raise AssertionError."""
    model = JambaSLM(tiny_jamba_config)
    idx = torch.randint(0, tiny_jamba_config.vocab_size,
                        (1, tiny_jamba_config.block_size + 1))
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_length(tiny_jamba_config):
    """generate() must return (B, T + max_new_tokens) tokens."""
    model = JambaSLM(tiny_jamba_config)
    idx = torch.randint(0, tiny_jamba_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_count_parameters(tiny_jamba_config):
    """count_parameters() must return a positive integer."""
    model = JambaSLM(tiny_jamba_config)
    n = model.count_parameters()
    assert isinstance(n, int) and n > 0


def test_block_type_alternation(tiny_jamba_config):
    """Even-index blocks must be Mamba; odd-index blocks must be Attention."""
    model = JambaSLM(tiny_jamba_config)
    for i, block in enumerate(model.blocks):
        assert block.uses_mamba == (i % 2 == 0), (
            f"Block {i}: expected uses_mamba={i % 2 == 0}, got {block.uses_mamba}"
        )


def test_weight_tying(tiny_jamba_config):
    """Token embedding and LM head must share the same weight tensor."""
    model = JambaSLM(tiny_jamba_config)
    assert model.wte.weight is model.lm_head.weight
