"""Tests for the RetNet SLM."""

import torch
import pytest

from src.models.retnet.model import RetNetSLM, MultiScaleRetention
from src.models.retnet.config import RetNetConfig
from src.core.generation import generate


@pytest.fixture
def tiny_retnet_config():
    """Minimal RetNetConfig for fast CPU tests."""
    return RetNetConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=32,
        intermediate_size=64,
        gamma_min=0.85,
        gamma_max=0.999,
        dropout=0.0,
    )


def test_forward_with_targets_shapes(tiny_retnet_config):
    """Training forward: logits (B,T,V) and scalar loss."""
    model = RetNetSLM(tiny_retnet_config)
    B, T = 2, tiny_retnet_config.block_size
    idx = torch.randint(0, tiny_retnet_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_retnet_config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, tiny_retnet_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_forward_without_targets_shapes(tiny_retnet_config):
    """Generation forward: logits (B,1,V) and loss is None."""
    model = RetNetSLM(tiny_retnet_config)
    idx = torch.randint(0, tiny_retnet_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, tiny_retnet_config.vocab_size)
    assert loss is None


def test_forward_exceeds_block_size(tiny_retnet_config):
    """Passing T > block_size must raise AssertionError."""
    model = RetNetSLM(tiny_retnet_config)
    idx = torch.randint(
        0, tiny_retnet_config.vocab_size, (1, tiny_retnet_config.block_size + 1)
    )
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_length(tiny_retnet_config):
    """generate() must return (B, T + max_new_tokens) tokens."""
    model = RetNetSLM(tiny_retnet_config)
    idx = torch.randint(0, tiny_retnet_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_count_parameters(tiny_retnet_config):
    """count_parameters() must return a positive integer."""
    model = RetNetSLM(tiny_retnet_config)
    n = model.count_parameters()
    assert isinstance(n, int) and n > 0


def test_retention_mask_is_causal(tiny_retnet_config):
    """Decay mask D[h, i, j] must be 0 for j > i (upper triangle = 0)."""
    msr = MultiScaleRetention(tiny_retnet_config)
    T = 8
    D = msr._build_decay_mask(T, device=torch.device("cpu"))
    # D shape: (1, n_head, T, T)
    # Check upper triangle (j > i) is zero
    for h in range(tiny_retnet_config.n_head):
        mask = D[0, h]
        for i in range(T):
            for j in range(i + 1, T):
                assert mask[i, j].item() == 0.0, (
                    f"Non-zero at head={h}, i={i}, j={j}: {mask[i, j].item()}"
                )


def test_retention_mask_diagonal_is_one(tiny_retnet_config):
    """D[h, i, i] = gamma_h^0 = 1 for all heads and positions."""
    msr = MultiScaleRetention(tiny_retnet_config)
    T = 4
    D = msr._build_decay_mask(T, device=torch.device("cpu"))
    for h in range(tiny_retnet_config.n_head):
        diag = torch.diagonal(D[0, h])
        assert torch.allclose(diag, torch.ones(T), atol=1e-5)


def test_gamma_range(tiny_retnet_config):
    """All gammas must lie in [gamma_min, gamma_max]."""
    msr = MultiScaleRetention(tiny_retnet_config)
    g = msr.gammas
    assert g.min().item() >= tiny_retnet_config.gamma_min - 1e-6
    assert g.max().item() <= tiny_retnet_config.gamma_max + 1e-6


def test_weight_tying(tiny_retnet_config):
    """Token embedding and LM head must share the same weight tensor."""
    model = RetNetSLM(tiny_retnet_config)
    assert model.wte.weight is model.lm_head.weight
