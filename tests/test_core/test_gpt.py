"""Tests for the GPT model forward pass, loss, and generation."""

import torch
import pytest

from models.gpt.model import GPT
from src.core.generation import generate


def test_forward_with_targets_shapes(tiny_model_config):
    """Training forward: logits shape (B,T,V) and scalar loss."""
    model = GPT(tiny_model_config)
    B, T = 2, tiny_model_config.block_size
    idx = torch.randint(0, tiny_model_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_model_config.vocab_size, (B, T))

    logits, loss = model(idx, targets)

    assert logits.shape == (B, T, tiny_model_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0.0


def test_forward_without_targets_shapes(tiny_model_config):
    """Generation forward: logits shape (B,1,V) and loss is None."""
    model = GPT(tiny_model_config)
    idx = torch.randint(0, tiny_model_config.vocab_size, (2, 8))
    logits, loss = model(idx)

    assert logits.shape == (2, 1, tiny_model_config.vocab_size)
    assert loss is None


def test_forward_exceeds_block_size(tiny_model_config):
    """Passing T > block_size must raise AssertionError."""
    model = GPT(tiny_model_config)
    idx = torch.randint(
        0, tiny_model_config.vocab_size, (1, tiny_model_config.block_size + 1)
    )
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_length(tiny_model_config):
    """generate() must return (B, T + max_new_tokens) tokens."""
    model = GPT(tiny_model_config)
    B, T, max_new = 1, 4, 5
    idx = torch.randint(0, tiny_model_config.vocab_size, (B, T))
    out = generate(model, idx, max_new_tokens=max_new)
    assert out.shape == (B, T + max_new)


def test_generate_with_top_k(tiny_model_config):
    """generate() with top_k should complete without error."""
    model = GPT(tiny_model_config)
    idx = torch.randint(0, tiny_model_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=3, top_k=10)
    assert out.shape == (1, 7)


def test_count_parameters(tiny_model_config):
    """count_parameters() must return a positive integer."""
    model = GPT(tiny_model_config)
    n = model.count_parameters()
    assert isinstance(n, int)
    assert n > 0
