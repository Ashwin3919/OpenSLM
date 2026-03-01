"""Tests for CausalSelfAttention."""

import torch
import pytest

from src.core.attention import CausalSelfAttention


def test_attention_output_shape(tiny_model_config):
    """Output shape must be (B, T, n_embd) — same as input."""
    attn = CausalSelfAttention(tiny_model_config)
    x = torch.randn(2, tiny_model_config.block_size, tiny_model_config.n_embd)
    assert attn(x).shape == x.shape


def test_attention_bad_n_head(tiny_model_config):
    """Should raise AssertionError when n_embd % n_head != 0."""
    from src.models.config import GPTConfig
    bad_cfg = GPTConfig(
        vocab_size=100, block_size=16, n_layer=2,
        n_head=3, n_embd=64, dropout=0.0, bias=True,
    )
    with pytest.raises(AssertionError):
        CausalSelfAttention(bad_cfg)


def test_attention_causal_mask(tiny_model_config):
    """Position 0 output must not change when only the last position changes."""
    attn = CausalSelfAttention(tiny_model_config)
    attn.eval()

    torch.manual_seed(0)
    x = torch.randn(1, tiny_model_config.block_size, tiny_model_config.n_embd)
    out1 = attn(x)

    # Perturb only the last token
    x2 = x.clone()
    x2[0, -1, :] = torch.randn(tiny_model_config.n_embd)
    out2 = attn(x2)

    # First position must be unaffected by the last token (causal)
    assert torch.allclose(out1[0, 0, :], out2[0, 0, :], atol=1e-5)


def test_attention_shorter_sequence(tiny_model_config):
    """Should handle sequences shorter than block_size without error."""
    attn = CausalSelfAttention(tiny_model_config)
    T = tiny_model_config.block_size // 2
    x = torch.randn(1, T, tiny_model_config.n_embd)
    assert attn(x).shape == (1, T, tiny_model_config.n_embd)
