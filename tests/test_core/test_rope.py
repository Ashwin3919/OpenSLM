"""Tests for Rotary Position Embeddings."""

import torch
import pytest

from src.core.rope import apply_rotary_emb, precompute_freqs_cis


@pytest.fixture
def rope_setup():
    """Standard dims used across RoPE tests."""
    return dict(dim=64, max_seq_len=32)


def test_precompute_freqs_cis_shape(rope_setup):
    """Frequency tensor shape must be (max_seq_len, head_dim // 2) complex."""
    f = precompute_freqs_cis(**rope_setup)
    assert f.shape == (rope_setup["max_seq_len"], rope_setup["dim"] // 2)
    assert f.is_complex()


def test_precompute_freqs_cis_unit_magnitude(rope_setup):
    """Complex frequencies must lie on the unit circle (|e^{i*θ}| = 1)."""
    f = precompute_freqs_cis(**rope_setup)
    mags = f.abs()
    assert torch.allclose(mags, torch.ones_like(mags), atol=1e-5)


def test_apply_rotary_emb_shape():
    """apply_rotary_emb must return tensors with the same shapes as input."""
    B, T, n_head, n_kv_head, head_dim = 2, 8, 4, 2, 32
    freqs = precompute_freqs_cis(head_dim, T)

    xq = torch.randn(B, T, n_head, head_dim)
    xk = torch.randn(B, T, n_kv_head, head_dim)

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)

    assert xq_out.shape == xq.shape
    assert xk_out.shape == xk.shape


def test_apply_rotary_emb_dtype_preserved():
    """Output dtype must match input dtype."""
    T, head_dim = 4, 16
    freqs = precompute_freqs_cis(head_dim, T)

    xq = torch.randn(1, T, 2, head_dim)
    xk = torch.randn(1, T, 1, head_dim)

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)

    assert xq_out.dtype == xq.dtype
    assert xk_out.dtype == xk.dtype


def test_apply_rotary_emb_position_zero_is_identity():
    """At position 0 the frequency is e^{i*0} = 1, so rotation is identity."""
    head_dim = 8
    freqs = precompute_freqs_cis(head_dim, max_seq_len=1)

    xq = torch.randn(1, 1, 2, head_dim)
    xk = torch.randn(1, 1, 1, head_dim)

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)

    # Position 0 → multiply by e^{i*0} = 1 → output == input
    assert torch.allclose(xq_out, xq, atol=1e-5)
    assert torch.allclose(xk_out, xk, atol=1e-5)


def test_apply_rotary_emb_gradient_flows():
    """Backward through apply_rotary_emb must not error."""
    T, head_dim = 4, 16
    freqs = precompute_freqs_cis(head_dim, T)

    xq = torch.randn(1, T, 2, head_dim, requires_grad=True)
    xk = torch.randn(1, T, 1, head_dim, requires_grad=True)

    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs)
    (xq_out.sum() + xk_out.sum()).backward()

    assert xq.grad is not None
    assert xk.grad is not None
