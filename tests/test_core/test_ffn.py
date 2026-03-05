"""Tests for SwiGLU Feed-Forward Network."""

import torch

from src.core.ffn import SwiGLU


def test_swiglu_output_shape():
    """Output shape must match (B, T, dim) for any hidden_dim."""
    ffn = SwiGLU(dim=64, hidden_dim=128)
    x = torch.randn(2, 8, 64)
    assert ffn(x).shape == (2, 8, 64)


def test_swiglu_1d_batch():
    """Works on unbatched (T, dim) input."""
    ffn = SwiGLU(dim=32, hidden_dim=64)
    x = torch.randn(4, 32)
    assert ffn(x).shape == (4, 32)


def test_swiglu_no_bias_by_default():
    """Default constructor must produce bias-free projections."""
    ffn = SwiGLU(dim=16, hidden_dim=32)
    assert ffn.w1.bias is None
    assert ffn.w2.bias is None
    assert ffn.w3.bias is None


def test_swiglu_with_bias():
    """bias=True must add bias parameters."""
    ffn = SwiGLU(dim=16, hidden_dim=32, bias=True)
    assert ffn.w1.bias is not None


def test_swiglu_gradient_flows():
    """Backward pass must complete without error."""
    ffn = SwiGLU(dim=8, hidden_dim=16)
    x = torch.randn(1, 4, 8, requires_grad=True)
    ffn(x).sum().backward()
    assert x.grad is not None


def test_swiglu_output_not_nan():
    """Random-init model must not produce NaN on normal inputs."""
    ffn = SwiGLU(dim=64, hidden_dim=128)
    x = torch.randn(2, 16, 64)
    out = ffn(x)
    assert not torch.isnan(out).any()


def test_swiglu_three_projections():
    """SwiGLU must have exactly three weight matrices: w1, w2, w3."""
    ffn = SwiGLU(dim=32, hidden_dim=64)
    assert hasattr(ffn, "w1") and hasattr(ffn, "w2") and hasattr(ffn, "w3")
