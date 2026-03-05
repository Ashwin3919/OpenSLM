"""Tests for RMSNorm."""

import torch
import pytest

from src.core.normalization import RMSNorm


def test_rmsnorm_output_shape():
    """Output shape must match input shape."""
    norm = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    assert norm(x).shape == (2, 16, 64)


def test_rmsnorm_1d_input():
    """Works on a single-step input (B, C)."""
    norm = RMSNorm(32)
    x = torch.randn(4, 32)
    assert norm(x).shape == (4, 32)


def test_rmsnorm_output_dtype_preserved():
    """Output dtype must match input dtype."""
    norm = RMSNorm(16)
    x = torch.randn(1, 4, 16).to(torch.float32)
    assert norm(x).dtype == torch.float32


def test_rmsnorm_zero_input_numerically_stable():
    """Near-zero input must not produce NaN thanks to eps."""
    norm = RMSNorm(8)
    x = torch.zeros(1, 4, 8)
    out = norm(x)
    assert not torch.isnan(out).any(), "NaN in output for zero input"


def test_rmsnorm_unit_weight_scales_correctly():
    """With weight=1 and unit-norm input, output should be close to input."""
    norm = RMSNorm(4)
    # Input whose RMS == 1 along last dim
    x = torch.ones(1, 1, 4) / 2.0   # RMS = 0.5 → normalised to 1 → weight scales back to 1
    x = x * 2.0                       # RMS = 1 exactly
    out = norm(x)
    # After normalisation, output should equal weight (ones) * 1 = 1 element-wise
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_rmsnorm_weight_is_trainable():
    """The weight parameter must require gradient."""
    norm = RMSNorm(16)
    assert norm.weight.requires_grad


def test_rmsnorm_gradient_flows():
    """Backward pass must succeed without error."""
    norm = RMSNorm(8)
    x = torch.randn(2, 4, 8, requires_grad=True)
    loss = norm(x).sum()
    loss.backward()
    assert x.grad is not None
