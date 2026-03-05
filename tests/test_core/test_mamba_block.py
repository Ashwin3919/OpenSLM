"""Tests for the shared MambaBlock SSM primitive."""

import torch
import pytest

from src.core.mamba_block import MambaBlock


@pytest.fixture
def tiny_block():
    """Small MambaBlock for fast CPU tests."""
    return MambaBlock(d_model=32, d_state=4, d_conv=4, expand=2)


def test_output_shape(tiny_block):
    """Output shape must match input (B, T, d_model)."""
    x = torch.randn(2, 8, 32)
    assert tiny_block(x).shape == (2, 8, 32)


def test_single_token(tiny_block):
    """Works on T=1 (autoregressive generation step)."""
    x = torch.randn(1, 1, 32)
    assert tiny_block(x).shape == (1, 1, 32)


def test_gradient_flows(tiny_block):
    """Backward through the SSM scan must not error and must produce gradients."""
    x = torch.randn(1, 4, 32, requires_grad=True)
    tiny_block(x).sum().backward()
    assert x.grad is not None


def test_output_not_nan(tiny_block):
    """Random-initialised block must not produce NaN on normal inputs."""
    x = torch.randn(2, 16, 32)
    out = tiny_block(x)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_inner_dimension():
    """d_inner must equal d_model * expand."""
    block = MambaBlock(d_model=16, d_state=4, d_conv=4, expand=3)
    assert block.d_inner == 48


def test_a_log_shape():
    """A_log parameter must be (d_inner, d_state)."""
    block = MambaBlock(d_model=32, d_state=8, d_conv=4, expand=2)
    assert block.A_log.shape == (64, 8)  # d_inner=64, d_state=8


def test_d_skip_shape(tiny_block):
    """D skip connection parameter must be (d_inner,)."""
    assert tiny_block.D.shape == (tiny_block.d_inner,)


def test_output_changes_with_input():
    """Different inputs must produce different outputs (non-trivial mapping)."""
    block = MambaBlock(d_model=32, d_state=4, d_conv=4, expand=2)
    x1 = torch.randn(1, 8, 32)
    x2 = torch.randn(1, 8, 32)
    assert not torch.allclose(block(x1), block(x2))


def test_variable_sequence_length():
    """Block must handle different T values without error."""
    block = MambaBlock(d_model=32, d_state=4, d_conv=4, expand=2)
    for t in (1, 4, 16, 32):
        x = torch.randn(1, t, 32)
        assert block(x).shape == (1, t, 32)
