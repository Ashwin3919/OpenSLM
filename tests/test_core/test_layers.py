"""Tests for LayerNorm and MLP."""

import torch
import pytest

from src.core.layers import LayerNorm, MLP


def test_layer_norm_output_shape():
    """Output shape must match input shape."""
    ln = LayerNorm(64, bias=True)
    x = torch.randn(2, 16, 64)
    assert ln(x).shape == x.shape


def test_layer_norm_no_bias():
    """Bias parameter must be None when bias=False."""
    ln = LayerNorm(64, bias=False)
    assert ln.bias is None
    x = torch.randn(2, 16, 64)
    assert ln(x).shape == x.shape


def test_layer_norm_normalises():
    """Output should have approximately zero mean and unit variance."""
    ln = LayerNorm(64, bias=False)
    x = torch.randn(4, 8, 64) * 100 + 50   # large offset and scale
    out = ln(x)
    # Check across the last dimension per token
    assert out.mean(dim=-1).abs().max() < 1e-4
    assert (out.std(dim=-1) - 1.0).abs().max() < 1e-2


def test_mlp_output_shape(tiny_model_config):
    """MLP output shape must equal input shape."""
    mlp = MLP(tiny_model_config)
    x = torch.randn(2, tiny_model_config.block_size, tiny_model_config.n_embd)
    assert mlp(x).shape == x.shape
