"""Tests for the LLaMA-style SLM."""

import torch
import pytest

from src.models.llama.model import GQAttention, LlamaSLM
from src.models.llama.config import LlamaConfig
from src.core.generation import generate


@pytest.fixture
def tiny_llama_config():
    """Minimal LlamaConfig for fast CPU tests."""
    return LlamaConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        intermediate_size=64,
        dropout=0.0,
    )


def test_forward_with_targets_shapes(tiny_llama_config):
    """Training forward: logits (B,T,V) and scalar loss."""
    model = LlamaSLM(tiny_llama_config)
    B, T = 2, tiny_llama_config.block_size
    idx = torch.randint(0, tiny_llama_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_llama_config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, tiny_llama_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_forward_without_targets_shapes(tiny_llama_config):
    """Generation forward: logits (B,1,V) and loss is None."""
    model = LlamaSLM(tiny_llama_config)
    idx = torch.randint(0, tiny_llama_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, tiny_llama_config.vocab_size)
    assert loss is None


def test_forward_exceeds_block_size(tiny_llama_config):
    """Passing T > block_size must raise AssertionError."""
    model = LlamaSLM(tiny_llama_config)
    idx = torch.randint(
        0, tiny_llama_config.vocab_size, (1, tiny_llama_config.block_size + 1)
    )
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_length(tiny_llama_config):
    """generate() must return (B, T + max_new_tokens) tokens."""
    model = LlamaSLM(tiny_llama_config)
    idx = torch.randint(0, tiny_llama_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_generate_with_top_k(tiny_llama_config):
    """generate() with top_k must complete without error."""
    model = LlamaSLM(tiny_llama_config)
    idx = torch.randint(0, tiny_llama_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=3, top_k=10)
    assert out.shape == (1, 7)


def test_count_parameters(tiny_llama_config):
    """count_parameters() must return a positive integer."""
    model = LlamaSLM(tiny_llama_config)
    n = model.count_parameters()
    assert isinstance(n, int) and n > 0


def test_gqa_n_head_not_divisible_by_n_kv_head():
    """GQAttention constructor must raise if n_head % n_kv_head != 0."""
    bad_config = LlamaConfig(
        vocab_size=100, block_size=16, n_layer=1,
        n_head=6, n_kv_head=4, n_embd=24, intermediate_size=32,
    )
    with pytest.raises(AssertionError):
        LlamaSLM(bad_config)


def test_weight_tying(tiny_llama_config):
    """Token embedding and LM head must share the same weight tensor."""
    model = LlamaSLM(tiny_llama_config)
    assert model.wte.weight is model.lm_head.weight


def test_freqs_cis_buffer_shape(tiny_llama_config):
    """Precomputed RoPE frequencies buffer shape must be (block_size, head_dim//2)."""
    model = LlamaSLM(tiny_llama_config)
    head_dim = tiny_llama_config.n_embd // tiny_llama_config.n_head
    assert model.freqs_cis.shape == (tiny_llama_config.block_size, head_dim // 2)
