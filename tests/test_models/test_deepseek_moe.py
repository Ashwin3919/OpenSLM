"""Tests for the DeepSeek-style MoE SLM."""

import torch
import pytest

from src.models.deepseek_moe.model import (
    DeepSeekMoESLM,
    MoELayer,
    TopKRouter,
    load_balancing_loss,
)
from src.models.deepseek_moe.config import DeepSeekMoEConfig
from src.core.generation import generate


@pytest.fixture
def tiny_moe_config():
    """Minimal DeepSeekMoEConfig for fast CPU tests."""
    return DeepSeekMoEConfig(
        vocab_size=100,
        block_size=16,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=32,
        intermediate_size=64,
        n_shared_experts=1,
        n_routed_experts=4,
        top_k=2,
        expert_hidden_dim=16,
        dense_layers=[0, 1],
        router_aux_loss_coef=0.01,
        dropout=0.0,
    )


def test_forward_with_targets_shapes(tiny_moe_config):
    """Training forward: logits (B,T,V) and scalar loss (CE + aux)."""
    model = DeepSeekMoESLM(tiny_moe_config)
    B, T = 2, tiny_moe_config.block_size
    idx = torch.randint(0, tiny_moe_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_moe_config.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, tiny_moe_config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0
    assert loss.item() > 0.0


def test_forward_without_targets_shapes(tiny_moe_config):
    """Generation forward: logits (B,1,V) and loss is None."""
    model = DeepSeekMoESLM(tiny_moe_config)
    idx = torch.randint(0, tiny_moe_config.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, tiny_moe_config.vocab_size)
    assert loss is None


def test_forward_exceeds_block_size(tiny_moe_config):
    """Passing T > block_size must raise AssertionError."""
    model = DeepSeekMoESLM(tiny_moe_config)
    idx = torch.randint(
        0, tiny_moe_config.vocab_size, (1, tiny_moe_config.block_size + 1)
    )
    with pytest.raises(AssertionError):
        model(idx)


def test_generate_output_length(tiny_moe_config):
    """generate() must return (B, T + max_new_tokens) tokens."""
    model = DeepSeekMoESLM(tiny_moe_config)
    idx = torch.randint(0, tiny_moe_config.vocab_size, (1, 4))
    out = generate(model, idx, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_count_parameters(tiny_moe_config):
    """count_parameters() must return a positive integer."""
    model = DeepSeekMoESLM(tiny_moe_config)
    n = model.count_parameters()
    assert isinstance(n, int) and n > 0


def test_router_output_shapes(tiny_moe_config):
    """TopKRouter must return correct weight/index/logit shapes."""
    router = TopKRouter(
        n_embd=tiny_moe_config.n_embd,
        n_experts=tiny_moe_config.n_routed_experts,
        top_k=tiny_moe_config.top_k,
    )
    x = torch.randn(2, 8, tiny_moe_config.n_embd)
    weights, indices, logits = router(x)
    assert weights.shape == (2, 8, tiny_moe_config.top_k)
    assert indices.shape == (2, 8, tiny_moe_config.top_k)
    assert logits.shape == (2, 8, tiny_moe_config.n_routed_experts)


def test_router_weights_sum_to_one(tiny_moe_config):
    """Routing weights for each token must sum to 1 (softmax over top-k)."""
    router = TopKRouter(
        n_embd=tiny_moe_config.n_embd,
        n_experts=tiny_moe_config.n_routed_experts,
        top_k=tiny_moe_config.top_k,
    )
    x = torch.randn(2, 8, tiny_moe_config.n_embd)
    weights, _, _ = router(x)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8), atol=1e-5)


def test_load_balancing_loss_positive(tiny_moe_config):
    """Load-balancing auxiliary loss must be a non-negative scalar."""
    n = tiny_moe_config.n_routed_experts
    top_k = tiny_moe_config.top_k
    router_logits = torch.randn(2, 8, n)
    indices = torch.randint(0, n, (2, 8, top_k))
    aux = load_balancing_loss(router_logits, indices, n)
    assert aux.ndim == 0
    assert aux.item() >= 0.0


def test_moe_layer_output_shape(tiny_moe_config):
    """MoELayer must return output of the same shape as input."""
    layer = MoELayer(tiny_moe_config)
    x = torch.randn(2, 8, tiny_moe_config.n_embd)
    out, router_logits = layer(x)
    assert out.shape == x.shape
    assert router_logits.shape == (2, 8, tiny_moe_config.n_routed_experts)


def test_dense_layers_have_no_router(tiny_moe_config):
    """Dense layers must not be MoELayer instances."""
    model = DeepSeekMoESLM(tiny_moe_config)
    for i in tiny_moe_config.dense_layers:
        assert not isinstance(model.blocks[i].ffn, MoELayer), (
            f"Layer {i} is in dense_layers but has a MoELayer FFN"
        )
