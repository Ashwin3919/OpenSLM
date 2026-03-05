"""DeepSeek-style Mixture-of-Experts SLM configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DeepSeekMoEConfig:
    """Model architecture parameters for the DeepSeek-MoE SLM.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Total number of transformer blocks.
        n_head: Number of query attention heads.
        n_kv_head: Number of key/value heads (GQA). Must divide ``n_head``.
        n_embd: Embedding / hidden dimension.
        intermediate_size: FFN hidden dim for dense layers.
        n_shared_experts: Number of shared experts (always active in MoE layers).
        n_routed_experts: Number of routed experts to select from per MoE layer.
        top_k: How many routed experts to activate per token.
        expert_hidden_dim: SwiGLU hidden dim per expert (smaller than dense).
        dense_layers: Layer indices that use a dense SwiGLU FFN (not MoE).
        router_aux_loss_coef: Weight of the load-balancing auxiliary loss.
        dropout: Dropout probability.
        rope_theta: Base frequency for RoPE.
    """

    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 384
    intermediate_size: int = 1024
    n_shared_experts: int = 1
    n_routed_experts: int = 8
    top_k: int = 2
    expert_hidden_dim: int = 256
    dense_layers: List[int] = field(default_factory=lambda: [0, 1])
    router_aux_loss_coef: float = 0.01
    dropout: float = 0.0
    rope_theta: float = 10_000.0
