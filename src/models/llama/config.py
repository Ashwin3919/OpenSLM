"""LLaMA-style SLM configuration."""

from dataclasses import dataclass


@dataclass
class LlamaConfig:
    """Model architecture parameters for the LLaMA-style SLM.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Number of transformer blocks.
        n_head: Number of query attention heads per block.
        n_kv_head: Number of key/value heads (Grouped Query Attention).
            Must divide ``n_head`` evenly. Use ``n_head`` for MHA.
        n_embd: Embedding / hidden dimension. Must be divisible by ``n_head``.
        intermediate_size: Hidden dimension of the SwiGLU FFN.
            Smaller than ``4 * n_embd`` because SwiGLU has three matrices.
        dropout: Dropout probability (set 0.0 for inference).
        rope_theta: Base frequency for Rotary Position Embeddings.
    """

    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 384
    intermediate_size: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10_000.0
