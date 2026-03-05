"""BitNet 1.58-bit SLM configuration."""

from dataclasses import dataclass


@dataclass
class BitNetConfig:
    """Model architecture parameters for the BitNet 1.58b SLM.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Number of transformer blocks.
        n_head: Number of query attention heads.
        n_kv_head: Number of key/value heads (GQA). Must divide ``n_head``.
        n_embd: Embedding / hidden dimension. Must be divisible by ``n_head``.
        intermediate_size: SwiGLU hidden dimension.
        dropout: Dropout probability.
        rope_theta: Base frequency for RoPE.

    Note:
        Weights in all ``BitLinear`` layers are quantized to {-1, 0, +1} during
        the forward pass via a Straight-Through Estimator (STE). Full-precision
        weights are kept for gradient computation. Embeddings and the LM head
        are **not** quantized (standard BitNet b1.58 convention).
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
