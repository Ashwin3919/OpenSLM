"""RetNet SLM configuration."""

from dataclasses import dataclass


@dataclass
class RetNetConfig:
    """Model architecture parameters for the RetNet SLM.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Number of RetNet blocks.
        n_head: Number of retention heads. Each head gets a unique decay rate γ.
        n_embd: Embedding / hidden dimension. Must be divisible by ``n_head``.
        intermediate_size: SwiGLU hidden dimension.
        gamma_min: Smallest per-head decay rate (head attending shortest range).
        gamma_max: Largest per-head decay rate (head attending longest range).
        dropout: Dropout probability.
    """

    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    intermediate_size: int = 1024
    gamma_min: float = 0.85
    gamma_max: float = 0.999
    dropout: float = 0.0
