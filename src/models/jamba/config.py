"""Jamba-style Hybrid Mamba-Attention SLM configuration."""

from dataclasses import dataclass


@dataclass
class JambaConfig:
    """Model architecture parameters for the Jamba Hybrid SLM.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Total number of hybrid blocks (Mamba and Attention alternate).
        n_embd: Embedding / hidden dimension.
        n_head: Number of attention heads (used in attention-type blocks only).
        mamba_d_state: SSM latent state dimension for Mamba blocks.
        mamba_d_conv: Depthwise conv kernel size for Mamba blocks.
        mamba_expand: Inner dimension expansion factor for Mamba blocks.
        intermediate_size: Hidden dim of the SwiGLU FFN shared by all blocks.
        dropout: Dropout probability.
    """

    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 8
    n_embd: int = 384
    n_head: int = 6
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    intermediate_size: int = 1024
    dropout: float = 0.0
