"""Mamba SSM configuration."""

from dataclasses import dataclass


@dataclass
class MambaConfig:
    """Model architecture parameters for the Mamba SSM SLM.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Number of Mamba blocks (more layers than a transformer because
            each Mamba block is computationally cheaper than self-attention).
        d_model: Hidden / embedding dimension.
        d_state: Dimension of the SSM latent state *H*.
        d_conv: Kernel size for the depthwise 1-D convolution inside each block.
        expand: Inner dimension expansion factor (``d_inner = d_model * expand``).
        dropout: Dropout probability applied to the token embedding.
    """

    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 12
    d_model: int = 384
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.0
