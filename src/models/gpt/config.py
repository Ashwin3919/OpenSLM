"""GPT model architecture configuration."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Model architecture parameters.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Number of transformer blocks.
        n_head: Number of attention heads per block.
        n_embd: Embedding / hidden dimension. Must be divisible by n_head.
        dropout: Dropout probability applied to embeddings, attention, and MLP.
        bias: Whether to include bias terms in Linear and LayerNorm layers.
    """

    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
