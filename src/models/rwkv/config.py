"""RWKV-style SLM configuration."""

from dataclasses import dataclass


@dataclass
class RWKVConfig:
    """Model architecture parameters for the RWKV SLM.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Number of RWKV blocks.
        n_embd: Embedding / hidden dimension.
        n_head: Number of heads for the time-mixing WKV computation.
            Must divide ``n_embd`` evenly.
        ffn_mult: Hidden-dimension multiplier for the channel-mix FFN.
        dropout: Dropout probability applied to the token embedding.
    """

    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    ffn_mult: int = 4
    dropout: float = 0.0
