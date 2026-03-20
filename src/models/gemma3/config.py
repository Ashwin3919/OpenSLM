"""Gemma 3 SLM configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Gemma3Config:
    """Model architecture parameters for the Gemma 3-style SLM.

    Attributes:
        vocab_size: Vocabulary size (GPT-2 tokenizer = 50257).
        block_size: Maximum context window length in tokens.
        n_layer: Number of transformer blocks.
        n_head: Number of query attention heads per block.
        n_kv_head: Number of key/value heads (Grouped Query Attention).
            Must divide ``n_head`` evenly.
        n_embd: Embedding / hidden dimension. Must be divisible by ``n_head``.
        intermediate_size: Hidden dimension for the GeGLU FFN.
        sliding_window: Local attention window size (tokens). Each token in a
            local layer attends only to the previous ``sliding_window`` tokens.
        local_rope_theta: RoPE base frequency for local attention layers.
            Gemma 3 uses 10 000 — same as the original LLaMA.
        global_rope_theta: RoPE base frequency for global attention layers.
            Gemma 3 uses 1 000 000 — much higher to encode longer-range positions.
        attn_logit_cap: Tanh soft-cap applied to raw attention scores before
            softmax. Prevents extreme values that destabilise training.
            Gemma 3 uses 50.0.
        final_logit_cap: Tanh soft-cap applied to output logits before the
            cross-entropy loss (training) or generation sampling (inference).
            Gemma 3 uses 30.0.
        global_layers: List of layer indices that use full causal (global)
            attention. All other layers use sliding-window (local) attention.
            Default [5] gives a 5:1 local:global ratio for n_layer=6.
        dropout: Dropout probability (set 0.0 for inference).
    """

    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 384
    intermediate_size: int = 1024
    sliding_window: int = 64
    local_rope_theta: float = 10_000.0
    global_rope_theta: float = 1_000_000.0
    attn_logit_cap: float = 50.0
    final_logit_cap: float = 30.0
    global_layers: List[int] = field(default_factory=lambda: [5])
    dropout: float = 0.0
