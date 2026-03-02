"""Transformer block: composes CausalSelfAttention and MLP.

Implements the pre-LayerNorm variant used by GPT-2 / nanoGPT:

    x = x + attn(ln1(x))
    x = x + mlp(ln2(x))

Pre-LN places normalisation before each sub-layer rather than after,
which empirically provides more stable training.
"""

from typing import Any

import torch
import torch.nn as nn

from src.core.attention import CausalSelfAttention
from src.core.layers import LayerNorm, MLP


class TransformerBlock(nn.Module):
    """Single pre-LN transformer block (attention + MLP with residuals).

    Args:
        config: ``GPTConfig`` passed through to ``CausalSelfAttention`` and
            ``MLP`` sub-modules.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention and MLP sub-layers.

        Args:
            x: Input tensor of shape ``(B, T, n_embd)``.

        Returns:
            Output tensor of shape ``(B, T, n_embd)``.
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
