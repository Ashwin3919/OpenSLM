"""Reusable layer components: LayerNorm and MLP.

Both are pure PyTorch — no IO, no logging, no config reading.
They accept a ``config`` object so they can be composed into blocks
without knowing about the broader system.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalisation with an optional bias parameter.

    Wraps ``F.layer_norm`` with learnable ``weight`` and an optional
    ``bias`` (following the nanoGPT convention: bias can be disabled
    to save parameters when ``GPTConfig.bias = False``).

    Args:
        ndim: Normalised dimension size (equals ``n_embd``).
        bias: If ``True``, adds a learnable bias; otherwise bias is ``None``.
    """

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer norm.

        Args:
            x: Input tensor of any shape with last dim = ``ndim``.

        Returns:
            Normalised tensor of the same shape as *x*.
        """
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    """Position-wise feed-forward network (two-layer MLP with GELU).

    Expands the embedding dimension by 4× internally, applies GELU
    activation, projects back, and applies dropout — matching the
    original GPT-2 / nanoGPT design.

    Args:
        config: ``GPTConfig`` providing ``n_embd``, ``bias``, and ``dropout``.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape ``(B, T, n_embd)``.

        Returns:
            Output tensor of the same shape as *x*.
        """
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
