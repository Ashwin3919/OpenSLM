"""SwiGLU Feed-Forward Network.

Used by LLaMA, DeepSeek-MoE (expert FFN), Jamba, and RetNet.

SwiGLU replaces the two-matrix GELU MLP with a three-matrix gated structure::

    output = W2( SiLU(W1(x)) * W3(x) )

The gate (W3) learns which features to pass through, giving the network more
expressive capacity for the same depth. The hidden dimension is typically
smaller than 4× to keep the parameter count comparable to a GELU MLP.

Reference: Noam Shazeer, 2020 — "GLU Variants Improve Transformer".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Gated feed-forward block using SiLU activation.

    Three linear projections: gate (w1), up (w3), and down (w2).
    No bias terms — consistent with LLaMA / modern practice.

    Args:
        dim: Input (and output) feature dimension.
        hidden_dim: Inner dimension for the up/gate projections.
            Typically smaller than 4 × dim because the gating effectively
            doubles the expressivity compared with a plain MLP.
        bias: Whether to add bias to the linear projections. Default False.
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)   # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)   # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)   # up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transformation.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor of shape ``(..., dim)``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
