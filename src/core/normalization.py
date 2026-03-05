"""RMSNorm: Root Mean Square Layer Normalisation.

Used by LLaMA, DeepSeek-MoE, Mamba, RWKV, Jamba, BitNet, and RetNet.
Unlike LayerNorm, RMSNorm omits the mean-centering step and the bias term,
giving a simpler and slightly faster normalisation.

Reference: Zhang & Sennrich, 2019 — "Root Mean Square Layer Normalization".
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square normalisation without mean-centering or bias.

    Computes::

        y = x / RMS(x) * weight

    where ``RMS(x) = sqrt(mean(x²) + eps)``.

    Args:
        dim: Feature dimension to normalise over (last dimension of input).
        eps: Small constant added for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise *x* along its last dimension.

        Args:
            x: Input tensor of any shape ``(..., dim)``.

        Returns:
            Normalised tensor of the same shape as *x*.
        """
        # Cast to float32 for numerical stability then return in original dtype
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight
