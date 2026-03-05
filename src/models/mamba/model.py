"""Mamba SSM SLM: O(n) sequence model with no attention.

Architecture:
    token embedding → N × MambaLayer → RMSNorm → lm_head

Each MambaLayer is a pre-norm residual wrap around the MambaBlock from
``src.core.mamba_block``. There are no positional embeddings — the sequential
state space is inherently positional.

Reference: Gu & Dao, 2023 — "Mamba: Linear-Time Sequence Modeling with
Selective State Spaces".
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseSLM
from src.core.mamba_block import MambaBlock
from src.core.normalization import RMSNorm
from .config import MambaConfig


class MambaLayer(nn.Module):
    """Pre-RMSNorm residual block wrapping one MambaBlock.

    Args:
        config: ``MambaConfig``.
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = MambaBlock(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-norm Mamba residual.

        Args:
            x: Input ``(B, T, d_model)``.

        Returns:
            Output ``(B, T, d_model)``.
        """
        return x + self.mixer(self.norm(x))


class MambaSLM(BaseSLM):
    """Mamba-based small language model (~32 M parameters).

    Replaces self-attention entirely with selective state spaces.
    Uses twice as many layers as the GPT baseline because each Mamba
    block is cheaper than one transformer block.

    Args:
        config: ``MambaConfig`` defining the model dimensions.
    """

    config_class = MambaConfig

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MambaLayer(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the Mamba model.

        Args:
            idx: Token indices ``(B, T)``.
            targets: Optional target indices ``(B, T)`` for loss computation.

        Returns:
            ``(logits, loss)`` in training mode; ``(logits, None)`` for generation.

        Raises:
            AssertionError: If *T* > ``block_size``.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.drop(self.wte(idx))
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
