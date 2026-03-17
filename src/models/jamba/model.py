"""Jamba-style Hybrid Mamba-Attention SLM.

Interleaves Mamba SSM blocks (even layers) and causal self-attention blocks
(odd layers). The hypothesis is that SSM layers handle broad context at O(n)
cost while attention layers perform precise positional lookups when needed.

References:
- Lieber et al., 2024 — "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21 Labs)
- Glorioso et al., 2024 — "Zamba: A Compact 7B SSM Hybrid Model" (Zyphra)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.attention import CausalSelfAttention
from src.core.base import BaseSLM
from src.core.ffn import SwiGLU
from src.core.mamba_block import MambaBlock
from src.core.normalization import RMSNorm
from .config import JambaConfig


class _JambaAttention(nn.Module):
    """Thin wrapper around CausalSelfAttention for use inside HybridBlock.

    ``CausalSelfAttention`` was written for GPTConfig but only reads
    ``n_embd``, ``n_head``, ``dropout``, and ``bias`` — so we expose those.

    Args:
        config: ``JambaConfig``.
    """

    def __init__(self, config: JambaConfig) -> None:
        super().__init__()

        # Build a lightweight namespace that CausalSelfAttention reads
        class _Cfg:
            n_embd = config.n_embd
            n_head = config.n_head
            dropout = config.dropout
            bias = False
            block_size = config.block_size

        self.attn = CausalSelfAttention(_Cfg())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x)


class HybridBlock(nn.Module):
    """One hybrid block: pre-RMSNorm + (Mamba | Attention) + pre-RMSNorm + SwiGLU.

    Even-indexed layers use a MambaBlock mixer; odd-indexed layers use causal
    self-attention.  Both are followed by a shared SwiGLU FFN.

    Args:
        config: ``JambaConfig``.
        layer_idx: Position of this block in the stack (0-based).
    """

    def __init__(self, config: JambaConfig, layer_idx: int) -> None:
        super().__init__()
        self.uses_mamba = (layer_idx % 2 == 0)

        self.norm1 = RMSNorm(config.n_embd)
        if self.uses_mamba:
            self.mixer: nn.Module = MambaBlock(
                d_model=config.n_embd,
                d_state=config.mamba_d_state,
                d_conv=config.mamba_d_conv,
                expand=config.mamba_expand,
            )
        else:
            self.mixer = _JambaAttention(config)

        self.norm2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config.n_embd, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one hybrid block.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Output ``(B, T, C)``.
        """
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class JambaSLM(BaseSLM):
    """Jamba-style hybrid SLM (~35 M parameters).

    Architecture:
        token embedding → N × HybridBlock → RMSNorm → lm_head.

    Even-indexed blocks use Mamba SSM; odd-indexed blocks use causal
    self-attention. Weight tying between token embedding and LM head.

    Args:
        config: ``JambaConfig`` defining the model dimensions.
    """

    config_class = JambaConfig

    def __init__(self, config: JambaConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            HybridBlock(config, i) for i in range(config.n_layer)
        ])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
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
        """Forward pass through the Jamba hybrid model.

        Args:
            idx: Token indices ``(B, T)``.
            targets: Optional target indices ``(B, T)``.

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
        for block in self.blocks:
            x = block(x)
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
