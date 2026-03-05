"""RetNet SLM: multi-scale exponential decay as a replacement for softmax attention.

RetNet replaces the O(n²) softmax attention with a linear retention mechanism
that uses exponential decay masks. The retention score between position i and j
is ``gamma^(i - j)`` for ``i >= j`` (causal) and 0 otherwise — no softmax.

This gives three equivalent computation modes:
- **Parallel**: matrix form (used for training here).
- **Recurrent**: O(1) per step (inference efficiency, not implemented here).
- **Chunkwise**: block-diagonal parallel (balance of both, not implemented here).

Reference: Sun et al., 2023 — "Retentive Network: A Successor to Transformer
for Large Language Models" (Microsoft Research).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseSLM
from src.core.ffn import SwiGLU
from src.core.normalization import RMSNorm
from .config import RetNetConfig


class MultiScaleRetention(nn.Module):
    """Multi-scale retention: linear causal attention with per-head decay.

    Each head uses a different γ ∈ [gamma_min, gamma_max] (log-spaced), so
    different heads attend to different temporal scales.

    Args:
        config: ``RetNetConfig``.
    """

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # GroupNorm on head outputs (RetNet paper convention)
        self.group_norm = nn.GroupNorm(config.n_head, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Per-head decay rates γ, log-spaced in [gamma_min, gamma_max]
        gammas = 1.0 - torch.exp(
            torch.linspace(
                math.log(1.0 - config.gamma_max),
                math.log(1.0 - config.gamma_min),
                config.n_head,
            )
        )
        self.register_buffer("gammas", gammas)  # (n_head,)

    def _build_decay_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Build the causal decay matrix D of shape ``(1, n_head, T, T)``.

        ``D[h, i, j] = gamma_h^(i - j)``  for ``i >= j``, else 0.

        Args:
            T: Sequence length.
            device: Target device.

        Returns:
            Decay mask ``(1, n_head, T, T)`` on *device*.
        """
        positions = torch.arange(T, device=device, dtype=torch.float)
        dist = positions.unsqueeze(1) - positions.unsqueeze(0)          # (T, T)
        causal = (dist >= 0).float()                                     # (T, T)
        # gammas: (n_head,) → (1, n_head, 1, 1)
        g = self.gammas.view(1, self.n_head, 1, 1)
        # dist: (T, T) → (1, 1, T, T)
        d = dist.unsqueeze(0).unsqueeze(0)
        D = (g ** d) * causal.unsqueeze(0).unsqueeze(0)                 # (1, H, T, T)
        return D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale retention.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Output ``(B, T, C)``.
        """
        B, T, C = x.shape
        H = self.n_head
        d = self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)   # (B, H, T, d)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        D = self._build_decay_mask(T, x.device)                # (1, H, T, T)

        # Retention = (Q @ K^T) ⊙ D  @ V   (no softmax)
        retention = (q @ k.transpose(-2, -1)) * D              # (B, H, T, T)
        y = self.drop(retention) @ v                           # (B, H, T, d)

        y = y.transpose(1, 2).contiguous().view(B, T, C)       # (B, T, C)
        # GroupNorm normalises across n_head groups (RetNet convention)
        y = self.group_norm(y.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(y)


class RetNetBlock(nn.Module):
    """One RetNet block: pre-RMSNorm + MSR + pre-RMSNorm + SwiGLU.

    Args:
        config: ``RetNetConfig``.
    """

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.retention = MultiScaleRetention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config.n_embd, config.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one RetNet block.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Output ``(B, T, C)``.
        """
        x = x + self.retention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RetNetSLM(BaseSLM):
    """RetNet-style small language model (~29 M parameters).

    Architecture:
        token embedding → N × RetNetBlock → RMSNorm → lm_head.

    No positional embeddings (retention decay encodes relative position).
    Weight tying between the token embedding and LM head.

    Args:
        config: ``RetNetConfig`` defining the model dimensions.
    """

    config_class = RetNetConfig

    def __init__(self, config: RetNetConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([RetNetBlock(config) for _ in range(config.n_layer)])
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
        """Forward pass through the RetNet model.

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
