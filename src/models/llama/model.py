"""LLaMA-style SLM: RMSNorm + RoPE + GQA + SwiGLU.

Key differences from GPT-2 baseline:
- RMSNorm instead of LayerNorm (no mean-centering, no bias)
- Rotary Position Embeddings instead of learned position embeddings
- Grouped Query Attention (fewer KV heads than Q heads — saves memory)
- SwiGLU FFN instead of GELU MLP (gated, three weight matrices)
- No bias terms in Linear layers
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseSLM
from src.core.ffn import SwiGLU
from src.core.normalization import RMSNorm
from src.core.rope import apply_rotary_emb, precompute_freqs_cis
from .config import LlamaConfig


class GQAttention(nn.Module):
    """Grouped Query Attention with Rotary Position Embeddings.

    ``n_kv_head`` KV projections are each shared by ``n_head // n_kv_head``
    query heads, reducing the KV cache size at inference while retaining most
    of the expressivity of full multi-head attention.

    Args:
        config: ``LlamaConfig`` defining the model dimensions.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        assert config.n_head % config.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head  # repeat factor for GQA

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """Compute GQA with RoPE.

        Args:
            x: Input ``(B, T, C)``.
            freqs_cis: Precomputed RoPE frequencies ``(max_len, head_dim // 2)``.

        Returns:
            Output tensor ``(B, T, C)``.
        """
        B, T, C = x.shape

        # Project
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply RoPE to q and k
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Expand KV heads to match Q heads (repeat_interleave along head dim)
        # k, v: (B, T, n_kv_head, head_dim) → (B, T, n_head, head_dim)
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose to (B, n_head, T, head_dim) for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention (PyTorch ≥ 2.0) with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.out_proj(y))


class LlamaBlock(nn.Module):
    """Single LLaMA transformer block: pre-RMSNorm + GQA + pre-RMSNorm + SwiGLU.

    Args:
        config: ``LlamaConfig``.
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = GQAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.ffn = SwiGLU(config.n_embd, config.intermediate_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """Apply one LLaMA block.

        Args:
            x: Input ``(B, T, C)``.
            freqs_cis: RoPE frequencies passed through from the model.

        Returns:
            Output tensor ``(B, T, C)``.
        """
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x


class LlamaSLM(BaseSLM):
    """LLaMA-style small language model (~31 M parameters).

    Architecture:
        token embedding (no position embedding — RoPE handles positions) →
        dropout → N × LlamaBlock → RMSNorm → lm_head.

    Weight tying between the token embedding and LM head is applied following
    the same convention as the GPT baseline.

    Args:
        config: ``LlamaConfig`` defining the model dimensions.
    """

    config_class = LlamaConfig

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([LlamaBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Precompute RoPE frequencies (registered as buffer so they move to device)
        head_dim = config.n_embd // config.n_head
        freqs = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise weights with N(0, 0.02); zero-init biases."""
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
        """Forward pass through the LLaMA model.

        Args:
            idx: Token indices ``(B, T)``.
            targets: Optional target indices ``(B, T)`` for loss computation.

        Returns:
            ``(logits, loss)`` — training mode returns ``(B, T, vocab_size)`` logits
            and a scalar loss; generation mode returns ``(B, 1, vocab_size)`` and
            ``None``.

        Raises:
            AssertionError: If *T* > ``block_size``.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.drop(self.wte(idx))
        freqs_cis = self.freqs_cis[:T]

        for block in self.blocks:
            x = block(x, freqs_cis)
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
