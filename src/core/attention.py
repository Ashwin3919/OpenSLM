"""Causal multi-head self-attention.

Uses ``F.scaled_dot_product_attention`` (Flash Attention) when available in
the installed PyTorch version, and falls back to an explicit masked attention
implementation otherwise — exactly matching the original notebook behaviour.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional Flash Attention.

    The causal mask ensures that position *t* can only attend to positions
    ≤ *t*, enforcing autoregressive generation.

    Args:
        config: ``GPTConfig`` with ``n_embd``, ``n_head``, ``block_size``,
            ``bias``, and ``dropout``.

    Raises:
        AssertionError: If ``n_embd`` is not divisible by ``n_head``.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0, (
            f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
        )
        # Fused QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Fallback: register causal mask as a buffer so it moves with the model
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute causal self-attention.

        Args:
            x: Input tensor of shape ``(B, T, C)`` where ``C = n_embd``.

        Returns:
            Output tensor of shape ``(B, T, C)``.
        """
        B, T, C = x.size()
        head_size = C // self.n_head

        # Project to Q, K, V and reshape to (B, n_head, T, head_size)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Re-assemble heads and project back to embedding dimension
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
