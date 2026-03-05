"""BitNet 1.58-bit SLM: ternary-weight language model.

Each weight matrix in the linear layers is constrained to {-1, 0, +1} during
the forward pass, while activations are quantized to 8-bit integers. Both
quantizations use the Straight-Through Estimator (STE) so gradients flow
through as if quantization did not occur.

At full precision (16-bit storage) this model has ~30 M parameters and ~60 MB
on disk. With proper 1.58-bit packing it shrinks to ~6 MB. The purpose here is
to train and evaluate quality — not to demonstrate compressed storage.

Reference: Ma et al., 2024 — "The Era of 1-bit LLMs: All Large Language Models
are in 1.58 Bits".
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseSLM
from src.core.normalization import RMSNorm
from src.core.rope import apply_rotary_emb, precompute_freqs_cis
from .config import BitNetConfig


# ---------------------------------------------------------------------------
# BitLinear
# ---------------------------------------------------------------------------


class BitLinear(nn.Module):
    """1.58-bit linear layer with Straight-Through Estimator.

    During the forward pass:
    1. The input is pre-normalised with RMSNorm (required for stable quantization).
    2. Activations are quantized to int8 range via STE.
    3. Weights are quantized to {-1, 0, +1} via STE.
    4. A standard ``F.linear`` is computed with the quantized tensors.

    Gradients flow through both quantization steps unchanged (STE).
    Full-precision weights are kept in ``self.weight`` for the optimizer.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to add a bias term (default False — BitNet convention).
        activation_bits: Bit-width for activation quantization (default 8).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8,
    ) -> None:
        super().__init__()
        self.activation_bits = activation_bits
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rms_norm = RMSNorm(in_features)

    def _ternary_weight(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize *w* to {-1, 0, +1} with STE.

        The scale factor is the mean absolute value of the weight matrix,
        providing a per-tensor absmean quantization.

        Args:
            w: Full-precision weight ``(out, in)``.

        Returns:
            Ternary-quantized weight with gradient straight-through.
        """
        scale = w.abs().mean().clamp(min=1e-8)
        w_ternary = torch.clamp(torch.round(w / scale), -1.0, 1.0)
        return w + (w_ternary - w).detach()   # STE

    def _quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize *x* to signed integer range with STE.

        Per-token (last-dim) absolute-max scaling keeps the quantization
        range tight for each token independently.

        Args:
            x: Input activations ``(..., in_features)``.

        Returns:
            Quantized activations with gradient straight-through.
        """
        Qp = 2 ** (self.activation_bits - 1) - 1
        Qn = -(2 ** (self.activation_bits - 1))
        scale = Qp / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * scale).round().clamp(Qn, Qp) / scale
        return x + (x_q - x).detach()   # STE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BitLinear transformation.

        Args:
            x: Input ``(..., in_features)``.

        Returns:
            Output ``(..., out_features)``.
        """
        x = self.rms_norm(x)
        x = self._quantize_activations(x)
        w = self._ternary_weight(self.weight)
        return F.linear(x, w, self.bias_param)


# ---------------------------------------------------------------------------
# Reuse LLaMA-style GQA + SwiGLU with BitLinear projections
# ---------------------------------------------------------------------------


class _BitSwiGLU(nn.Module):
    """SwiGLU FFN with BitLinear projections."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = BitLinear(dim, hidden_dim)
        self.w2 = BitLinear(hidden_dim, dim)
        self.w3 = BitLinear(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class _BitGQAttention(nn.Module):
    """Grouped Query Attention with BitLinear projections and RoPE."""

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head
        self.drop = nn.Dropout(config.dropout)

        self.q_proj = BitLinear(config.n_embd, config.n_head * self.head_dim)
        self.k_proj = BitLinear(config.n_embd, config.n_kv_head * self.head_dim)
        self.v_proj = BitLinear(config.n_embd, config.n_kv_head * self.head_dim)
        self.out_proj = BitLinear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.out_proj(y))


class BitNetBlock(nn.Module):
    """One BitNet block: pre-RMSNorm + BitGQA + pre-RMSNorm + BitSwiGLU."""

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = _BitGQAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.ffn = _BitSwiGLU(config.n_embd, config.intermediate_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class BitNetSLM(BaseSLM):
    """BitNet 1.58b-style small language model (~30 M parameters, ~6 MB packed).

    Identical structure to LLamaSLM but every ``nn.Linear`` inside attention
    and FFN is replaced by ``BitLinear``. Embeddings and the LM head use
    standard full-precision ``nn.Linear`` / ``nn.Embedding``.

    Args:
        config: ``BitNetConfig`` defining the model dimensions.
    """

    config_class = BitNetConfig

    def __init__(self, config: BitNetConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings and LM head remain full-precision (BitNet convention)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([BitNetBlock(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        head_dim = config.n_embd // config.n_head
        freqs = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear) and not isinstance(module, BitLinear):
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
        """Forward pass through the BitNet model.

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
