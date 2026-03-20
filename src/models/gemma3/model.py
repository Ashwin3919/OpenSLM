"""Gemma 3 SLM: local/global attention + QK norm + logit soft-capping + GeGLU.

Key innovations vs. LLaMA baseline:

- **Interleaved local/global attention** — Most layers use sliding-window
  (local) attention; one layer in six uses full causal (global) attention.
  The 5:1 ratio is the default in Gemma 3. Local layers use a smaller RoPE
  theta; global layers use a much larger theta to handle longer-range positions.

- **QK normalisation** — RMSNorm is applied to query and key vectors per-head
  (over head_dim) *after* projection but *before* RoPE. This prevents
  attention logit explosion at scale without requiring gradient clipping
  adjustments.

- **Attention logit soft-capping** — Raw attention scores are passed through
  ``tanh(scores / cap) * cap`` before softmax. This bounds the pre-softmax
  values regardless of sequence length, stabilising training.

- **GeGLU FFN** — Uses GELU as the gate activation instead of SiLU (SwiGLU).
  Identical three-matrix structure: output = W2(GELU(W1(x)) * W3(x)).

- **Pre + post normalisation** — Each sub-block (attention and FFN) is
  wrapped with both a pre-norm and a post-norm RMSNorm. The post-norm is
  applied to the sub-block output *before* the residual addition:
  ``x = x + post_norm(sub_block(pre_norm(x)))``.

- **Final logit soft-capping** — Output logits are capped with tanh before
  cross-entropy (training) and before sampling (inference).

Reference: Gemma Team, Google DeepMind, 2025.
"Gemma 3 Technical Report." arXiv:2503.19786.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseSLM
from src.core.normalization import RMSNorm
from src.core.rope import apply_rotary_emb, precompute_freqs_cis
from .config import Gemma3Config


# ---------------------------------------------------------------------------
# GeGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class GeGLU(nn.Module):
    """Gated feed-forward block using GELU activation.

    Structure: ``output = W2( GELU(W1(x)) * W3(x) )``

    Identical to SwiGLU except the gate uses GELU instead of SiLU.  Three
    weight matrices, no bias terms.

    Args:
        dim: Input (and output) feature dimension.
        hidden_dim: Inner dimension for the gate and up projections.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the GeGLU transformation.

        Args:
            x: Input tensor of shape ``(..., dim)``.

        Returns:
            Output tensor of shape ``(..., dim)``.
        """
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Gemma 3 Attention
# ---------------------------------------------------------------------------


class Gemma3Attention(nn.Module):
    """Gemma 3 attention: GQA + QK norm + RoPE + attention logit soft-capping.

    Supports both local (sliding-window) and global (full causal) attention
    via a boolean mask passed at forward time.  The caller (Gemma3SLM)
    selects the correct mask and RoPE frequencies per layer.

    Args:
        config: ``Gemma3Config`` defining the model dimensions.
    """

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head
        self.attn_logit_cap = config.attn_logit_cap
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # QK normalisation — normalises over head_dim independently per head.
        # This is key: dim = head_dim, NOT n_embd.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention with QK norm, RoPE, soft-capping, and masking.

        Args:
            x: Input ``(B, T, C)``.
            freqs_cis: Precomputed RoPE frequencies ``(T, head_dim // 2)``.
                Local and global layers receive different precomputed tensors.
            mask: Boolean attention mask ``(block_size, block_size)``.
                ``True`` = attend, ``False`` = mask out.  Will be sliced to
                the actual sequence length T at forward time.

        Returns:
            Output tensor ``(B, T, C)``.
        """
        B, T, C = x.shape
        scale = 1.0 / math.sqrt(self.head_dim)

        # Project into Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        # QK normalisation (over head_dim, per head) — applied before RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Expand KV heads to match Q heads (GQA repeat)
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # (B, n_head, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, T, T)

        # Attention logit soft-capping — tanh bounds the values before softmax
        scores = torch.tanh(scores / self.attn_logit_cap) * self.attn_logit_cap

        # Apply causal / sliding-window mask
        # mask is (block_size, block_size); slice to current T
        scores = scores.masked_fill(
            ~mask[:T, :T].unsqueeze(0).unsqueeze(0),
            float("-inf"),
        )

        # Softmax + dropout
        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.dropout if self.training else 0.0)

        # Weighted sum of values
        y = torch.matmul(probs, v)           # (B, H, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Gemma 3 Block
# ---------------------------------------------------------------------------


class Gemma3Block(nn.Module):
    """Single Gemma 3 transformer block: pre+post norm + attention + GeGLU FFN.

    Normalisation pattern (applied to *both* sub-blocks):
        ``x = x + post_norm( sub_block( pre_norm(x) ) )``

    Args:
        config: ``Gemma3Config``.
    """

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.pre_attn_norm = RMSNorm(config.n_embd)
        self.attn = Gemma3Attention(config)
        self.post_attn_norm = RMSNorm(config.n_embd)

        self.pre_ffn_norm = RMSNorm(config.n_embd)
        self.ffn = GeGLU(config.n_embd, config.intermediate_size)
        self.post_ffn_norm = RMSNorm(config.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one Gemma 3 block.

        Args:
            x: Input ``(B, T, C)``.
            freqs_cis: RoPE frequencies for this layer (local or global theta).
            mask: Attention mask (sliding-window or full causal).

        Returns:
            Output tensor ``(B, T, C)``.
        """
        x = x + self.post_attn_norm(self.attn(self.pre_attn_norm(x), freqs_cis, mask))
        x = x + self.post_ffn_norm(self.ffn(self.pre_ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class Gemma3SLM(BaseSLM):
    """Gemma 3-style small language model (~29 M parameters).

    Architecture:
        token embedding → dropout → N × Gemma3Block → RMSNorm →
        lm_head (with final logit soft-capping).

    Each block is one of two types determined by ``config.global_layers``:
        - **Local block** (default): sliding-window attention with
          ``local_rope_theta`` and a window mask of size ``sliding_window``.
        - **Global block** (indices in ``global_layers``): full causal
          attention with ``global_rope_theta`` and a standard causal mask.

    Two sets of RoPE frequencies and two attention masks are precomputed at
    init and registered as non-persistent buffers (they move to the correct
    device automatically but are not saved in checkpoints).

    Args:
        config: ``Gemma3Config`` defining the model dimensions.
    """

    config_class = Gemma3Config

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            Gemma3Block(config) for _ in range(config.n_layer)
        ])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding and LM head share parameters
        self.wte.weight = self.lm_head.weight

        # Dual RoPE frequency buffers — one per attention type
        head_dim = config.n_embd // config.n_head
        freqs_local = precompute_freqs_cis(
            head_dim, config.block_size, config.local_rope_theta
        )
        freqs_global = precompute_freqs_cis(
            head_dim, config.block_size, config.global_rope_theta
        )
        self.register_buffer("freqs_cis_local", freqs_local, persistent=False)
        self.register_buffer("freqs_cis_global", freqs_global, persistent=False)

        # Attention masks — precomputed once, sliced at forward time
        local_mask = self._build_local_mask(config.block_size, config.sliding_window)
        causal_mask = torch.ones(
            config.block_size, config.block_size, dtype=torch.bool
        ).tril()
        self.register_buffer("local_mask", local_mask, persistent=False)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.apply(self._init_weights)

    @staticmethod
    def _build_local_mask(block_size: int, window: int) -> torch.Tensor:
        """Build a causal sliding-window boolean mask.

        A token at position i can attend to position j when:
          - j <= i  (causal: no future tokens)
          - i - j < window  (local: only the past ``window`` tokens)

        Args:
            block_size: Maximum sequence length.
            window: Sliding window size.

        Returns:
            Bool tensor ``(block_size, block_size)``: True = attend.
        """
        i = torch.arange(block_size).unsqueeze(1)   # (T, 1)
        j = torch.arange(block_size).unsqueeze(0)   # (1, T)
        return (j <= i) & ((i - j) < window)

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
        """Forward pass through the Gemma 3 model.

        Args:
            idx: Token indices ``(B, T)``.
            targets: Optional target indices ``(B, T)`` for loss computation.

        Returns:
            ``(logits, loss)`` — training returns ``(B, T, vocab_size)`` logits
            (after soft-capping) and scalar cross-entropy loss; generation
            returns ``(B, 1, vocab_size)`` and ``None``.

        Raises:
            AssertionError: If T > ``block_size``.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.drop(self.wte(idx))

        for i, block in enumerate(self.blocks):
            is_global = i in self.config.global_layers
            freqs_cis = (
                self.freqs_cis_global[:T] if is_global else self.freqs_cis_local[:T]
            )
            mask = self.causal_mask if is_global else self.local_mask
            x = block(x, freqs_cis, mask)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            # Final logit soft-capping before cross-entropy
            logits = torch.tanh(logits / self.config.final_logit_cap) * self.config.final_logit_cap
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = torch.tanh(logits / self.config.final_logit_cap) * self.config.final_logit_cap
            return logits, None

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
