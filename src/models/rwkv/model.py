"""RWKV-style SLM: linear attention RNN.

RWKV is the "best of both worlds" architecture:
- **Trains like a transformer**: all positions can be computed in parallel using
  cumulative sums (WKV mechanism).
- **Infers like an RNN**: O(1) per step — the hidden state is compressed into a
  fixed-size vector, enabling constant-memory generation.

Key components:
- **TimeMix**: token-shifted WKV weighted key-value with a per-head learned
  decay and a current-token bonus, replacing causal self-attention.
- **ChannelMix**: token-shifted gated FFN with squared-ReLU, replacing the MLP.
- **Token shift**: each sub-block mixes the current token embedding with the
  previous token embedding using learnable interpolation coefficients, giving
  the model implicit positional context without explicit positional embeddings.

Reference: Peng et al., 2023 — "RWKV: Reinventing RNNs for the Transformer Era".
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseSLM
from src.core.normalization import RMSNorm
from .config import RWKVConfig


@torch.jit.script
def _wkv_forward(
    k: torch.Tensor,
    v: torch.Tensor,
    exp_w: torch.Tensor,
    exp_u: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled WKV sequential scan (kept as reference / fallback).

    Eliminates Python interpreter dispatch overhead by compiling the loop.
    Precomputed ``exp(k)`` and ``exp(k)*v`` avoid repeated exponential ops
    inside the loop.  Numerically identical to the original sequential scan.

    Args:
        k: Token-shifted key projections ``(B, T, H, head_dim)``.
        v: Token-shifted value projections ``(B, T, H, head_dim)``.
        exp_w: Per-head decay factor ``(1, H, 1)`` = ``exp(time_decay)``.
        exp_u: Per-head current-token bonus ``(1, H, 1)`` = ``exp(time_first)``.

    Returns:
        WKV output ``(B, T, H, head_dim)``.
    """
    B: int = k.shape[0]
    T: int = k.shape[1]
    H: int = k.shape[2]
    D: int = k.shape[3]

    exp_k = torch.exp(k)                                              # (B, T, H, D)
    exp_ek_v = exp_k * v                                              # (B, T, H, D)

    state_a = torch.zeros(B, H, D, dtype=k.dtype, device=k.device)
    state_b = torch.zeros(B, H, 1, dtype=k.dtype, device=k.device)
    out = torch.zeros(B, T, H, D, dtype=k.dtype, device=k.device)

    for t in range(T):
        kt = exp_k[:, t]                                              # (B, H, D)
        exp_u_k = exp_u * kt                                          # (B, H, D)
        num = state_a + exp_u_k * v[:, t]
        den = (state_b + exp_u_k.sum(dim=-1, keepdim=True)).clamp(min=1e-6)
        out[:, t] = num / den
        state_a = exp_w * state_a + exp_ek_v[:, t]
        state_b = exp_w * state_b + kt.sum(dim=-1, keepdim=True)

    return out


def _wkv_parallel(
    k: torch.Tensor,
    v: torch.Tensor,
    exp_w: torch.Tensor,
    exp_u: torch.Tensor,
) -> torch.Tensor:
    """Loop-free WKV scan via exponentially-decayed cumulative sum.

    Because ``exp_w`` is a **constant** per-head scalar (a learned parameter,
    not a function of the input), the linear recurrence

        state[t] = exp_w * state[t-1] + b[t-1]

    has a closed-form solution as a weighted prefix sum:

        state[t] = exp_w^t  *  Σ_{s<t}  (b[s] / exp_w^{s+1})
                 = exp_w^t  *  exclusive_cumsum( b[s] / exp_w^{s+1} )

    This replaces the entire ``for t in range(T)`` loop with five vectorised
    tensor ops (log, exp, cumsum, cat, div) — all O(T), no Python loop.
    Numerically equivalent to ``_wkv_forward`` for typical model parameters.

    Args:
        k: ``(B, T, H, D)`` token-shifted keys.
        v: ``(B, T, H, D)`` token-shifted values.
        exp_w: ``(1, H, 1)`` per-head decay = ``exp(-exp(time_decay))``.
        exp_u: ``(1, H, 1)`` per-head current-token bonus = ``exp(time_first)``.

    Returns:
        WKV output ``(B, T, H, D)``.
    """
    B, T, H, D = k.shape

    # ── Step 1: precompute exp(k) and exp(k)*v ─────────────────────────
    exp_k = torch.exp(k)          # (B, T, H, D)
    exp_ek_v = exp_k * v          # (B, T, H, D)

    # ── Step 2: build alpha^t and alpha^{t+1} for t = 0..T-1 ──────────
    # Use log-space to avoid underflow: alpha^t = exp(t * log(alpha))
    # exp_w in (0,1) always, so log(exp_w) < 0.
    t_idx = torch.arange(T, device=k.device, dtype=k.dtype)          # (T,)
    log_alpha = torch.log(exp_w).view(1, 1, H, 1)                    # (1,1,H,1)
    alpha_pow = torch.exp(t_idx.view(1, T, 1, 1) * log_alpha)        # (1,T,H,1)  = alpha^t
    alpha_next_pow = alpha_pow * exp_w.view(1, 1, H, 1)               # (1,T,H,1)  = alpha^{t+1}

    # ── Step 3: normalize inputs ────────────────────────────────────────
    # f_a[s] = exp_ek_v[s] / alpha^{s+1}  so that
    # state_a[t] = alpha^t * Σ_{s<t} f_a[s]
    f_a = exp_ek_v / alpha_next_pow                                   # (B,T,H,D)
    f_b = exp_k.sum(-1, keepdim=True) / alpha_next_pow                # (B,T,H,1)

    # ── Step 4: exclusive cumsum → Σ_{s<t} f[s] ────────────────────────
    cs_a = f_a.cumsum(dim=1)                                          # inclusive
    cs_b = f_b.cumsum(dim=1)
    # shift right: excl[t] = cs[t-1], excl[0] = 0
    excl_a = torch.cat([torch.zeros_like(cs_a[:, :1]), cs_a[:, :-1]], dim=1)  # (B,T,H,D)
    excl_b = torch.cat([torch.zeros_like(cs_b[:, :1]), cs_b[:, :-1]], dim=1)  # (B,T,H,1)

    # ── Step 5: recover states and compute output ───────────────────────
    # nan_to_num guards the 0*inf=NaN edge case when alpha underflows to 0
    # in float32 for pathologically large time_decay values (>~1.7 for T=128).
    # This never occurs with normal initialisation (time_decay ~ randn-5).
    state_a = (alpha_pow * excl_a).nan_to_num(nan=0.0, posinf=0.0)  # (B,T,H,D)
    state_b = (alpha_pow * excl_b).nan_to_num(nan=0.0, posinf=0.0)  # (B,T,H,1)

    exp_u_k = exp_u * exp_k        # (B,T,H,D)
    num = state_a + exp_u_k * v                                        # (B,T,H,D)
    den = (state_b + exp_u_k.sum(-1, keepdim=True)).clamp(min=1e-6)   # (B,T,H,1)
    return num / den                                                    # (B,T,H,D)


class RWKV_TimeMix(nn.Module):
    """Time-mixing sub-block: WKV weighted key-value recurrence.

    For training (parallel mode), the WKV is computed with a sequential scan
    that matches the recurrent formulation. At inference time the recurrent
    state can be maintained externally for O(1) per-step cost (not exposed
    here — left as an exercise / future optimisation).

    Args:
        config: ``RWKVConfig``.
    """

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        # Learnable token-shift mixing coefficients (one per channel)
        self.mix_k = nn.Parameter(torch.full((1, 1, config.n_embd), 0.5))
        self.mix_v = nn.Parameter(torch.full((1, 1, config.n_embd), 0.5))
        self.mix_r = nn.Parameter(torch.full((1, 1, config.n_embd), 0.5))

        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.output = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # Per-head decay and current-token bonus (learned)
        self.time_decay = nn.Parameter(torch.randn(config.n_head) - 5.0)
        self.time_first = nn.Parameter(torch.randn(config.n_head))

    def _wkv_fast(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
        C: int,
    ) -> torch.Tensor:
        """Vectorised WKV scan — no Python loop (default path).

        Delegates to the module-level ``_wkv_parallel`` function which
        replaces the sequential recurrence with five vectorised tensor ops
        using an exponentially-decayed cumulative sum.

        Args:
            k: Key ``(B, T, C)``.
            v: Value ``(B, T, C)``.
            B, T, C: Batch size, sequence length, channel dim.

        Returns:
            WKV output ``(B, T, C)``.
        """
        H = self.n_head
        head_dim = C // H
        k = k.view(B, T, H, head_dim)
        v = v.view(B, T, H, head_dim)
        exp_w = torch.exp(-torch.exp(self.time_decay)).view(1, H, 1)
        exp_u = torch.exp(self.time_first).view(1, H, 1)
        out = _wkv_parallel(k, v, exp_w, exp_u)                       # (B, T, H, D)
        return out.view(B, T, C)

    def _wkv_sequential(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
        C: int,
    ) -> torch.Tensor:
        """JIT-compiled WKV sequential scan — kept as reference / fallback.

        Delegates to the module-level ``_wkv_forward`` TorchScript function.

        Args:
            k: Key ``(B, T, C)``.
            v: Value ``(B, T, C)``.
            B, T, C: Batch size, sequence length, channel dim.

        Returns:
            WKV output ``(B, T, C)``.
        """
        H = self.n_head
        head_dim = C // H
        k = k.view(B, T, H, head_dim)
        v = v.view(B, T, H, head_dim)
        exp_w = torch.exp(-torch.exp(self.time_decay)).view(1, H, 1)
        exp_u = torch.exp(self.time_first).view(1, H, 1)
        out = _wkv_forward(k, v, exp_w, exp_u)                        # (B, T, H, D)
        return out.view(B, T, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time-mixing.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Output ``(B, T, C)``.
        """
        B, T, C = x.shape
        # Token shift: mix current token with the previous one
        x_prev = F.pad(x, (0, 0, 1, -1))         # shift right by 1 position

        k = self.key(x * self.mix_k + x_prev * (1.0 - self.mix_k))
        v = self.value(x * self.mix_v + x_prev * (1.0 - self.mix_v))
        r = self.receptance(x * self.mix_r + x_prev * (1.0 - self.mix_r))
        r = torch.sigmoid(r)                       # receptance gate in [0, 1]

        wkv = self._wkv_fast(k, v, B, T, C)       # parallel scan (no loop)
        return self.output(r * wkv)


class RWKV_ChannelMix(nn.Module):
    """Channel-mixing sub-block: token-shifted gated FFN.

    Uses squared-ReLU (relu²) as the activation, which is sparser and more
    efficient than GELU for this architecture.

    Args:
        config: ``RWKVConfig``.
    """

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        hidden_dim = config.n_embd * config.ffn_mult
        self.mix_k = nn.Parameter(torch.full((1, 1, config.n_embd), 0.5))
        self.mix_r = nn.Parameter(torch.full((1, 1, config.n_embd), 0.5))
        self.key = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.receptance = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-mixing.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Output ``(B, T, C)``.
        """
        x_prev = F.pad(x, (0, 0, 1, -1))
        k = self.key(x * self.mix_k + x_prev * (1.0 - self.mix_k))
        r = self.receptance(x * self.mix_r + x_prev * (1.0 - self.mix_r))
        return torch.sigmoid(r) * self.value(F.relu(k) ** 2)   # squared ReLU


class RWKVBlock(nn.Module):
    """One RWKV block: pre-RMSNorm + TimeMix + pre-RMSNorm + ChannelMix.

    Args:
        config: ``RWKVConfig``.
    """

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.time_mix = RWKV_TimeMix(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.channel_mix = RWKV_ChannelMix(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one RWKV block.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            Output ``(B, T, C)``.
        """
        x = x + self.time_mix(self.norm1(x))
        x = x + self.channel_mix(self.norm2(x))
        return x


class RWKVSLM(BaseSLM):
    """RWKV-based small language model (~33 M parameters).

    Architecture:
        token embedding → N × RWKVBlock → RMSNorm → lm_head.

    No positional embeddings: token-shift provides implicit positional context.
    Weight tying between the token embedding and LM head is applied.

    Args:
        config: ``RWKVConfig`` defining the model dimensions.
    """

    config_class = RWKVConfig

    def __init__(self, config: RWKVConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([RWKVBlock(config) for _ in range(config.n_layer)])
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
        """Forward pass through the RWKV model.

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
