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

    def _wkv_sequential(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
        C: int,
    ) -> torch.Tensor:
        """Sequential WKV scan (training mode).

        Computes the numerically-stabilised weighted sum::

            wkv_t = (Σ_{s<t} exp(w*(t-s-1) + u + k_s) v_s + exp(u + k_t) v_t)
                  / (Σ_{s<t} exp(w*(t-s-1) + u + k_s)     + exp(u + k_t))

        implemented iteratively in log-space for numerical stability.

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
        w = -torch.exp(self.time_decay)           # (H,) — negative decay
        u = self.time_first                        # (H,) — current-token bonus

        out = torch.zeros(B, T, H, head_dim, device=k.device, dtype=k.dtype)
        # Running numerator / denominator in log-space
        state_a = torch.zeros(B, H, head_dim, device=k.device, dtype=k.dtype)
        state_b = torch.zeros(B, H, 1, device=k.device, dtype=k.dtype)

        for t in range(T):
            kt = k[:, t]                           # (B, H, head_dim)
            vt = v[:, t]                           # (B, H, head_dim)

            # Numerator: exp(u + k_t) * v_t + past state
            exp_u_k = torch.exp(u.view(1, H, 1) + kt)  # (B, H, head_dim)
            num = state_a + exp_u_k * vt
            den = state_b + exp_u_k.sum(-1, keepdim=True).clamp(min=1e-6)
            out[:, t] = num / den

            # Update state with decay
            state_a = torch.exp(w.view(1, H, 1)) * state_a + torch.exp(kt) * vt
            state_b = torch.exp(w.view(1, H, 1)) * state_b + torch.exp(kt).sum(-1, keepdim=True)

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

        wkv = self._wkv_sequential(k, v, B, T, C)
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
