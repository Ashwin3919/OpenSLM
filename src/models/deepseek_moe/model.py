"""DeepSeek-style Mixture-of-Experts SLM.

Key innovations vs. standard MoE:
- **Shared expert**: One SwiGLU expert is always active, ensuring a stable
  gradient path and consistent baseline computation.
- **Fine-grained routed experts**: Many small experts (rather than few large
  ones) with a top-k router gives dense coverage of the expert space.
- **Dense early layers**: The first ``dense_layers`` use a standard SwiGLU
  FFN; MoE is applied to subsequent layers where specialisation matters more.
- **Load-balancing loss**: Auxiliary loss encouraging uniform expert utilisation
  is added to the cross-entropy loss during training.

Reference: DeepSeek-V2 technical report (2024).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseSLM
from src.core.ffn import SwiGLU
from src.core.normalization import RMSNorm
from src.core.rope import apply_rotary_emb, precompute_freqs_cis
from .config import DeepSeekMoEConfig


# ---------------------------------------------------------------------------
# GQA (same as LLaMA — duplicated here to keep each plugin self-contained)
# ---------------------------------------------------------------------------


class _GQAttention(nn.Module):
    """Grouped Query Attention with RoPE (internal to this plugin)."""

    def __init__(self, config: DeepSeekMoEConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.n_rep = config.n_head // config.n_kv_head

        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.drop = nn.Dropout(config.dropout)

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


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class TopKRouter(nn.Module):
    """Soft top-k router: selects the highest-scoring experts per token.

    Args:
        n_embd: Input feature dimension.
        n_experts: Total number of routed experts.
        top_k: Number of experts to activate per token.
    """

    def __init__(self, n_embd: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, n_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights and indices.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            ``(weights, indices, router_logits)`` where:
            - *weights* ``(B, T, top_k)``: softmax-normalised weights for selected experts.
            - *indices* ``(B, T, top_k)``: indices of the selected experts.
            - *router_logits* ``(B, T, n_experts)``: raw logits used for the aux loss.
        """
        logits = self.gate(x)                               # (B, T, n_experts)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        return weights, top_k_indices, logits


# ---------------------------------------------------------------------------
# Load-balancing auxiliary loss
# ---------------------------------------------------------------------------


def load_balancing_loss(
    router_logits: torch.Tensor,
    top_k_indices: torch.Tensor,
    n_experts: int,
) -> torch.Tensor:
    """Encourage uniform expert utilisation across tokens.

    The loss is the dot product of the average routing probability and the
    average expert frequency, scaled by *n_experts* so that a uniform
    distribution produces a loss of 1.0.

    Args:
        router_logits: Raw logits ``(B, T, n_experts)``.
        top_k_indices: Selected expert indices ``(B, T, top_k)``.
        n_experts: Total number of routed experts.

    Returns:
        Scalar auxiliary loss tensor.
    """
    probs = F.softmax(router_logits, dim=-1)            # (B, T, n_experts)
    avg_probs = probs.mean(dim=[0, 1])                  # (n_experts,)

    # Fraction of tokens routed to each expert
    freq = torch.zeros(n_experts, device=router_logits.device, dtype=torch.float)
    freq.scatter_add_(
        0,
        top_k_indices.reshape(-1),
        torch.ones(top_k_indices.numel(), device=router_logits.device),
    )
    freq = freq / top_k_indices.numel()

    return (avg_probs * freq).sum() * n_experts


# ---------------------------------------------------------------------------
# MoE layer
# ---------------------------------------------------------------------------


class MoELayer(nn.Module):
    """Mixture-of-Experts FFN layer (DeepSeek-style).

    One shared expert is always applied.  ``top_k`` of the routed experts are
    additionally selected per token, their outputs weighted by the router and
    summed together with the shared expert output.

    Args:
        config: ``DeepSeekMoEConfig``.
    """

    def __init__(self, config: DeepSeekMoEConfig) -> None:
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.top_k

        # Shared expert — always active
        self.shared_expert = SwiGLU(config.n_embd, config.expert_hidden_dim)

        # Routed experts — only top-k active per token
        self.experts = nn.ModuleList([
            SwiGLU(config.n_embd, config.expert_hidden_dim)
            for _ in range(config.n_routed_experts)
        ])

        self.router = TopKRouter(config.n_embd, config.n_routed_experts, config.top_k)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MoE layer.

        Args:
            x: Input ``(B, T, C)``.

        Returns:
            ``(output, router_logits)`` where *output* is ``(B, T, C)`` and
            *router_logits* is ``(B, T, n_routed_experts)`` for aux loss.
        """
        # Shared expert — applies to all tokens
        shared_out = self.shared_expert(x)

        # Routed experts
        weights, indices, router_logits = self.router(x)   # (B, T, top_k)
        B, T, C = x.shape
        expert_out = torch.zeros_like(x)

        for i, expert in enumerate(self.experts):
            # Mask: which (batch, seq) positions selected this expert
            mask = (indices == i).any(dim=-1)              # (B, T) bool
            if not mask.any():
                continue
            # Weight for expert i: sum of routing weights where index == i
            w = (weights * (indices == i).float()).sum(dim=-1)  # (B, T)
            expert_out[mask] += w[mask].unsqueeze(-1) * expert(x[mask])

        return shared_out + expert_out, router_logits


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class DeepSeekBlock(nn.Module):
    """One transformer block: GQA + dense or MoE FFN.

    Args:
        config: ``DeepSeekMoEConfig``.
        layer_idx: Block index. Determines whether to use dense or MoE FFN.
    """

    def __init__(self, config: DeepSeekMoEConfig, layer_idx: int) -> None:
        super().__init__()
        self.is_dense = layer_idx in config.dense_layers

        self.norm1 = RMSNorm(config.n_embd)
        self.attn = _GQAttention(config)
        self.norm2 = RMSNorm(config.n_embd)

        if self.is_dense:
            self.ffn: nn.Module = SwiGLU(config.n_embd, config.intermediate_size)
        else:
            self.ffn = MoELayer(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply one block.

        Returns:
            ``(x, router_logits_or_None)`` — MoE blocks return router logits
            for the auxiliary loss; dense blocks return ``None``.
        """
        x = x + self.attn(self.norm1(x), freqs_cis)
        if self.is_dense:
            x = x + self.ffn(self.norm2(x))
            return x, None
        else:
            ffn_out, router_logits = self.ffn(self.norm2(x))
            x = x + ffn_out
            return x, router_logits


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class DeepSeekMoESLM(BaseSLM):
    """DeepSeek-style MoE small language model (~48 M total, ~25 M active).

    Architecture:
        token embedding → N × DeepSeekBlock → RMSNorm → lm_head.

    Dense layers use SwiGLU.  MoE layers add routing logits to the return
    value, which are accumulated and used to compute the load-balancing
    auxiliary loss.

    Args:
        config: ``DeepSeekMoEConfig``.
    """

    config_class = DeepSeekMoEConfig

    def __init__(self, config: DeepSeekMoEConfig) -> None:
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            DeepSeekBlock(config, i) for i in range(config.n_layer)
        ])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        head_dim = config.n_embd // config.n_head
        freqs = precompute_freqs_cis(head_dim, config.block_size, config.rope_theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

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
        """Forward pass; accumulates load-balancing aux loss from MoE layers.

        Args:
            idx: Token indices ``(B, T)``.
            targets: Optional target indices ``(B, T)``.

        Returns:
            ``(logits, loss)`` where *loss* = CE + aux when training.

        Raises:
            AssertionError: If *T* > ``block_size``.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        x = self.drop(self.wte(idx))
        freqs_cis = self.freqs_cis[:T]

        all_router_logits: List[torch.Tensor] = []
        all_indices: List[torch.Tensor] = []

        for block in self.blocks:
            x, router_logits = block(x, freqs_cis)
            if router_logits is not None:
                # Extract top-k indices from router logits for the aux loss
                _, top_k_indices = router_logits.topk(self.config.top_k, dim=-1)
                all_router_logits.append(router_logits)
                all_indices.append(top_k_indices)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            # Accumulate load-balancing loss across MoE layers
            aux_loss = torch.tensor(0.0, device=x.device)
            for rl, idx_k in zip(all_router_logits, all_indices):
                aux_loss = aux_loss + load_balancing_loss(
                    rl, idx_k, self.config.n_routed_experts
                )
            if all_router_logits:
                aux_loss = aux_loss / len(all_router_logits)
            loss = ce_loss + self.config.router_aux_loss_coef * aux_loss
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
