"""Rotary Position Embeddings (RoPE).

Used by LLaMA, DeepSeek-MoE (attention layers), and Jamba (attention layers).

RoPE encodes absolute position information by rotating query and key vectors
in 2-D subspaces using complex-number multiplication. Unlike learned positional
embeddings, RoPE generalises to sequence lengths longer than seen during training.

Reference: Su et al., 2021 — "RoFormer: Enhanced Transformer with Rotary Position Embedding".
"""

import torch


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """Precompute complex-valued rotation frequencies for RoPE.

    The returned tensor can be sliced along the sequence dimension and passed
    to :func:`apply_rotary_emb` at forward time.

    Args:
        dim: Head dimension (must be even). Each pair of dimensions shares a
            rotation frequency.
        max_seq_len: Maximum sequence length to precompute for.
        theta: Base for the geometric sequence of frequencies (default 10 000).

    Returns:
        Complex64 tensor of shape ``(max_seq_len, dim // 2)`` where each entry
        ``e^{i * t * theta_k}`` is the rotation for position *t* and
        frequency index *k*.
    """
    # freqs[k] = 1 / theta^(2k / dim)  for k in 0 .. dim/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    # Convert to complex representation e^{i * freq}
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors.

    The input tensors are interpreted as sequences of 2-D vectors, rotated by
    the precomputed complex frequencies.

    Args:
        xq: Query tensor of shape ``(B, T, n_head, head_dim)``.
        xk: Key tensor of shape ``(B, T, n_kv_head, head_dim)``.
        freqs_cis: Precomputed frequencies of shape ``(max_seq_len, head_dim // 2)``.
            Will be sliced to the actual sequence length *T*.

    Returns:
        A 2-tuple ``(xq_rotated, xk_rotated)`` with the same shapes as the inputs,
        converted back to the original dtype.
    """
    # View as complex: (..., head_dim) → (..., head_dim // 2) complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Slice to actual sequence length and broadcast over batch + head dims
    t = xq.shape[1]  # sequence length
    freqs = freqs_cis[:t]  # (T, head_dim // 2)
    # freqs must broadcast: (1, T, 1, head_dim//2)
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    xq_out = torch.view_as_real(xq_ * freqs).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)
