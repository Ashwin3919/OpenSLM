"""Token generation logic, fully decoupled from the GPT model class.

Keeping generation separate from the model means generation strategies
(top-k, nucleus, temperature, beam search) can evolve independently
without touching architecture code.
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.core.gpt import GPT


@torch.no_grad()
def generate(
    model: "GPT",
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Generate tokens autoregressively from a conditioning sequence.

    Context is truncated to ``model.config.block_size`` at each step so the
    model never sees more tokens than it was trained on.

    Matches the original notebook behaviour exactly: temperature scaling,
    optional top-k filtering, then multinomial sampling.

    Args:
        model: A ``GPT`` model instance (can be on any device).
        idx: Conditioning token indices of shape ``(B, T)``.
        max_new_tokens: Number of new tokens to append.
        temperature: Sampling temperature.  1.0 leaves logits unchanged;
            values < 1.0 make the distribution sharper (more confident);
            values > 1.0 make it flatter (more random).
        top_k: If set, only the top-*k* logits are kept before sampling.
            All others are set to ``-inf``.

    Returns:
        Token indices of shape ``(B, T + max_new_tokens)``.
    """
    model.eval()
    for _ in range(max_new_tokens):
        # Crop to block_size if the context has grown too long
        idx_cond = (
            idx
            if idx.size(1) <= model.config.block_size
            else idx[:, -model.config.block_size :]
        )
        logits, _ = model(idx_cond)
        # logits shape: (B, 1, vocab_size) → squeeze to (B, vocab_size)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
