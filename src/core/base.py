"""Abstract base class for all SLM model implementations.

Every model registered in the plugin registry must inherit from ``BaseSLM``
and implement ``forward()``.  The default ``generate()`` works out of the box
for any autoregressive model that returns ``(logits, loss)`` from ``forward()``.
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class BaseSLM(nn.Module, ABC):
    """Abstract base for all SLM architectures.

    Subclasses must:
    - Set ``config_class`` to their hyperparameter dataclass.
    - Implement ``forward(idx, targets)``.

    ``generate()`` is provided for free and delegates to
    ``src.core.generation.generate``.  Override it only for custom sampling.
    """

    config_class = None  # Set to the model's config dataclass by each subclass

    @abstractmethod
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """Compute logits and optionally the cross-entropy loss.

        Args:
            idx: Token index tensor of shape ``(B, T)``.
            targets: Optional target indices of shape ``(B, T)``.

        Returns:
            ``(logits, loss)`` when *targets* is given; ``(logits, None)``
            otherwise.
        """
        ...

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive token generation.

        Delegates to ``src.core.generation.generate``.  Override for custom
        generation strategies (e.g. beam search, speculative decoding).

        Args:
            idx: Conditioning token indices of shape ``(B, T)``.
            max_new_tokens: Number of tokens to append.
            temperature: Sampling temperature.
            top_k: If set, restrict sampling to the top-k tokens.

        Returns:
            Token indices of shape ``(B, T + max_new_tokens)``.
        """
        from src.core.generation import generate

        return generate(self, idx, max_new_tokens, temperature, top_k)
