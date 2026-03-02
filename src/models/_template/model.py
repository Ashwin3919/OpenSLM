"""Stub model implementation — replace with your SLM architecture.

Steps:
    1. Rename ``MyModel`` to something descriptive.
    2. Build your layers in ``__init__``.
    3. Implement ``forward()``.  It must return ``(logits, loss)`` when
       *targets* is given and ``(logits, None)`` when *targets* is ``None``.
    4. To generate text, call ``src.core.generation.generate(model, idx, ...)``.
       Override this logic by providing your own generation method if needed.
"""

import torch
import torch.nn as nn

from src.core.base import BaseSLM
from .config import MyModelConfig


class MyModel(BaseSLM):
    config_class = MyModelConfig

    def __init__(self, config: MyModelConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: define your layers here, e.g.:
        # self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.layers = nn.ModuleList([...])
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            idx: Token indices of shape ``(B, T)``.
            targets: Optional target indices of shape ``(B, T)``.

        Returns:
            ``(logits, loss)`` if *targets* is given, ``(logits, None)``
            otherwise.
        """
        # TODO: implement your forward pass, e.g.:
        # x = self.embed(idx)
        # for layer in self.layers:
        #     x = layer(x)
        # logits = self.lm_head(x)
        # if targets is not None:
        #     import torch.nn.functional as F
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        #     return logits, loss
        # return logits[:, [-1], :], None
        raise NotImplementedError("Implement forward() in your model.")
