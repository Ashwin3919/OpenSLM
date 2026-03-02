"""GPT language model: token + position embeddings → TransformerBlocks → LM head.

Weight tying between the token embedding and LM head follows the original
GPT-2 design and halves the parameter count for the output projection.

Generation logic lives in ``src.core.generation`` — this module contains
only the model architecture and weight initialisation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base import BaseSLM
from src.core.blocks import TransformerBlock
from src.core.layers import LayerNorm
from .config import GPTConfig


class GPT(BaseSLM):
    """GPT-style causal language model.

    Architecture:
        wte (token embedding) + wpe (position embedding) →
        dropout → N × TransformerBlock → LayerNorm → lm_head (linear).

    The token embedding weight is tied to the LM head weight, so the
    vocabulary projection shares parameters with the input embedding.

    Args:
        config: ``GPTConfig`` defining the model dimensions.
    """

    config_class = GPTConfig

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.n_embd, config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding and LM head share the same parameter tensor
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise weights using GPT-2 conventions
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) as in GPT-2
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialise Linear and Embedding weights with N(0, 0.02).

        Args:
            module: Any sub-module encountered during ``self.apply``.
        """
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
        """Forward pass through the full model.

        When *targets* is provided the full sequence logits and cross-entropy
        loss are returned (training mode).  When *targets* is ``None``, only
        the last-token logits are returned for efficient autoregressive
        generation.

        Args:
            idx: Token index tensor of shape ``(B, T)``.
            targets: Optional target indices of shape ``(B, T)`` for loss
                computation.

        Returns:
            A 2-tuple ``(logits, loss)`` where:
                - In training mode: ``logits`` shape ``(B, T, vocab_size)``
                  and ``loss`` is a scalar tensor.
                - In generation mode: ``logits`` shape ``(B, 1, vocab_size)``
                  and ``loss`` is ``None``.

        Raises:
            AssertionError: If sequence length *T* exceeds ``block_size``.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Sequence length {t} exceeds model block_size {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            return logits, loss
        else:
            # Return only the last position's logits to keep generation cheap
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Sum of ``numel()`` for all parameters with ``requires_grad=True``.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
