"""EvaluationPipeline: estimate average loss over train and validation splits.

Implements the ``estimate_loss`` logic from the original notebook as a
reusable pipeline.  Can be used standalone (``execute()``) or called from
within ``TrainingPipeline`` at evaluation intervals (``run()`` only).
"""

import logging
from typing import Any, Dict

import torch

from src.models.config import AppConfig
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class EvaluationPipeline(BasePipeline):
    """Computes mean cross-entropy loss over a fixed number of batches per split.

    The model is set to ``eval()`` mode during evaluation and restored to
    ``train()`` mode afterwards, so this pipeline is safe to call from inside
    a training loop.

    Args:
        config: Full ``AppConfig`` — only ``config.training.eval_batches`` is used.
        model: The GPT model to evaluate (shared reference with the trainer).
        batch_loader: ``BatchLoader`` instance pointing at the data files.
        ctx: Autocast context manager (returned by ``get_device_context``).
    """

    def __init__(
        self,
        config: AppConfig,
        model: torch.nn.Module,
        batch_loader: Any,
        ctx: Any,
    ) -> None:
        super().__init__(config)
        self.model = model
        self.batch_loader = batch_loader
        self.ctx = ctx
        self._metrics: Dict[str, float] = {}

    # configure and validate are no-ops — all components are injected at init
    def configure(self) -> None:  # noqa: D102
        pass

    def validate(self) -> None:  # noqa: D102
        pass

    def run(self) -> None:
        """Estimate loss on both splits and store results in ``self.metrics``.

        Iterates ``eval_batches`` times per split, accumulates losses, and
        returns the mean.  All computation is done under ``torch.inference_mode``
        so no gradients are tracked.
        """
        self._metrics = self._estimate_loss()
        logger.info(
            f"Evaluation | "
            f"train_loss={self._metrics['train']:.4f}  "
            f"val_loss={self._metrics['validation']:.4f}"
        )

    def _estimate_loss(self) -> Dict[str, float]:
        """Run ``eval_batches`` forward passes per split and return mean losses.

        Returns:
            Dict mapping split name (``"train"``, ``"validation"``) to mean loss.
        """
        out: Dict[str, float] = {}
        eval_batches = self.config.training.eval_batches

        self.model.eval()
        with torch.inference_mode():
            for split in ("train", "validation"):
                losses = torch.zeros(eval_batches)
                for k in range(eval_batches):
                    X, Y = self.batch_loader.get_batch(split)
                    with self.ctx:
                        _, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = float(losses.mean())
        self.model.train()
        return out

    @property
    def metrics(self) -> Dict[str, float]:
        """Return the most recent loss estimates from the last ``run()`` call."""
        return self._metrics
