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
        model: torch.nn.Module = None,
        batch_loader: Any = None,
        ctx: Any = None,
    ) -> None:
        super().__init__(config)
        self.model = model
        self.batch_loader = batch_loader
        self.ctx = ctx
        self._metrics: Dict[str, float] = {}

    def configure(self) -> None:
        """Load components if running standalone."""
        if self.model is None:
            from pathlib import Path
            from src.infra.device import get_device_context
            from src.core.registry import create_model
            from src.infra.io import BatchLoader, load_checkpoint

            self._device, self._device_type, _, _, self.ctx = get_device_context(self.config.device)
            
            data_cfg = self.config.data
            train_cfg = self.config.training
            self.batch_loader = BatchLoader(
                train_path=str(Path(data_cfg.output_dir) / data_cfg.train_file),
                validation_path=str(Path(data_cfg.output_dir) / data_cfg.validation_file),
                block_size=train_cfg.block_size,
                batch_size=train_cfg.batch_size,
                device=self._device,
                device_type=self._device_type,
            )

            self.model = create_model(self.config.model_type, self.config.model).to(self._device)
            
            ckpt_path = self.config.inference.checkpoint_path
            if not ckpt_path:
                ckpt_path = str(Path(self.config.training.checkpoint_path) / "best_model.pt")
                self.config.inference.checkpoint_path = ckpt_path
                
            load_checkpoint(ckpt_path, self.model, device=self._device)

    def validate(self) -> None:
        """Ensure model and batch loader are available."""
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
