"""TrainingPipeline: full training loop with gradient accumulation and checkpointing.

Orchestrates every component produced in earlier phases:
- BatchLoader for data
- GPT model
- AdamW + SequentialLR scheduler + GradScaler
- EvaluationPipeline called at eval_interval
- Checkpoint saving on best validation loss
- Metrics written to JSON for notebook consumption

Matches the original notebook behaviour exactly for the same seed and config.
"""

import dataclasses
import json
import logging
from pathlib import Path
from typing import List

import torch
from tqdm.auto import tqdm

from src.core.registry import create_model
from src.infra.config import validate_config
from src.infra.device import get_device_context
from src.infra.io import BatchLoader, load_checkpoint, save_checkpoint
from src.infra.logging import setup_logging
from src.models.config import AppConfig
from src.pipelines.base import BasePipeline
from src.pipelines.evaluation import EvaluationPipeline
from src.utils.training import (
    build_optimizer,
    build_scheduler,
    build_scaler,
    count_params,
)

logger = logging.getLogger(__name__)


class TrainingPipeline(BasePipeline):
    """Runs the full training lifecycle for a GPT-style language model.

    Features:
        - Mixed-precision (float16/bfloat16) via ``torch.amp.autocast``
        - Gradient accumulation over ``gradient_accumulation_steps`` micro-batches
        - Linear LR warmup → cosine LR decay
        - Gradient clipping at ``max_grad_norm``
        - Evaluation on both splits every ``eval_interval`` iterations
        - Full checkpoint (model + optimiser + scheduler + scaler) on best val loss
        - Optional resume from a previous checkpoint
        - Loss history saved to ``outputs/metrics.json``

    Args:
        config: Fully-loaded and validated ``AppConfig``.
    """

    def configure(self) -> None:
        """Set up logging, seed, device context, model, optimiser, scheduler."""
        setup_logging(self.config.logging.level, self.config.logging.file)
        validate_config(self.config)
        torch.manual_seed(self.config.project.seed)

        self._device, self._device_type, self._dtype, _, self._ctx = (
            get_device_context(self.config.device)
        )
        logger.info(f"Device: {self._device} | dtype: {self._dtype}")

        # --- Data ---
        data_cfg = self.config.data
        train_cfg = self.config.training
        self._batch_loader = BatchLoader(
            train_path=str(Path(data_cfg.output_dir) / data_cfg.train_file),
            validation_path=str(
                Path(data_cfg.output_dir) / data_cfg.validation_file
            ),
            block_size=train_cfg.block_size,
            batch_size=train_cfg.batch_size,
            device=self._device,
            device_type=self._device_type,
        )

        # --- Model ---
        self._model = create_model(self.config.model_type, self.config.model).to(self._device)
        n_params = count_params(self._model)
        logger.info(f"Parameters: {n_params:,}  ({n_params / 1e6:.1f}M)")

        # --- Optimiser, scheduler, scaler ---
        self._optimizer = build_optimizer(self._model, train_cfg)
        self._scheduler = build_scheduler(self._optimizer, train_cfg)
        self._scaler = build_scaler(self._dtype)

        # --- Evaluation sub-pipeline (shared model reference) ---
        self._eval_pipeline = EvaluationPipeline(
            self.config, self._model, self._batch_loader, self._ctx
        )

        # --- Output directories ---
        out_dir = Path(self.config.project.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = out_dir / "metrics.json"
        self._ckpt_dir = Path(train_cfg.checkpoint_path)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # --- State tracking ---
        self._start_iter = 0
        self._best_val_loss = float("inf")
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []

        # --- Optional resume ---
        if train_cfg.resume_from:
            meta = load_checkpoint(
                train_cfg.resume_from,
                self._model,
                self._optimizer,
                self._scheduler,
                self._scaler,
                device=self._device,
            )
            self._start_iter = meta.get("iteration", 0)
            self._best_val_loss = meta.get("val_loss", float("inf"))
            logger.info(f"Resumed from iteration {self._start_iter}")

    def validate(self) -> None:
        """Confirm that data ``.bin`` files exist before starting the loop."""
        data_cfg = self.config.data
        for fname in (data_cfg.train_file, data_cfg.validation_file):
            p = Path(data_cfg.output_dir) / fname
            if not p.exists():
                raise FileNotFoundError(
                    f"Data file not found: {p}\n"
                    "Run 'make prep' (or DataPrepPipeline) first."
                )

    def run(self) -> None:
        """Execute the training loop from start_iter to max_iters."""
        train_cfg = self.config.training
        logger.info(
            f"Training: max_iters={train_cfg.max_iters}, "
            f"batch={train_cfg.batch_size}, "
            f"grad_accum={train_cfg.gradient_accumulation_steps}"
        )

        for iteration in tqdm(
            range(self._start_iter, train_cfg.max_iters),
            desc="Training",
            initial=self._start_iter,
            total=train_cfg.max_iters,
        ):
            # ── Evaluation ────────────────────────────────────────────────
            if iteration % train_cfg.eval_interval == 0 and iteration != 0:
                self._eval_pipeline.run()
                metrics = self._eval_pipeline.metrics
                train_loss = metrics["train"]
                val_loss = metrics["validation"]
                current_lr = self._optimizer.param_groups[0]["lr"]

                logger.info(
                    f"iter {iteration:>6d} | "
                    f"train={train_loss:.4f}  val={val_loss:.4f} | "
                    f"lr={current_lr:.6f}"
                )
                self._train_losses.append(train_loss)
                self._val_losses.append(val_loss)

                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._save_best_checkpoint(iteration, val_loss)

            # ── Forward + backward ────────────────────────────────────────
            X, y = self._batch_loader.get_batch("train")
            with self._ctx:
                _, loss = self._model(X, y)
                loss = loss / train_cfg.gradient_accumulation_steps
            self._scaler.scale(loss).backward()

            # ── Optimiser step (after accumulating enough micro-batches) ──
            step_due = (iteration + 1) % train_cfg.gradient_accumulation_steps == 0
            last_iter = iteration + 1 == train_cfg.max_iters
            if step_due or last_iter:
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), train_cfg.max_grad_norm
                )
                self._scaler.step(self._optimizer)
                self._scaler.update()
                self._optimizer.zero_grad(set_to_none=True)

            self._scheduler.step()

        logger.info(f"Training complete. Best val_loss: {self._best_val_loss:.4f}")

    def save_results(self) -> None:
        """Write training loss history to ``outputs/metrics.json``."""
        metrics = {
            "train_losses": self._train_losses,
            "val_losses": self._val_losses,
            "best_val_loss": self._best_val_loss,
            "eval_interval": self.config.training.eval_interval,
        }
        with open(self._metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info(f"Metrics saved → {self._metrics_path}")

    def _save_best_checkpoint(self, iteration: int, val_loss: float) -> None:
        """Save a checkpoint when a new best validation loss is achieved.

        Args:
            iteration: Current training iteration.
            val_loss: Validation loss that triggered the save.
        """
        path = str(self._ckpt_dir / "best_model.pt")
        save_checkpoint(
            path,
            self._model,
            self._optimizer,
            self._scheduler,
            self._scaler,
            iteration,
            val_loss,
            dataclasses.asdict(self.config),
        )
