"""Training utility factories: optimiser, scheduler, scaler, and param count.

These are stateless helper functions — no IO, no side effects.  Pipelines
call them during ``configure()`` to build training components from config.
"""

import logging

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.models.config import TrainingConfig

logger = logging.getLogger(__name__)


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Construct an AdamW optimiser from ``TrainingConfig.optimizer``.

    Args:
        model: The model whose parameters will be optimised.
        config: ``TrainingConfig`` containing the nested ``OptimizerConfig``.

    Returns:
        A configured ``torch.optim.AdamW`` instance.

    Raises:
        ValueError: If ``optimizer.type`` is not ``"adamw"``.
    """
    opt_cfg = config.optimizer
    if opt_cfg.type.lower() != "adamw":
        raise ValueError(
            f"Only 'adamw' is supported as optimizer.type, got '{opt_cfg.type}'"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.learning_rate,
        betas=tuple(opt_cfg.betas),
        weight_decay=opt_cfg.weight_decay,
        eps=opt_cfg.eps,
    )
    logger.debug(
        f"AdamW: lr={opt_cfg.learning_rate}, betas={tuple(opt_cfg.betas)}, "
        f"wd={opt_cfg.weight_decay}, eps={opt_cfg.eps}"
    )
    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> SequentialLR:
    """Construct a linear-warmup → cosine-decay LR scheduler.

    Matches the original notebook exactly:
    - ``LinearLR`` with default ``start_factor=1/3`` ramps LR from
      ``learning_rate / 3`` to ``learning_rate`` over ``warmup_steps``.
    - ``CosineAnnealingLR`` then decays to ``min_lr`` over the remaining steps.

    Args:
        optimizer: The optimiser whose LR will be scheduled.
        config: ``TrainingConfig`` containing ``max_iters`` and the nested
            ``SchedulerConfig``.

    Returns:
        A ``SequentialLR`` that switches from warmup to decay at
        ``warmup_steps``.
    """
    sched_cfg = config.scheduler
    warmup_steps = sched_cfg.warmup_steps
    decay_steps = config.max_iters - warmup_steps

    warmup = LinearLR(optimizer, total_iters=warmup_steps)
    decay = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=sched_cfg.min_lr)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
    )
    logger.debug(
        f"Scheduler: {warmup_steps} warmup steps, cosine decay to "
        f"min_lr={sched_cfg.min_lr} over {decay_steps} steps"
    )
    return scheduler


def build_scaler(dtype: str) -> torch.cuda.amp.GradScaler:
    """Construct a ``GradScaler`` for mixed-precision training.

    The scaler is enabled only when ``dtype == "float16"`` because
    ``bfloat16`` does not require loss scaling (it has the same dynamic
    range as ``float32``).

    Args:
        dtype: Resolved dtype string, e.g. ``"float16"``, ``"bfloat16"``,
            or ``"float32"``.

    Returns:
        A ``GradScaler`` (enabled for float16, no-op otherwise).
    """
    enabled = dtype == "float16"
    logger.debug(f"GradScaler: enabled={enabled} (dtype={dtype})")
    return torch.cuda.amp.GradScaler(enabled=enabled)


def count_params(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*.

    Args:
        model: Any ``nn.Module``.

    Returns:
        Sum of ``numel()`` for all parameters where ``requires_grad`` is
        ``True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
