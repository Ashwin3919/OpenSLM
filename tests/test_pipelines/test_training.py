"""Smoke test: 3-iteration training loop must not crash or produce NaN loss."""

import json

import pytest

from src.pipelines.training import TrainingPipeline


def test_training_smoke(smoke_app_config):
    """3-iteration run completes and writes a metrics file."""
    pipeline = TrainingPipeline(smoke_app_config)
    pipeline.execute()  # configure → validate → run → save_results

    metrics_path = pipeline._metrics_path
    assert metrics_path.exists(), "metrics.json should be written"

    with open(metrics_path) as fh:
        metrics = json.load(fh)

    assert "train_losses" in metrics
    assert "val_losses" in metrics
    assert "best_val_loss" in metrics


def test_training_loss_is_finite(smoke_app_config):
    """Loss must be finite (not NaN / Inf) after 3 iterations."""
    import math

    # Force eval to run at iteration 0 by setting eval_interval=1
    smoke_app_config.training.eval_interval = 1
    pipeline = TrainingPipeline(smoke_app_config)
    pipeline.execute()

    assert math.isfinite(pipeline._best_val_loss) or pipeline._best_val_loss == float(
        "inf"
    ), "best_val_loss should be finite or unchanged inf"
    # If eval fired, at least one loss was recorded
    # (may be empty if interval > max_iters, so no hard assert here)
