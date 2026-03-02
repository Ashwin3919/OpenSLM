"""Shared pytest fixtures.

All fixtures use CPU + zero dropout so tests run fast without a GPU
and produce deterministic outputs.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from models.gpt.config import GPTConfig
from src.models.config import (
    AppConfig,
    DataConfig,
    DeviceConfig,
    LoggingConfig,
    OptimizerConfig,
    ProjectConfig,
    SchedulerConfig,
    TrainingConfig,
)


@pytest.fixture
def tiny_model_config() -> GPTConfig:
    """Minimal GPTConfig for fast CPU tests: 2 layers, 2 heads, 64 dim."""
    return GPTConfig(
        vocab_size=100,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True,
    )


@pytest.fixture
def sample_batch(tiny_model_config: GPTConfig):
    """A single (x, y) batch of random token IDs on CPU."""
    B, T = 2, tiny_model_config.block_size
    x = torch.randint(0, tiny_model_config.vocab_size, (B, T))
    y = torch.randint(0, tiny_model_config.vocab_size, (B, T))
    return x, y


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory — cleaned up automatically after each test."""
    return tmp_path


@pytest.fixture
def smoke_app_config(tmp_path: Path) -> AppConfig:
    """Full AppConfig wired to tiny in-memory data files for smoke tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    n_tokens = 512
    for fname in ("train.bin", "validation.bin"):
        arr = np.memmap(
            str(data_dir / fname), dtype=np.uint16, mode="w+", shape=(n_tokens,)
        )
        rng = np.random.default_rng(seed=0)
        arr[:] = rng.integers(0, 100, size=n_tokens, dtype=np.uint16)
        arr.flush()

    return AppConfig(
        project=ProjectConfig(
            name="smoke", seed=42, output_dir=str(tmp_path / "outputs")
        ),
        logging=LoggingConfig(level="WARNING"),
        device=DeviceConfig(type="cpu", dtype="float32"),
        model=GPTConfig(
            vocab_size=100,
            block_size=16,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=True,
        ),
        data=DataConfig(
            output_dir=str(data_dir),
            train_file="train.bin",
            validation_file="validation.bin",
        ),
        training=TrainingConfig(
            max_iters=3,
            batch_size=2,
            block_size=16,
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            eval_interval=100,   # no eval triggered during 3 iters
            eval_batches=2,
            checkpoint_path=str(tmp_path / "checkpoints"),
            optimizer=OptimizerConfig(learning_rate=1e-3),
            scheduler=SchedulerConfig(warmup_steps=1, min_lr=1e-4),
        ),
    )
