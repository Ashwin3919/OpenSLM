"""Negative-path and edge-case tests for BatchLoader and checkpoint I/O."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.infra.io import BatchLoader, load_checkpoint, save_checkpoint


def _make_bin(path: Path, n_tokens: int) -> None:
    arr = np.memmap(str(path), dtype=np.uint16, mode="w+", shape=(n_tokens,))
    arr[:] = np.arange(n_tokens, dtype=np.uint16) % 100
    arr.flush()


@pytest.fixture
def tiny_loader(tmp_path: Path) -> BatchLoader:
    _make_bin(tmp_path / "train.bin", 512)
    _make_bin(tmp_path / "validation.bin", 512)
    return BatchLoader(
        train_path=str(tmp_path / "train.bin"),
        validation_path=str(tmp_path / "validation.bin"),
        block_size=16,
        batch_size=2,
        device="cpu",
        device_type="cpu",
    )


# ── BatchLoader construction ──────────────────────────────────────────────────


def test_batchloader_rejects_zero_block_size(tmp_path: Path) -> None:
    """block_size=0 must be rejected at construction time."""
    with pytest.raises(ValueError, match="block_size"):
        BatchLoader(
            train_path=str(tmp_path / "train.bin"),
            validation_path=str(tmp_path / "val.bin"),
            block_size=0,
            batch_size=2,
            device="cpu",
            device_type="cpu",
        )


def test_batchloader_rejects_zero_batch_size(tmp_path: Path) -> None:
    """batch_size=0 must be rejected at construction time."""
    with pytest.raises(ValueError, match="batch_size"):
        BatchLoader(
            train_path=str(tmp_path / "train.bin"),
            validation_path=str(tmp_path / "val.bin"),
            block_size=16,
            batch_size=0,
            device="cpu",
            device_type="cpu",
        )


# ── BatchLoader.get_batch ─────────────────────────────────────────────────────


def test_get_batch_invalid_split(tiny_loader: BatchLoader) -> None:
    """Unrecognised split names must raise ValueError."""
    with pytest.raises(ValueError, match="split"):
        tiny_loader.get_batch("test")


def test_get_batch_data_too_small(tmp_path: Path) -> None:
    """Data shorter than block_size must raise ValueError on get_batch."""
    _make_bin(tmp_path / "train.bin", 10)
    _make_bin(tmp_path / "validation.bin", 10)
    loader = BatchLoader(
        train_path=str(tmp_path / "train.bin"),
        validation_path=str(tmp_path / "validation.bin"),
        block_size=16,
        batch_size=2,
        device="cpu",
        device_type="cpu",
    )
    with pytest.raises(ValueError, match="block_size"):
        loader.get_batch("train")


def test_get_batch_returns_correct_shapes(tiny_loader: BatchLoader) -> None:
    """get_batch should return (batch_size, block_size) int64 tensors."""
    x, y = tiny_loader.get_batch("train")
    assert x.shape == (2, 16)
    assert y.shape == (2, 16)
    assert x.dtype == torch.int64
    assert y.dtype == torch.int64


def test_get_batch_y_is_x_shifted(tiny_loader: BatchLoader) -> None:
    """y must be the next-token target: y[i] == x[i+1] for all positions."""
    torch.manual_seed(0)
    x, y = tiny_loader.get_batch("train")
    # The last token of x and first token of y are from consecutive positions
    # Check that y is x shifted by one in the sequence dimension
    assert torch.equal(x[:, 1:], y[:, :-1])


# ── load_checkpoint ───────────────────────────────────────────────────────────


def test_load_checkpoint_missing_file(tmp_path: Path) -> None:
    """Loading a nonexistent file must raise FileNotFoundError."""
    import torch.nn as nn

    model = nn.Linear(4, 4)
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(str(tmp_path / "missing.pt"), model)


def test_load_checkpoint_missing_model_key(tmp_path: Path) -> None:
    """Checkpoint without 'model' key must raise KeyError."""
    import torch.nn as nn

    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"iteration": 0, "val_loss": 1.0}, str(bad_ckpt))

    model = nn.Linear(4, 4)
    with pytest.raises(KeyError, match="model"):
        load_checkpoint(str(bad_ckpt), model)


def test_save_and_load_checkpoint_roundtrip(tmp_path: Path, tiny_model_config) -> None:
    """Weights saved by save_checkpoint must be restored correctly by load_checkpoint."""
    from src.models.gpt.model import GPT

    model = GPT(tiny_model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    ckpt_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, iteration=7, val_loss=2.5, config_dict={})

    model2 = GPT(tiny_model_config)
    meta = load_checkpoint(ckpt_path, model2)

    assert meta["iteration"] == 7
    assert abs(meta["val_loss"] - 2.5) < 1e-6
    # Verify weights are actually equal
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
