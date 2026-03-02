"""Filesystem operations: batch loading, memmap writing, checkpoint save/load.

This is the only module that touches the filesystem for binary data.
All path construction and file-existence checks live here.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BatchLoader:
    """Loads random mini-batches from memory-mapped binary token files.

    The memmap is re-created on every call to ``get_batch`` to avoid a
    known memory-leak in NumPy when the same memmap object is reused across
    many iterations.  See:
    https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122

    Args:
        train_path: Path to the ``train.bin`` memmap file.
        validation_path: Path to the ``validation.bin`` memmap file.
        block_size: Context window length (tokens per sequence).
        batch_size: Number of sequences sampled per batch.
        device: Target device string (e.g. ``"cuda"`` or ``"cpu"``).
        device_type: Base device type without index (e.g. ``"cuda"``).
            Used to decide whether to pin memory for async CUDA transfers.
    """

    def __init__(
        self,
        train_path: str,
        validation_path: str,
        block_size: int,
        batch_size: int,
        device: str,
        device_type: str,
    ) -> None:
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        self._paths: Dict[str, str] = {
            "train": train_path,
            "validation": validation_path,
        }
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.device_type = device_type

    def get_batch(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch from the specified split.

        Args:
            split: Either ``"train"`` or ``"validation"``.

        Returns:
            A 2-tuple ``(x, y)`` of input and target tensors, each of shape
            ``(batch_size, block_size)`` with dtype ``torch.int64``.

        Raises:
            ValueError: If *split* is not ``"train"`` or ``"validation"``.
        """
        if split not in self._paths:
            raise ValueError(
                f"split must be 'train' or 'validation', got '{split}'"
            )

        # Re-create memmap each call to prevent memory leak
        data = np.memmap(self._paths[split], dtype=np.uint16, mode="r")
        if len(data) <= self.block_size:
            raise ValueError(
                f"Data file '{self._paths[split]}' has {len(data)} tokens, "
                f"but block_size is {self.block_size}. Need at least block_size + 1 tokens."
            )

        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack(
            [
                torch.from_numpy(data[i : i + self.block_size].astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    data[i + 1 : i + 1 + self.block_size].astype(np.int64)
                )
                for i in ix
            ]
        )

        if self.device_type == "cuda":
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)

        return x, y


def write_memmap(
    path: str,
    tokens: np.ndarray,
    dtype: np.dtype = np.uint16,
) -> None:
    """Write a 1-D token array to a memory-mapped binary file.

    Args:
        path: Destination file path (created or overwritten).
        tokens: 1-D NumPy array of token IDs to write.
        dtype: Storage dtype.  Defaults to ``uint16``, which covers the full
            GPT-2 vocabulary (max token value 50256 < 2¹⁶).
    """
    arr = np.memmap(path, dtype=dtype, mode="w+", shape=(len(tokens),))
    arr[:] = tokens
    arr.flush()
    logger.info(f"Wrote {len(tokens):,} tokens → {path}")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler: torch.cuda.amp.GradScaler,
    iteration: int,
    val_loss: float,
    config_dict: dict,
) -> None:
    """Save a full training state checkpoint to disk.

    Saves model weights, optimiser state, scheduler state, scaler state,
    current iteration, and validation loss so that training can be resumed
    exactly from this point.

    Args:
        path: Destination ``.pt`` file path.  Parent directory is created
            automatically.
        model: The GPT model.
        optimizer: AdamW optimiser.
        scheduler: LR scheduler.
        scaler: GradScaler for mixed-precision training.
        iteration: Training iteration at the time of saving.
        val_loss: Validation loss at this checkpoint.
        config_dict: Serialisable config dict stored for reproducibility.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "iteration": iteration,
            "val_loss": val_loss,
            "config": config_dict,
        },
        path,
    )
    logger.info(
        f"Checkpoint saved → {path}  (iter {iteration}, val_loss {val_loss:.4f})"
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[object] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: str = "cpu",
) -> Dict:
    """Load a checkpoint and restore state into the provided objects.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: Model whose weights will be restored.
        optimizer: Optional optimiser to restore state into.
        scheduler: Optional scheduler to restore state into.
        scaler: Optional ``GradScaler`` to restore state into.
        device: Device to map tensors to during loading.

    Returns:
        A dict with keys ``"iteration"``, ``"val_loss"``, and ``"config"``
        from the checkpoint (whichever are present).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint at '{path}' is missing required key 'model'")
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    logger.info(
        f"Checkpoint loaded ← {path}  (iter {ckpt.get('iteration', '?')})"
    )
    return {k: ckpt[k] for k in ("iteration", "val_loss", "config") if k in ckpt}
