"""DataPrepPipeline: download, tokenise, and write TinyStories to .bin files.

Reproduces the original notebook's data preparation exactly:
- Downloads via HuggingFace ``datasets``
- Tokenises with tiktoken (``encode_ordinary``, no special tokens)
- Writes contiguous uint16 token arrays to memory-mapped ``.bin`` files

Skips any file that already exists so repeated runs are idempotent.
"""

import logging
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm

from src.models.config import AppConfig
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class DataPrepPipeline(BasePipeline):
    """Downloads and tokenises a HuggingFace dataset into ``.bin`` memmap files.

    Output files are written to ``config.data.output_dir``.  Existing files
    are not overwritten.

    Args:
        config: ``AppConfig`` — only ``config.data`` fields are used.
    """

    def configure(self) -> None:
        """Initialise the tokeniser and resolve output paths from config."""
        self._enc = tiktoken.get_encoding(self.config.data.encoding)
        self._output_dir = Path(self.config.data.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._train_path = self._output_dir / self.config.data.train_file
        self._val_path = self._output_dir / self.config.data.validation_file

    def validate(self) -> None:
        """Log a notice when output files already exist."""
        if self._train_path.exists() and self._val_path.exists():
            logger.info(
                "Both data files already exist — skipping download and tokenisation.\n"
                f"  train: {self._train_path}\n"
                f"  validation: {self._val_path}"
            )

    def run(self) -> None:
        """Download the dataset, tokenise all splits, and write ``.bin`` files."""
        if self._train_path.exists() and self._val_path.exists():
            return

        logger.info(f"Loading dataset: {self.config.data.dataset}")
        ds = load_dataset(self.config.data.dataset)

        logger.info("Tokenising all splits …")
        tokenized = ds.map(
            self._tokenize,
            remove_columns=["text"],
            desc="Tokenising",
            num_proc=self.config.data.num_proc,
        )

        split_paths = {
            "train": str(self._train_path),
            "validation": str(self._val_path),
        }

        for split, path in split_paths.items():
            if split not in tokenized:
                logger.warning(f"Split '{split}' not found in dataset — skipping.")
                continue
            if Path(path).exists():
                logger.info(f"{path} already exists — skipping.")
                continue
            self._write_split(tokenized[split], path, split)

    def _tokenize(self, example: dict) -> dict:
        """Tokenise a single text example with ``encode_ordinary``.

        ``encode_ordinary`` ignores special tokens, matching the original
        notebook exactly.

        Args:
            example: A dict with a ``"text"`` key containing a story string.

        Returns:
            A dict with ``"ids"`` (list of int token IDs) and ``"len"`` (int).
        """
        ids = self._enc.encode_ordinary(example["text"])
        return {"ids": ids, "len": len(ids)}

    def _write_split(self, dset, path: str, split_name: str) -> None:
        """Write a tokenised HuggingFace dataset split to a uint16 memmap file.

        Uses sharded iteration to keep memory usage bounded regardless of
        dataset size.

        Args:
            dset: HuggingFace ``Dataset`` with ``"ids"`` and ``"len"`` columns.
            path: Destination ``.bin`` file path.
            split_name: Human-readable name used in progress bar and logging.
        """
        arr_len = int(np.sum(dset["len"], dtype=np.uint64))
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr_len,))
        total_shards = self.config.data.total_shards

        idx = 0
        for batch_idx in tqdm(range(total_shards), desc=f"Writing {split_name}.bin"):
            batch = (
                dset.shard(num_shards=total_shards, index=batch_idx, contiguous=True)
                .with_format("numpy")
            )
            arr_batch = np.concatenate(batch["ids"])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()
        logger.info(f"Wrote {arr_len:,} tokens → {path}")
