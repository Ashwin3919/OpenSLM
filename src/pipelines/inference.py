"""InferencePipeline: load a checkpoint and generate text from a prompt.

All generation parameters (checkpoint path, prompt, temperature, top_k,
max_new_tokens) come from ``config.inference`` — nothing is hardcoded.
"""

import logging
from pathlib import Path

import tiktoken
import torch

from src.core.generation import generate
from src.core.gpt import GPT
from src.infra.device import get_device_context
from src.infra.io import load_checkpoint
from src.models.config import AppConfig
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class InferencePipeline(BasePipeline):
    """Loads a trained checkpoint and generates text from a user prompt.

    Generated text is accessible via the ``output`` property after ``run()``.

    Args:
        config: ``AppConfig`` — ``config.inference``, ``config.model``,
            ``config.data``, and ``config.device`` are all used.
    """

    def configure(self) -> None:
        """Set up device context, tokeniser, and model skeleton."""
        self._device, _, _, _, _ = get_device_context(self.config.device)
        self._enc = tiktoken.get_encoding(self.config.data.encoding)
        self._model = GPT(self.config.model).to(self._device)
        self._output: str = ""

    def validate(self) -> None:
        """Verify that required inference fields are set and files exist.

        Raises:
            ValueError: If ``checkpoint_path`` or ``prompt`` is empty.
            FileNotFoundError: If the checkpoint file does not exist.
        """
        inf_cfg = self.config.inference
        if not inf_cfg.checkpoint_path:
            raise ValueError("inference.checkpoint_path must be set.")
        if not Path(inf_cfg.checkpoint_path).exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {inf_cfg.checkpoint_path}"
            )
        if not inf_cfg.prompt:
            raise ValueError("inference.prompt must be set.")

    def run(self) -> None:
        """Load weights, tokenise prompt, generate, and decode output."""
        inf_cfg = self.config.inference
        load_checkpoint(inf_cfg.checkpoint_path, self._model, device=self._device)

        token_ids = self._enc.encode_ordinary(inf_cfg.prompt)
        context = torch.tensor(
            token_ids, dtype=torch.long, device=self._device
        ).unsqueeze(0)

        output_ids = generate(
            self._model,
            context,
            max_new_tokens=inf_cfg.max_new_tokens,
            temperature=inf_cfg.temperature,
            top_k=inf_cfg.top_k,
        )
        self._output = self._enc.decode(output_ids.squeeze().tolist())
        logger.info(
            f"Generated {inf_cfg.max_new_tokens} tokens from prompt: "
            f"'{inf_cfg.prompt[:40]}…'"
        )

    @property
    def output(self) -> str:
        """Return the full generated text (prompt + continuation).

        Returns:
            Decoded string produced by the model.  Empty string until
            ``run()`` has been called.
        """
        return self._output
