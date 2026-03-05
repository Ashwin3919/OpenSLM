"""Shared pytest fixtures.

All fixtures use CPU + zero dropout so tests run fast without a GPU
and produce deterministic outputs.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.gpt.config import GPTConfig
from src.models.llama.config import LlamaConfig
from src.models.deepseek_moe.config import DeepSeekMoEConfig
from src.models.mamba.config import MambaConfig
from src.models.rwkv.config import RWKVConfig
from src.models.jamba.config import JambaConfig
from src.models.bitnet.config import BitNetConfig
from src.models.retnet.config import RetNetConfig
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
def tiny_llama_config() -> LlamaConfig:
    """Minimal LlamaConfig for fast CPU tests."""
    return LlamaConfig(
        vocab_size=100, block_size=16, n_layer=2, n_head=4, n_kv_head=2,
        n_embd=32, intermediate_size=64, dropout=0.0,
    )


@pytest.fixture
def tiny_moe_config() -> DeepSeekMoEConfig:
    """Minimal DeepSeekMoEConfig for fast CPU tests."""
    return DeepSeekMoEConfig(
        vocab_size=100, block_size=16, n_layer=4, n_head=4, n_kv_head=2,
        n_embd=32, intermediate_size=64, n_shared_experts=1,
        n_routed_experts=4, top_k=2, expert_hidden_dim=16,
        dense_layers=[0, 1], router_aux_loss_coef=0.01, dropout=0.0,
    )


@pytest.fixture
def tiny_mamba_config() -> MambaConfig:
    """Minimal MambaConfig for fast CPU tests."""
    return MambaConfig(
        vocab_size=100, block_size=16, n_layer=2, d_model=32,
        d_state=4, d_conv=4, expand=2, dropout=0.0,
    )


@pytest.fixture
def tiny_rwkv_config() -> RWKVConfig:
    """Minimal RWKVConfig for fast CPU tests."""
    return RWKVConfig(
        vocab_size=100, block_size=16, n_layer=2, n_embd=32,
        n_head=4, ffn_mult=2, dropout=0.0,
    )


@pytest.fixture
def tiny_jamba_config() -> JambaConfig:
    """Minimal JambaConfig for fast CPU tests (4 layers: 2 Mamba + 2 Attn)."""
    return JambaConfig(
        vocab_size=100, block_size=16, n_layer=4, n_embd=32, n_head=4,
        mamba_d_state=4, mamba_d_conv=4, mamba_expand=2,
        intermediate_size=64, dropout=0.0,
    )


@pytest.fixture
def tiny_bitnet_config() -> BitNetConfig:
    """Minimal BitNetConfig for fast CPU tests."""
    return BitNetConfig(
        vocab_size=100, block_size=16, n_layer=2, n_head=4, n_kv_head=2,
        n_embd=32, intermediate_size=64, dropout=0.0,
    )


@pytest.fixture
def tiny_retnet_config() -> RetNetConfig:
    """Minimal RetNetConfig for fast CPU tests."""
    return RetNetConfig(
        vocab_size=100, block_size=16, n_layer=2, n_head=4, n_embd=32,
        intermediate_size=64, gamma_min=0.85, gamma_max=0.999, dropout=0.0,
    )


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
