"""Tests for YAML config loading, _includes_ merging, and validation."""

from pathlib import Path

import pytest
import yaml

from src.infra.config import _deep_merge, load_config, validate_config
from src.models.gpt.config import GPTConfig
from src.models.config import AppConfig


# ── _deep_merge ──────────────────────────────────────────────────────────────

def test_deep_merge_non_overlapping():
    a = {"x": 1, "y": {"a": 2}}
    b = {"z": 3, "y": {"b": 4}}
    assert _deep_merge(a, b) == {"x": 1, "z": 3, "y": {"a": 2, "b": 4}}


def test_deep_merge_override_scalar():
    assert _deep_merge({"x": 1}, {"x": 99})["x"] == 99


def test_deep_merge_nested_override():
    a = {"model": {"n_layer": 6, "n_embd": 384}}
    b = {"model": {"n_layer": 12}}
    result = _deep_merge(a, b)
    assert result["model"]["n_layer"] == 12
    assert result["model"]["n_embd"] == 384  # sibling key preserved


# ── load_config end-to-end ───────────────────────────────────────────────────

def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data))


def test_load_experiment_config(tmp_path: Path):
    """compose base + model YAML via _includes_ and load into AppConfig."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, {
        "project": {"name": "test", "seed": 7, "output_dir": "out/"},
        "logging": {"level": "DEBUG", "file": None},
        "device": {"type": "cpu", "dtype": "float32"},
    })

    (tmp_path / "model").mkdir()
    _write_yaml(tmp_path / "model" / "tiny.yaml", {
        "model": {
            "vocab_size": 100, "block_size": 16,
            "n_layer": 2, "n_head": 2, "n_embd": 64,
            "dropout": 0.0, "bias": True,
        }
    })

    (tmp_path / "exp").mkdir()
    exp = tmp_path / "exp" / "test.yaml"
    _write_yaml(exp, {"_includes_": ["../base.yaml", "../model/tiny.yaml"]})

    config = load_config(str(exp))

    assert isinstance(config, AppConfig)
    assert config.project.name == "test"
    assert config.project.seed == 7
    assert config.model.n_embd == 64
    assert config.model.n_layer == 2


def test_load_config_override(tmp_path: Path):
    """Values in the experiment file override included values."""
    base = tmp_path / "base.yaml"
    _write_yaml(base, {"project": {"name": "base", "seed": 42, "output_dir": "out/"}})

    exp = tmp_path / "exp.yaml"
    _write_yaml(exp, {
        "_includes_": ["base.yaml"],
        "project": {"name": "overridden"},
    })

    config = load_config(str(exp))
    assert config.project.name == "overridden"
    assert config.project.seed == 42  # inherited


# ── validate_config ───────────────────────────────────────────────────────────

def test_validate_bad_n_head():
    """n_embd not divisible by n_head should raise ValueError."""
    config = AppConfig()
    config.model = GPTConfig(
        vocab_size=100, block_size=16, n_layer=2,
        n_head=3, n_embd=64,   # 64 % 3 != 0
        dropout=0.0, bias=True,
    )
    with pytest.raises(ValueError, match="n_embd"):
        validate_config(config)


def test_validate_bad_max_iters():
    config = AppConfig()
    config.training.max_iters = 0
    with pytest.raises(ValueError, match="max_iters"):
        validate_config(config)


def test_validate_warmup_too_large():
    config = AppConfig()
    config.training.max_iters = 100
    config.training.scheduler.warmup_steps = 200
    with pytest.raises(ValueError, match="warmup_steps"):
        validate_config(config)


def test_validate_bad_dropout():
    """dropout outside [0, 1] should raise ValueError."""
    config = AppConfig()
    config.model = GPTConfig(
        vocab_size=100, block_size=16, n_layer=2,
        n_head=2, n_embd=64, dropout=1.5, bias=True,
    )
    with pytest.raises(ValueError, match="dropout"):
        validate_config(config)


def test_validate_bad_batch_size():
    """batch_size <= 0 should raise ValueError."""
    config = AppConfig()
    config.training.batch_size = -1
    with pytest.raises(ValueError, match="batch_size"):
        validate_config(config)


def test_validate_bad_gradient_accumulation_steps():
    """gradient_accumulation_steps <= 0 should raise ValueError."""
    config = AppConfig()
    config.training.gradient_accumulation_steps = 0
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        validate_config(config)
