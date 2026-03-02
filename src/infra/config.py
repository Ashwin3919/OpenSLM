"""YAML config loading, ``_includes_`` merging, and validation.

Usage::

    config = load_config("configs/experiments/exp_001_baseline.yaml")
    validate_config(config)

The ``_includes_`` key in a YAML file lists paths (relative to that file) to
merge in order before applying the file's own keys.  Later keys override
earlier ones; nested dicts are merged recursively rather than replaced.
"""

import dataclasses
from pathlib import Path
from typing import Any, Optional, get_args, get_origin, get_type_hints
from typing import Union

import yaml

from src.models.config import (
    AppConfig,
    DataConfig,
    DeviceConfig,
    InferenceConfig,
    LoggingConfig,
    OptimizerConfig,
    ProjectConfig,
    SchedulerConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_raw(path: str) -> dict:
    """Read a single YAML file and return its contents as a plain dict.

    Args:
        path: Absolute or relative path to a YAML file.

    Returns:
        Parsed dict, or an empty dict if the file is empty.
    """
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict.

    Scalar values in *override* replace those in *base*.  Nested dicts are
    merged recursively so that sibling keys are preserved.

    Args:
        base: Starting dictionary.
        override: Dictionary whose values take precedence.

    Returns:
        A new dict containing the merged result.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _resolve_includes(path: str) -> dict:
    """Load a YAML file and recursively resolve any ``_includes_`` directives.

    Args:
        path: Path to the YAML file to load.

    Returns:
        Merged dict with all included files applied in order.
    """
    raw = _load_raw(path)
    includes: list = raw.pop("_includes_", [])
    base_dir = Path(path).parent

    merged: dict = {}
    for include in includes:
        include_path = str(base_dir / include)
        included = _resolve_includes(include_path)
        merged = _deep_merge(merged, included)

    return _deep_merge(merged, raw)


def _unwrap_optional(tp: Any) -> Any:
    """Return the inner type of ``Optional[X]``, or *tp* unchanged.

    Args:
        tp: A type annotation, possibly ``Optional[X]`` (i.e. ``Union[X, None]``).

    Returns:
        The unwrapped inner type if *tp* is Optional, otherwise *tp* itself.
    """
    if get_origin(tp) is Union:
        non_none = [a for a in get_args(tp) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return tp


def _to_dataclass(cls: type, data: Any) -> Any:
    """Recursively convert a plain dict to a dataclass instance.

    Missing keys use the field's default or default_factory.  Extra keys in
    *data* that have no matching field are silently ignored.

    Args:
        cls: The target dataclass type.
        data: A dict whose keys match field names of *cls*.

    Returns:
        An instance of *cls* populated from *data*.
    """
    if not dataclasses.is_dataclass(cls) or not isinstance(data, dict):
        return data

    hints = get_type_hints(cls)
    kwargs: dict = {}

    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue  # use field default / default_factory
        val = data[f.name]
        ft = _unwrap_optional(hints[f.name])
        if dataclasses.is_dataclass(ft) and isinstance(val, dict):
            val = _to_dataclass(ft, val)
        kwargs[f.name] = val

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str) -> AppConfig:
    """Load and merge a YAML experiment config into an ``AppConfig`` instance.

    Resolves ``_includes_`` directives recursively, deep-merges all included
    files, then converts the resulting dict to typed dataclasses.

    The ``model:`` section is parsed using the config_class registered for the
    ``model_type`` key (default ``"gpt"``), so new architectures only need to
    register their config dataclass — no changes here.

    Args:
        path: Path to the top-level experiment YAML file.

    Returns:
        Fully populated ``AppConfig``.

    Raises:
        FileNotFoundError: If *path* or any included file does not exist.
        yaml.YAMLError: If any YAML file is malformed.
        KeyError: If ``model_type`` names an unregistered architecture.
    """
    # Trigger auto-discovery so all src/models/<name>/__init__.py files run and
    # their register_model() calls are executed before we look up config_class.
    try:
        import importlib

        importlib.import_module("models")
    except ImportError:
        pass

    raw = _resolve_includes(path)

    # Parse the model section using the registered config_class so that
    # _to_dataclass (which sees `model: Any`) gets a fully typed instance.
    from src.core.registry import MODEL_REGISTRY

    model_type = raw.get("model_type", "gpt")
    model_cls = MODEL_REGISTRY.get(model_type)
    if model_cls is None:
        raise KeyError(
            f"model_type '{model_type}' is not registered. "
            f"Available: {list(MODEL_REGISTRY)}"
        )
    raw["model"] = model_cls.config_class(**raw.get("model", {}))

    return _to_dataclass(AppConfig, raw)


def validate_config(config: AppConfig) -> None:
    """Validate an ``AppConfig`` and raise on the first error found.

    Checks architectural constraints and required field presence so that
    problems surface immediately at startup rather than mid-training.

    Args:
        config: The config to validate.

    Raises:
        ValueError: If any field contains an invalid value.
    """
    mc = config.model
    if hasattr(mc, "n_embd") and hasattr(mc, "n_head") and mc.n_embd % mc.n_head != 0:
        raise ValueError(
            f"model.n_embd ({mc.n_embd}) must be divisible by "
            f"model.n_head ({mc.n_head})"
        )
    if hasattr(mc, "dropout") and not (0.0 <= mc.dropout <= 1.0):
        raise ValueError(f"model.dropout must be in [0, 1], got {mc.dropout}")

    tc = config.training
    if tc.max_iters <= 0:
        raise ValueError(f"training.max_iters must be > 0, got {tc.max_iters}")
    if tc.batch_size <= 0:
        raise ValueError(f"training.batch_size must be > 0, got {tc.batch_size}")
    if tc.gradient_accumulation_steps <= 0:
        raise ValueError(
            f"training.gradient_accumulation_steps must be > 0, "
            f"got {tc.gradient_accumulation_steps}"
        )
    if tc.scheduler.warmup_steps >= tc.max_iters:
        raise ValueError(
            f"training.scheduler.warmup_steps ({tc.scheduler.warmup_steps}) "
            f"must be < max_iters ({tc.max_iters})"
        )
