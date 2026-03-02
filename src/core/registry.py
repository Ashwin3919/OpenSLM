"""Model registry: maps model-type strings to SLM classes.

Usage::

    # Register a model (done once, usually in models/<name>/__init__.py):
    from src.core.registry import register_model

    @register_model("my_model")
    class MyModel(BaseSLM):
        ...

    # Instantiate by name (done in pipelines):
    from src.core.registry import create_model
    model = create_model(config.model_type, config.model)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from src.core.base import BaseSLM

MODEL_REGISTRY: dict[str, Type] = {}


def register_model(name: str):
    """Decorator that registers a ``BaseSLM`` subclass under *name*.

    Args:
        name: The string key used in ``model_type`` YAML fields.

    Returns:
        A decorator that registers the class and returns it unchanged.

    Example::

        @register_model("gpt")
        class GPT(BaseSLM):
            ...
    """

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def create_model(model_type: str, model_config) -> "BaseSLM":
    """Instantiate a registered model by its type name.

    If *model_type* is not yet in the registry, attempts a lazy import of the
    top-level ``models`` package to trigger auto-discovery.

    Args:
        model_type: String key matching a ``register_model`` call (e.g. "gpt").
        model_config: A config dataclass instance passed to the model's
            ``__init__``.

    Returns:
        An initialised ``BaseSLM`` instance.

    Raises:
        KeyError: If *model_type* is not registered after auto-discovery.
    """
    if model_type not in MODEL_REGISTRY:
        # Lazy auto-discovery: import the top-level models package so that
        # every models/<name>/__init__.py is executed and calls register_model.
        try:
            import importlib

            importlib.import_module("models")
        except ImportError:
            pass

    if model_type not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[model_type](model_config)
