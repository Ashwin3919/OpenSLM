"""Public exports for the models package (schemas only, no behaviour)."""

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
from src.models.types import Batch, DType, DeviceType, Metrics

import importlib
import os

_models_dir = os.path.dirname(__file__)

for _name in sorted(os.listdir(_models_dir)):
    _path = os.path.join(_models_dir, _name)
    if _name.startswith("_") or not os.path.isdir(_path):
        continue
    importlib.import_module(f"src.models.{_name}")

__all__ = [
    "AppConfig",
    "DataConfig",
    "DeviceConfig",
    "DType",
    "DeviceType",
    "Batch",
    "InferenceConfig",
    "LoggingConfig",
    "Metrics",
    "OptimizerConfig",
    "ProjectConfig",
    "SchedulerConfig",
    "TrainingConfig",
]
