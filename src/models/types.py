"""Type aliases and enums for the SLM system."""

from enum import Enum
from typing import Tuple

import torch


class DeviceType(str, Enum):
    """Supported compute device types."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"


class DType(str, Enum):
    """Supported floating-point dtypes."""

    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


# Type aliases
Batch = Tuple[torch.Tensor, torch.Tensor]  # (inputs, targets), both shape (B, T)
Metrics = dict  # str → float loss values keyed by split name
