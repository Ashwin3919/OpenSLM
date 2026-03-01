"""Device detection and autocast context management.

Single entry point: ``get_device_context(device_cfg)`` returns everything
downstream code needs to run on the correct device with the correct dtype.
No other module should contain device-detection logic.
"""

from contextlib import nullcontext
from typing import Any, Tuple

import torch

from src.models.config import DeviceConfig


def get_device_context(
    device_cfg: DeviceConfig,
) -> Tuple[str, str, str, torch.dtype, Any]:
    """Resolve device and dtype from config and return a ready-to-use context.

    Handles "auto" detection for both device and dtype.  Returns all
    device-related values as a single tuple so callers don't have to repeat
    the detection logic.

    Args:
        device_cfg: ``DeviceConfig`` with ``type`` and ``dtype`` fields.

    Returns:
        A 5-tuple of:
            - ``device``: Device string, e.g. ``"cuda"`` or ``"cpu"``.
            - ``device_type``: Base device type without index, e.g. ``"cuda"``.
            - ``dtype_str``: Resolved dtype name, e.g. ``"bfloat16"``.
            - ``pt_dtype``: The corresponding ``torch.dtype`` object.
            - ``ctx``: An autocast context manager (``nullcontext`` on CPU/MPS).

    Raises:
        ValueError: If ``device_cfg.dtype`` names an unsupported dtype.
    """
    # --- Resolve device ---
    if device_cfg.type == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_cfg.type

    # "cuda:0" → "cuda" for use in torch.amp.autocast
    device_type = device.split(":")[0]

    # --- Resolve dtype ---
    if device_cfg.dtype == "auto":
        if device_type == "cuda":
            dtype_str = (
                "bfloat16"
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else "float16"
            )
        else:
            dtype_str = "float32"
    else:
        dtype_str = device_cfg.dtype

    _dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_str not in _dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. "
            f"Choose from: {list(_dtype_map.keys())}"
        )
    pt_dtype = _dtype_map[dtype_str]

    # --- Autocast context ---
    if device_type == "cpu":
        ctx = nullcontext()
    else:
        ctx = torch.amp.autocast(device_type=device_type, dtype=pt_dtype)

    return device, device_type, dtype_str, pt_dtype, ctx
