"""Register the RWKV SLM under the key "rwkv"."""

from src.core.registry import register_model
from .model import RWKVSLM

register_model("rwkv")(RWKVSLM)
