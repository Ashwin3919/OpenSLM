"""Register the BitNet 1.58b SLM under the key "bitnet"."""

from src.core.registry import register_model
from .model import BitNetSLM

register_model("bitnet")(BitNetSLM)
