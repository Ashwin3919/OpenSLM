"""Register the RetNet SLM under the key "retnet"."""

from src.core.registry import register_model
from .model import RetNetSLM

register_model("retnet")(RetNetSLM)
