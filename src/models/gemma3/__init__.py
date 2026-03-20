"""Register the Gemma 3-style SLM under the key "gemma3"."""

from src.core.registry import register_model
from .model import Gemma3SLM

register_model("gemma3")(Gemma3SLM)
