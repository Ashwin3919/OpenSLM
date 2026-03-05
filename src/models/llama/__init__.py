"""Register the LLaMA-style SLM under the key "llama"."""

from src.core.registry import register_model
from .model import LlamaSLM

register_model("llama")(LlamaSLM)
