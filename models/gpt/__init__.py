"""Register the GPT model under the key "gpt"."""

from src.core.registry import register_model
from .model import GPT

register_model("gpt")(GPT)
