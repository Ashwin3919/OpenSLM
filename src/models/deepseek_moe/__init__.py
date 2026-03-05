"""Register the DeepSeek MoE SLM under the key "deepseek_moe"."""

from src.core.registry import register_model
from .model import DeepSeekMoESLM

register_model("deepseek_moe")(DeepSeekMoESLM)
