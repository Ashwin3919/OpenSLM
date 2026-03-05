"""Register the Mamba SSM SLM under the key "mamba"."""

from src.core.registry import register_model
from .model import MambaSLM

register_model("mamba")(MambaSLM)
