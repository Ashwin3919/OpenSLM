"""Register the Jamba Hybrid SLM under the key "jamba"."""

from src.core.registry import register_model
from .model import JambaSLM

register_model("jamba")(JambaSLM)
