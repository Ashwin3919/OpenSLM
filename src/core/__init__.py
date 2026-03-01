"""Public exports for the core package (pure model logic, no IO)."""

from src.core.attention import CausalSelfAttention
from src.core.blocks import TransformerBlock
from src.core.generation import generate
from src.core.gpt import GPT
from src.core.layers import LayerNorm, MLP

# Model registry: maps config.model.architecture → model class.
# Add new architectures here; pipeline code never changes.
MODEL_REGISTRY: dict = {"gpt": GPT}

__all__ = [
    "CausalSelfAttention",
    "GPT",
    "LayerNorm",
    "MLP",
    "MODEL_REGISTRY",
    "TransformerBlock",
    "generate",
]
