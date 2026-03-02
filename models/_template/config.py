"""Hyperparameter dataclass for MyModel.

Copy this file to models/<your_name>/config.py and add the fields your
architecture needs.  Keep all fields typed and default-valued so the config
system can parse them from YAML without extra code.
"""

from dataclasses import dataclass


@dataclass
class MyModelConfig:
    vocab_size: int = 50257   # match your tokenizer
    block_size: int = 256     # context window in tokens

    # TODO: add your hyperparameters here, e.g.:
    # n_layer: int = 6
    # hidden_size: int = 512
