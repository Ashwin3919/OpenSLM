"""Dataclass schemas for all configuration domains.

Each dataclass maps 1-to-1 with a YAML config section.  No methods, no
business logic — pure data containers validated at load time by
``src.infra.config.validate_config``.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


@dataclass
class OptimizerConfig:
    """AdamW optimizer hyperparameters.

    Attributes:
        type: Optimizer name (only "adamw" supported).
        learning_rate: Peak learning rate reached after warmup.
        betas: AdamW beta coefficients (momentum, second-moment decay).
        weight_decay: L2 regularisation weight.
        eps: Numerical stability epsilon.
    """

    type: str = "adamw"
    learning_rate: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-9


@dataclass
class SchedulerConfig:
    """LR scheduler parameters (linear warmup → cosine decay).

    Attributes:
        warmup_steps: Number of steps for linear LR warmup.
        min_lr: Minimum LR at the end of cosine decay (eta_min for CosineAnnealingLR).
    """

    warmup_steps: int = 1000
    min_lr: float = 5e-4


@dataclass
class TrainingConfig:
    """Training loop configuration.

    Attributes:
        max_iters: Total number of optimiser steps.
        batch_size: Number of sequences per micro-batch.
        block_size: Context window used by the data loader (must match GPTConfig.block_size).
        gradient_accumulation_steps: Accumulate this many micro-batches before stepping.
        max_grad_norm: Gradient clipping threshold.
        eval_interval: Run evaluation every this many iterations.
        eval_batches: Number of batches to average over during evaluation.
        checkpoint_path: Directory where checkpoint .pt files are saved.
        resume_from: Optional path to a checkpoint to resume training from.
        optimizer: Nested optimizer configuration.
        scheduler: Nested scheduler configuration.
    """

    max_iters: int = 20000
    batch_size: int = 32
    block_size: int = 128
    gradient_accumulation_steps: int = 32
    max_grad_norm: float = 0.5
    eval_interval: int = 500
    eval_batches: int = 500
    checkpoint_path: str = "outputs/checkpoints/"
    resume_from: Optional[str] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class DataConfig:
    """Dataset and tokenisation configuration.

    Attributes:
        dataset: HuggingFace dataset identifier.
        encoding: tiktoken encoding name (e.g. "gpt2").
        num_proc: Number of processes for parallel tokenisation.
        total_shards: Number of shards used when writing .bin files.
        output_dir: Directory where .bin files are written.
        train_file: Filename for the training split .bin file.
        validation_file: Filename for the validation split .bin file.
    """

    dataset: str = "roneneldan/TinyStories"
    encoding: str = "gpt2"
    num_proc: int = 8
    total_shards: int = 1024
    output_dir: str = "data/"
    train_file: str = "train.bin"
    validation_file: str = "validation.bin"


@dataclass
class InferenceConfig:
    """Text generation configuration.

    Attributes:
        checkpoint_path: Path to a .pt checkpoint file to load.
        prompt: Text prompt to condition generation on.
        max_new_tokens: Number of tokens to generate beyond the prompt.
        temperature: Sampling temperature (1.0 = unchanged, <1.0 = sharper).
        top_k: If set, restrict sampling to the top-k highest-probability tokens.
    """

    checkpoint_path: str = ""
    prompt: str = ""
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: Optional[int] = None


@dataclass
class ProjectConfig:
    """Top-level project metadata.

    Attributes:
        name: Human-readable project name used in logging and output paths.
        seed: Global random seed for reproducibility.
        output_dir: Root directory for all experiment outputs.
    """

    name: str = "slm"
    seed: int = 42
    output_dir: str = "outputs/"


@dataclass
class LoggingConfig:
    """Logging configuration.

    Attributes:
        level: Python logging level string (DEBUG, INFO, WARNING, ERROR).
        file: Optional path for a log file; stdout only when None.
    """

    level: str = "INFO"
    file: Optional[str] = None


@dataclass
class DeviceConfig:
    """Compute device and dtype selection.

    Attributes:
        type: Device string — "auto" detects CUDA → MPS → CPU.
        dtype: Floating-point dtype — "auto" selects bfloat16 on supported CUDA,
            float16 on other CUDA, float32 on CPU/MPS.
    """

    type: str = "auto"
    dtype: str = "auto"


@dataclass
class AppConfig:
    """Root configuration composed from all sub-configs.

    Loaded by ``src.infra.config.load_config`` from a YAML experiment file that
    uses ``_includes_`` to merge base / model / data / training configs.

    ``model_type`` selects which registered SLM architecture to use.
    ``model`` holds the corresponding architecture config dataclass (parsed
    from the ``model:`` YAML section using the registry's ``config_class``).
    """

    project: ProjectConfig = field(default_factory=ProjectConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    model_type: str = "gpt"
    model: Any = None
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
