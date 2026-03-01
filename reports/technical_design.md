# Technical Design: OpenSLM

**Author**: Ashwin Shirke

## Overview

This is a code repo to experiment with different SLM architectures and test possible architectures of the SLM. It is a clean-room migration
of a working Colab notebook into a system where:

- **Swapping architecture** = change one config line
- **New experiment** = new YAML file, zero code changes
- **New attention variant** = one new file in `src/core/`
- **Training, inference, data prep** = independent pipelines

## Layer Architecture

```
main.py (CLI dispatch)
    └── src/pipelines/           ← orchestration
        ├── TrainingPipeline
        ├── EvaluationPipeline
        ├── DataPrepPipeline
        └── InferencePipeline
            ├── src/core/        ← pure model logic (no IO)
            │   ├── GPT
            │   ├── TransformerBlock
            │   ├── CausalSelfAttention
            │   ├── LayerNorm / MLP
            │   └── generate()
            ├── src/infra/       ← filesystem, devices, logging
            │   ├── BatchLoader
            │   ├── save/load_checkpoint
            │   ├── get_device_context
            │   └── setup_logging
            ├── src/utils/       ← stateless helpers
            │   └── build_optimizer / build_scheduler / build_scaler
            └── src/models/      ← dataclasses only
                ├── AppConfig
                ├── GPTConfig
                └── TrainingConfig
```

## Key Design Decisions

### 1. `generate()` is decoupled from the GPT class

Generation strategy (top-k, temperature, nucleus, beam) is independent of
model architecture.  Keeping them separate allows experimenting with generation
without touching `gpt.py`.

### 2. Configs split by concern, composed per experiment

```yaml
# exp_001_baseline.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/gpt_small.yaml"
  - "../training/default.yaml"
```

An experiment override changes only the differing values:
```yaml
model:
  n_layer: 12   # everything else inherited
```

### 3. BatchLoader is a class, not a global function

Eliminates 4 implicit global dependencies (`block_size`, `batch_size`,
`device`, `device_type`).  Testable, injectable, allows multiple loaders.

### 4. Device/dtype resolution is centralised

`get_device_context(config.device)` returns `(device, device_type, dtype_str,
pt_dtype, autocast_ctx)`.  Called once per pipeline; no other module detects
devices.

### 5. Full checkpoints for resume support

Checkpoints include model + optimiser + scheduler + scaler + iteration number.
Resume training by setting `training.resume_from` in the experiment YAML.

### 6. Metrics written to disk; notebooks read from disk

`TrainingPipeline.save_results()` writes `outputs/metrics.json` after every
run.  Notebooks read from this file — they work mid-training and survive kernel
restarts.

## Notebook Responsibility Split

| Notebook | Purpose |
|---|---|
| `01_data_exploration` | Inspect TinyStories, token distributions, .bin sanity check |
| `02_architecture_lab` | Prototype new layers/attention from `src/core/` |
| `03_training_monitor` | Plot loss curves from `outputs/metrics.json` |
| `04_generation_demo` | Load checkpoint, generate text, compare outputs |
| `05_experiment_compare` | Compare metrics across experiments |

## Extension Scenarios

### New attention mechanism (e.g. RoPE)
1. Create `src/core/attention_rope.py`
2. Add to `MODEL_REGISTRY` in `src/core/__init__.py`
3. Set `model.attention_type: "rope"` in config
4. Done — no other files change

### New dataset
1. Create `configs/data/openwebtext.yaml`
2. Create `configs/experiments/exp_003_owt.yaml` including it
3. Run `make train CFG=configs/experiments/exp_003_owt.yaml`

### Resume training
```yaml
# in experiment YAML:
training:
  resume_from: "outputs/checkpoints/best_model.pt"
```
