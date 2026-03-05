# OpenSLM

Experiment platform for small language models. Swap architectures with one config line, define new experiments with a YAML file, add new SLM implementations without touching pipeline code.

**Author**: Ashwin Shirke

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Quick Start

```bash
# 1. Tokenise TinyStories (~10 min, one-time)
make prep     MODEL=miniGPT_config

# 2. Train the baseline GPT model (automatically uses GPU/MPS if available)
make train    MODEL=miniGPT_config

# 3. Generate text from the saved checkpoint
make generate MODEL=miniGPT_config
```

Or with explicit config targets:

```bash
python main.py prep     --config configs/miniGPT_config/experiments/exp_001_baseline.yaml
python main.py train    --config configs/miniGPT_config/experiments/exp_001_baseline.yaml
python main.py evaluate --config configs/miniGPT_config/experiments/exp_001_baseline.yaml
python main.py generate --config configs/miniGPT_config/experiments/exp_001_baseline.yaml \
                        --prompt "Once upon a time"
```

---

## Project Structure

```
main.py                      ← CLI entry point (prep / train / evaluate / generate)

src/
  models/                    ← SLM plugins and non-model config dataclasses
    gpt/
      config.py              ← GPTConfig dataclass
      model.py               ← GPT class
      __init__.py            ← registers "gpt" in the model registry
    _template/               ← copy-paste scaffold for a new model (src side)
    __init__.py              ← auto-discovery, exports config dataclasses (AppConfig, etc.)
  core/                      ← framework only (no model code)
    base.py                  ← BaseSLM ABC
    registry.py              ← register_model decorator, create_model factory
    attention.py             ← CausalSelfAttention
    blocks.py                ← TransformerBlock
    layers.py                ← LayerNorm, MLP
    generation.py            ← autoregressive generate()
  pipelines/                 ← orchestration (training, inference, data prep, evaluation)
  infra/                     ← I/O: config loading, checkpoints, device setup, logging
  utils/                     ← stateless helpers: optimizer, scheduler, scaler

configs/
  miniGPT_config/            ← config folder for miniGPT
    base.yaml                ← project / logging / device defaults
    model/                   ← gpt_tiny, gpt_small, gpt_medium
    data/                    ← tinystories
    training/                ← default, fast_debug
    experiments/             ← exp_001_baseline, exp_002_bigger_model
  config_template/           ← copy this when adding a new model; contains <<MODEL_NAME>> placeholders
    base.yaml
    model/ data/ training/ experiments/

scripts/
  miniGPT_scripts/           ← shell scripts for miniGPT
    run_data_prep.sh
    run_training.sh
    run_inference.sh
  template_scripts/          ← copy this when adding a new model; contains <<MODEL_CONFIG>> placeholders
    run_data_prep.sh
    run_training.sh
    run_inference.sh

notebooks/
  00_template/               ← copy this when adding a new model
  01_miniGPT/                ← data exploration, architecture lab, training monitor, generation demo

tests/                       ← pytest (CPU-only, no network)
reports/                     ← design docs and model reference manuals
  technical_design.md        ← codebase guide: how to add models, datasets, experiments
  gpt.md                     ← GPT architecture reference and parameter guide
```

---

## Running Experiments

```bash
# Default experiment (exp_001_baseline)
make train    MODEL=miniGPT_config

# Pick a specific experiment
make train    MODEL=miniGPT_config  EXP=exp_002_bigger_model

# Other tasks
make prep     MODEL=miniGPT_config
make evaluate MODEL=miniGPT_config
make generate MODEL=miniGPT_config
```

`MODEL` maps to a folder under `configs/`. `EXP` selects the experiment YAML inside that folder's `experiments/` directory (default: `exp_001_baseline`).

To add a new model: copy `configs/config_template/` → `configs/<your_model>_config/`, replace `<<MODEL_NAME>>` placeholders, then `make train MODEL=<your_model>_config`. See `reports/technical_design.md` for the full workflow.

---

## Tests

```bash
make test        # full suite
make test-core   # core only (fast, no GPU, no network)
make lint        # ruff
```

---

## Documentation

| File | Contents |
|---|---|
| `reports/technical_design.md` | Codebase architecture, adding models, changing datasets, running experiments |
| `reports/gpt.md` | GPT architecture, all parameters, preset configs, results guide |

---

## Credits

Architecture based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
