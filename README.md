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
make prep

# 2. Train the baseline GPT model
make train

# 3. Generate text from the saved checkpoint
make generate
```

Or with explicit config targets:

```bash
python main.py prep     --config configs/experiments/exp_001_baseline.yaml
python main.py tune     --config configs/experiments/exp_001_baseline.yaml
python main.py train    --config configs/experiments/exp_001_baseline.yaml
python main.py evaluate --config configs/experiments/exp_001_baseline.yaml
python main.py generate --config configs/experiments/exp_001_baseline.yaml \
                        --prompt "Once upon a time"
```

---

## Project Structure

```
main.py                   ← CLI entry point (prep / train / evaluate / generate)

src/
  models/                 ← SLM plugins and non-model config dataclasses
    gpt/
      config.py           ← GPTConfig dataclass
      model.py            ← GPT class
      __init__.py         ← registers "gpt" in the model registry
    _template/            ← copy-paste scaffold for a new model
    __init__.py           ← auto-discovery, exports config dataclasses (AppConfig, etc.)
  core/                   ← framework only (no model code)
    base.py               ← BaseSLM ABC
    registry.py           ← register_model decorator, create_model factory
    attention.py          ← CausalSelfAttention
    blocks.py             ← TransformerBlock
    layers.py             ← LayerNorm, MLP
    generation.py         ← autoregressive generate()
  pipelines/              ← orchestration (training, inference, data prep, evaluation)
  infra/                  ← I/O: config loading, checkpoints, device setup, logging
  utils/                  ← stateless helpers: optimizer, scheduler, scaler

configs/
  base.yaml               ← project / logging / device defaults
  model/                  ← gpt_tiny, gpt_small, gpt_medium
  data/                   ← tinystories
  training/               ← default, fast_debug
  experiments/            ← exp_001_baseline, exp_002_bigger_model

tests/                    ← pytest (28 tests, CPU-only, no network)
reports/                  ← design docs and model reference manuals
  technical_design.md     ← codebase guide: how to add models, datasets, experiments
  gpt.md                  ← GPT architecture reference and parameter guide
notebooks/                ← exploration only; all logic lives in src/
```

---

## Running Experiments

```bash
make train CFG=configs/experiments/exp_001_baseline.yaml
make train CFG=configs/experiments/exp_002_bigger_model.yaml
```

Create a new experiment by adding a YAML file — no code changes required. See `reports/technical_design.md` for the full workflow.

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
