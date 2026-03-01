# OpenSLM

A production-grade, modular GPT-style Small Language Model (~50–60M parameters)
trained from scratch on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

Migrated from a monolithic Colab notebook into a clean, experiment-driven system
where **swapping architecture = one config line, new experiment = new YAML file**.

---

## Quick Start

```bash
pip install -e ".[dev]"

# 1. Download and tokenise TinyStories (~10min first time)
make prep

# 2. Train with the baseline config
make train

# 3. Generate text from the best checkpoint
make generate
```

---

## Project Structure

```
src/
├── core/          # Pure model logic — LayerNorm, MLP, Attention, GPT, generate()
├── models/        # Dataclass schemas — GPTConfig, TrainingConfig, AppConfig
├── pipelines/     # Orchestration — DataPrep, Training, Evaluation, Inference
├── infra/         # Filesystem, devices, logging — BatchLoader, checkpoints
└── utils/         # Stateless helpers — build_optimizer, build_scheduler

configs/
├── base.yaml
├── model/         # gpt_small, gpt_medium, gpt_tiny
├── data/          # tinystories
├── training/      # default, fast_debug
└── experiments/   # exp_001_baseline, exp_002_bigger_model

notebooks/         # Exploration only — all logic lives in src/
tests/             # pytest — core, pipelines, infra
```

---

## Running Experiments

```bash
# Baseline (reproduces the original notebook)
make train CFG=configs/experiments/exp_001_baseline.yaml

# Bigger model
make train CFG=configs/experiments/exp_002_bigger_model.yaml

# Custom experiment: create a new YAML and run — zero code changes needed
```

Or directly:

```bash
python main.py prep     --config configs/experiments/exp_001_baseline.yaml
python main.py train    --config configs/experiments/exp_001_baseline.yaml
python main.py evaluate --config configs/experiments/exp_001_baseline.yaml
python main.py generate --config configs/experiments/exp_001_baseline.yaml \
                        --prompt "Once upon a time"
```

---

## Running Tests

```bash
make test        # all tests
make test-core   # core model tests only (fast, no GPU, no network)
make lint        # ruff
```

---

## Config System

Experiment YAML files use `_includes_` to compose base, model, data, and
training configs.  Override only the values that change:

```yaml
# configs/experiments/my_experiment.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/gpt_small.yaml"
  - "../training/default.yaml"

model:
  n_layer: 8   # only this changes; everything else inherits
```

---

## Resuming Training

```yaml
training:
  resume_from: "outputs/checkpoints/best_model.pt"
```

---

## Credits

Architecture based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
Original notebook by Vizuara AI Labs.
