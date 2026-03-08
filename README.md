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

## Architecture Zoo

8 SLM architectures implemented at the same ~30–35 M parameter scale, trained on TinyStories with identical tokenizer and training budget. See [`reports/architecture_zoo.md`](reports/architecture_zoo.md) for the full comparison plan.

| Registry Key | Params (total) | Active | Key Innovation | Report |
|---|---|---|---|---|
| `gpt` | ~29 M | 29 M | Vanilla transformer baseline | [gpt.md](reports/gpt.md) |
| `llama` | ~31 M | 31 M | RMSNorm + RoPE + GQA + SwiGLU | [llama.md](reports/llama.md) |
| `deepseek_moe` | ~48 M | ~25 M | Shared + routed MoE experts | [deepseek_moe.md](reports/deepseek_moe.md) |
| `mamba` | ~32 M | 32 M | Selective SSM, no attention | [mamba.md](reports/mamba.md) |
| `rwkv` | ~33 M | 33 M | Linear attention RNN | [rwkv.md](reports/rwkv.md) |
| `jamba` | ~35 M | 35 M | Hybrid Mamba + Attention | [jamba.md](reports/jamba.md) |
| `bitnet` | ~30 M | 30 M | Ternary weights {-1, 0, +1} | [bitnet.md](reports/bitnet.md) |
| `retnet` | ~29 M | 29 M | Multi-scale decay, no softmax | [retnet.md](reports/retnet.md) |

### Training any architecture

```bash
# Data prep is shared (run once)
make prep MODEL=miniGPT_config

# Train any model
make train MODEL=llama_config
make train MODEL=mamba_config
make train MODEL=deepseek_moe_config

# Architecture Zoo convenience targets
make train-llama
make train-mamba
make train-rwkv
make train-jamba
make train-bitnet
make train-retnet
make train-deepseek-moe

# Generate text
make generate MODEL=llama_config
```

---

## Project Structure

```
main.py                      ← CLI entry point (prep / train / evaluate / generate)

src/
  models/                    ← SLM plugins and non-model config dataclasses
    gpt/                     ← GPT-2 baseline
    llama/                   ← LLaMA-style (RoPE + RMSNorm + GQA + SwiGLU)
    deepseek_moe/            ← DeepSeek MoE (shared + routed experts)
    mamba/                   ← Mamba SSM (no attention, O(n))
    rwkv/                    ← RWKV (linear attention RNN)
    jamba/                   ← Jamba (Mamba + Attention interleaved)
    bitnet/                  ← BitNet 1.58b (ternary weights)
    retnet/                  ← RetNet (multi-scale retention)
    _template/               ← copy-paste scaffold for a new model
    __init__.py              ← auto-discovery, exports config dataclasses
  core/                      ← framework only (no model code)
    base.py                  ← BaseSLM ABC
    registry.py              ← register_model decorator, create_model factory
    attention.py             ← CausalSelfAttention (GPT + Jamba)
    blocks.py                ← TransformerBlock (GPT)
    layers.py                ← LayerNorm, MLP (GPT)
    generation.py            ← autoregressive generate()
    normalization.py         ← RMSNorm (all new architectures)
    rope.py                  ← Rotary Position Embeddings
    ffn.py                   ← SwiGLU FFN
    mamba_block.py           ← Mamba SSM block (Mamba + Jamba)
  pipelines/                 ← orchestration (training, inference, data prep, evaluation)
  infra/                     ← I/O: config loading, checkpoints, device setup, logging
  utils/                     ← stateless helpers: optimizer, scheduler, scaler

configs/
  miniGPT_config/            ← GPT-2 baseline configs
  llama_config/              ← LLaMA configs
  deepseek_moe_config/       ← DeepSeek MoE configs
  mamba_config/              ← Mamba configs
  rwkv_config/               ← RWKV configs
  jamba_config/              ← Jamba configs
  bitnet_config/             ← BitNet configs
  retnet_config/             ← RetNet configs
  config_template/           ← scaffold for new models

scripts/
  miniGPT_scripts/           ← run_data_prep.sh, run_training.sh, run_inference.sh
  llama_scripts/             ← same for LLaMA
  deepseek_moe_scripts/
  mamba_scripts/
  rwkv_scripts/
  jamba_scripts/
  bitnet_scripts/
  retnet_scripts/

tests/
  test_core/                 ← framework tests + shared primitive tests
  test_models/               ← per-architecture forward/generate/parameter tests
  test_infra/                ← config loading, checkpoints
  test_pipelines/            ← training loop smoke tests

reports/
  architecture_zoo.md        ← master comparison: all 8 architectures, benchmarking plan
  gpt.md                     ← GPT architecture reference
  llama.md                   ← LLaMA architecture reference
  deepseek_moe.md
  mamba.md
  rwkv.md
  jamba.md
  bitnet.md
  retnet.md
  technical_design.md        ← codebase guide: how to add models, datasets, experiments
```

---

## Running Experiments

```bash
# Default experiment (exp_001_baseline)
make train    MODEL=miniGPT_config

# Pick a specific experiment
make train    MODEL=llama_config  EXP=exp_002_bigger_model

# Other tasks
make prep     MODEL=miniGPT_config   # (data is shared — run once)
make evaluate MODEL=llama_config
make generate MODEL=mamba_config
```

`MODEL` maps to a folder under `configs/`. `EXP` selects the experiment YAML inside that folder's `experiments/` directory (default: `exp_001_baseline`).

---

## Tests

```bash
make test          # full suite
make test-core     # core framework + shared primitives (fast, CPU, no network)
make test-models   # all 7 new architecture tests
make lint          # ruff
```

---

## Documentation

| File | Contents |
|---|---|
| `reports/architecture_zoo.md` | All 8 architectures: comparison table, benchmarking plan, eval prompts |
| `reports/technical_design.md` | Codebase architecture, adding models, changing datasets |
| `reports/gpt.md` | GPT baseline: architecture, parameters, preset configs |
| `reports/llama.md` | LLaMA: RMSNorm, RoPE, GQA, SwiGLU |
| `reports/deepseek_moe.md` | DeepSeek MoE: routing, load balancing, shared experts |
| `reports/mamba.md` | Mamba SSM: selective scan, no attention |
| `reports/rwkv.md` | RWKV: WKV recurrence, token shift |
| `reports/jamba.md` | Jamba: hybrid Mamba + Attention |
| `reports/bitnet.md` | BitNet 1.58b: ternary weights, STE |
| `reports/retnet.md` | RetNet: multi-scale retention, no softmax |

---

## Credits

Architecture based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
