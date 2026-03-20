# CLAUDE.md — OpenSLM Project Guide

## What This Project Is

OpenSLM is a benchmarking platform for 9 small language model architectures (~29–53M parameters) trained on TinyStories under a controlled experimental protocol. The goal is controlled comparison: same dataset, same tokenizer, same training budget, same optimizer — so architectural differences are the only variable.

**Author:** Ashwin Shirke
**Models:** GPT, LLaMA, Gemma 3, DeepSeek MoE, Mamba, RWKV, Jamba, BitNet, RetNet

---

## Common Commands

```bash
make prep     MODEL=miniGPT_config          # tokenise TinyStories (run once, data is shared)
make train    MODEL=llama_config            # train a model
make train    MODEL=llama_config  EXP=exp_002_bigger_model  # specific experiment
make evaluate MODEL=mamba_config
make generate MODEL=gemma3_config
make test                                   # full pytest suite
make test-core                              # fast, CPU, no network
make test-models                            # per-architecture forward/generate tests
make lint && make format

# Architecture Zoo one-liners
make train-llama
make train-gemma3
make train-mamba
make train-rwkv
make train-jamba
make train-bitnet
make train-retnet
make train-deepseek-moe
```

`MODEL` = folder name under `configs/`. `EXP` = experiment yaml name (default: `exp_001_baseline`).

---

## Adding a New Architecture — The Plugin Pattern

This is the most common task. Follow these 4 steps exactly:

### Step 1: `src/models/<name>/config.py`
```python
from dataclasses import dataclass

@dataclass
class MyConfig:
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    # ... model-specific fields
```

### Step 2: `src/models/<name>/model.py`
```python
from src.core.base import BaseSLM
from src.core.registry import register_model
from .config import MyConfig

class MySLM(BaseSLM):
    config_class = MyConfig

    def __init__(self, config: MyConfig) -> None:
        super().__init__()
        # ... build layers

    def forward(self, idx, targets=None):
        # Must return (logits, loss) when targets given, (logits, None) otherwise
        # Training: logits shape (B, T, vocab_size)
        # Generation: logits shape (B, 1, vocab_size)  [only last token]
        ...

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### Step 3: `src/models/<name>/__init__.py`
```python
from src.core.registry import register_model
from .model import MySLM

register_model("my_model")(MySLM)
```

### Step 4: Config files under `configs/<name>_config/`
Copy `configs/config_template/` and fill in. Required structure:
```
configs/<name>_config/
    base.yaml
    data/tinystories.yaml          # copy verbatim from any existing model
    model/<name>_small.yaml        # set model_type: <registry_key>
    training/default.yaml          # use standard values (see below)
    experiments/exp_001_baseline.yaml
```

The model is auto-discovered — no changes to `src/models/__init__.py` or any pipeline code.

---

## Shared Experimental Protocol

All models must use these training hyperparameters for fair comparison:

```yaml
learning_rate: 3.0e-4
min_lr: 3.0e-5           # 10:1 ratio with max LR — cosine decay floor
max_grad_norm: 1.0
betas: [0.9, 0.95]
weight_decay: 0.1
eps: 1.0e-8
warmup_steps: 1000
max_iters: 20000
batch_size: 32
gradient_accumulation_steps: 32    # effective batch = 1024 tokens
dropout: 0.0
```

**Architecturally-motivated exceptions (do not "fix" these):**

| Model | Exception | Reason |
|-------|-----------|--------|
| BitNet | `weight_decay: 0.0` | L2 decay pushes latent weights to zero → ternary quantizer rounds them to 0 permanently (dead neurons, irreversible) |
| BitNet | `warmup_steps: 2000` | STE gradients are noisy until latent weights reach meaningful magnitudes; 2× warmup prevents early instability |
| RWKV | `betas: [0.9, 0.99]` | WKV time-decay parameters need smoother gradient estimates; β₂=0.95 destabilises the recurrence |
| RWKV | `weight_decay: 0.01` | Strong L2 collapses time-mix scalars to zero, destroying per-channel temporal dynamics |
| DeepSeek | `batch_size: 16, gradient_accumulation_steps: 64` | MoE activates all 8 experts before routing discards 6 — peak memory doubles; effective batch stays 1024 |

---

## Shared Core Primitives

Reuse these from `src/core/` — do not copy-paste into model files:

| Module | Location | Used by |
|--------|----------|---------|
| `RMSNorm` | `src/core/normalization.py` | LLaMA, BitNet, Mamba, RWKV, Jamba, RetNet, DeepSeek, Gemma3 |
| `precompute_freqs_cis`, `apply_rotary_emb` | `src/core/rope.py` | LLaMA, BitNet, DeepSeek, Gemma3 |
| `SwiGLU` | `src/core/ffn.py` | LLaMA, RetNet, Jamba, DeepSeek |
| `MambaBlock` | `src/core/mamba_block.py` | Mamba, Jamba |
| `generate()` | `src/core/generation.py` | All models (via BaseSLM.generate) |

The `generate()` loop works for any model satisfying the `(logits, loss_or_None)` contract. Do not reimplement it per model.

---

## Config System

Configs use `_includes_` for deep-merge composition:

```yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/llama_small.yaml"
  - "../training/default.yaml"

# Overrides go below — only set what changes
model:
  n_layer: 12
```

Later files in `_includes_` take precedence. Inline overrides take precedence over all includes. Config loading: `src/infra/config.py`.

---

## Project Structure (Critical Files)

```
main.py                         ← CLI: prep / train / evaluate / generate / tune
Makefile                        ← all user-facing commands

src/
  core/
    base.py                     ← BaseSLM ABC — all models inherit this
    registry.py                 ← register_model() + create_model() factory
    rope.py                     ← precompute_freqs_cis, apply_rotary_emb
    normalization.py            ← RMSNorm
    ffn.py                      ← SwiGLU
    mamba_block.py              ← MambaBlock
    generation.py               ← autoregressive generate() loop
  models/
    __init__.py                 ← auto-discovers all model subfolders on import
    gpt/ llama/ gemma3/ mamba/ rwkv/ jamba/ bitnet/ retnet/ deepseek_moe/
    _template/                  ← copy-paste scaffold for new models
  pipelines/
    training.py                 ← training loop (no model-specific code)
    data_prep.py                ← tokenises TinyStories → .bin files
  infra/
    config.py                   ← YAML loading + _includes_ merge
    io.py                       ← checkpoint save/load

configs/
  <model>_config/
    base.yaml                   ← project + device
    data/tinystories.yaml       ← dataset (same for all)
    model/<name>_small.yaml     ← architecture params
    training/default.yaml       ← hyperparameters
    experiments/exp_001_baseline.yaml

reports/
  01_architecture_zoo.md        ← master reference — update when adding models
  02_gpt.md ... 11_gemma3.md   ← per-architecture deep-dives
  10_compare.md                 ← training results across all models
```

---

## Architecture Registry (Current Models)

| Registry key | Params | Report |
|---|---|---|
| `gpt` | 30.0M | `reports/02_gpt.md` |
| `llama` | 28.7M | `reports/03_llama.md` |
| `bitnet` | 28.8M | `reports/04_bitnet.md` |
| `retnet` | 29.9M | `reports/05_retnet.md` |
| `deepseek_moe` | 34.7M total / ~27.6M active | `reports/06_deepseek_moe.md` |
| `jamba` | 34.8M | `reports/07_jamba.md` |
| `mamba` | 30.4M | `reports/08_mamba.md` |
| `rwkv` | 53.0M | `reports/09rwkv.md` |
| `gemma3` | 28.7M | `reports/11_gemma3.md` |

---

## Key Design Decisions (Don't Change Without Reason)

**No switch/elif on model type anywhere.** Pipelines call `create_model(config.model_type, config.model)` — adding a new model never touches pipeline code.

**Weight tying is standard.** `self.wte.weight = self.lm_head.weight` in every model. Don't break this.

**Device resolution happens once** in `src/infra/device.py`. Models and data move to device in pipelines. Don't put `.to(device)` calls inside model `__init__`.

**Non-persistent buffers for precomputed tensors.** RoPE frequencies, attention masks, etc. use `self.register_buffer("name", tensor, persistent=False)` — they move to device automatically but are not saved in checkpoints.

**forward() contract is strict:**
- Training (`targets` given): return `(logits_BTV, loss_scalar)`
- Generation (`targets=None`): return `(logits_B1V, None)` — only last token position

**RWKV has n_embd=512, not 384.** At 384-dim, RWKV lands at ~20M due to the large embedding table (50257×384=19.3M). 512-dim gives ~53M. This is intentional and documented — do not "fix" it to 384.

**DeepSeek parameter count: 34.7M total, ~27.6M active.** The model.py docstring and configs reflect this. The always-active portion is ~25.2M; top-2 routed experts add ~2.36M per forward pass.

---

## Reports — What to Update When

| Event | Files to update |
|-------|----------------|
| Adding a new model | `reports/01_architecture_zoo.md` (table + links), `README.md` (table, structure, docs), `Makefile` (target + help text), new `reports/NN_<model>.md` |
| Fixing a training config | `reports/01_architecture_zoo.md` (protocol section), `reports/10_compare.md` (experimental setup), per-model report training config table, `reports/02_gpt.md` if GPT |
| Training results available | `reports/10_compare.md` (results table + analysis) |

---

## Testing

```
tests/
  test_core/        ← RMSNorm, RoPE, SwiGLU, MambaBlock, WKV scan
  test_models/      ← per-architecture: forward pass, generation, parameter count
  test_infra/       ← config loading (_includes_ merge), checkpoint save/load
  test_pipelines/   ← training loop smoke tests
```

When adding a new model, add `tests/test_models/test_<name>.py` covering:
1. Forward pass with `targets` — check logit shape and finite loss
2. Forward pass without `targets` — check `(B, 1, vocab_size)` shape and `loss is None`
3. Parameter count in expected range
4. `model.generate()` produces correct output length

---
