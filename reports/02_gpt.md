# GPT — Architecture Reference

GPT is a decoder-only transformer: a stack of identical blocks, each applying causal self-attention followed by a position-wise MLP. This document covers the implementation in `src/models/gpt/`, its configuration, and how to interpret training results.

---

## Architecture

```
Input token IDs  (B, T)
        │
   ┌────┴────────────────────────┐
   │  wte: token embedding        │  (B, T, n_embd)
   │  wpe: position embedding     │  (B, T, n_embd)
   │  dropout                     │
   └────────────┬────────────────┘
                │
        ┌───────┴──────┐  × n_layer
        │ TransformerBlock         │
        │   LayerNorm              │
        │   CausalSelfAttention    │  residual connection around each
        │   LayerNorm              │
        │   MLP (4× expansion)     │
        └───────────────┘
                │
        LayerNorm (final)
                │
        lm_head: Linear(n_embd → vocab_size)   ← weight-tied to wte
                │
        Logits  (B, T, vocab_size)   [training]
        Logits  (B, 1, vocab_size)   [generation — last position only]
```

### Weight tying

The token embedding matrix (`wte.weight`) and the LM head projection (`lm_head.weight`) share the same parameter tensor. This halves the parameter count for the output projection and is the original GPT-2 design.

### Pre-LayerNorm

LayerNorm is applied *before* attention and MLP (pre-LN), not after (post-LN). Pre-LN gives more stable gradients and is the convention used by GPT-2 / nanoGPT.

### Attention

`CausalSelfAttention` uses a fused QKV projection (one `Linear` layer, split into three) and a causal mask so position `t` only attends to positions ≤ `t`. When `torch.nn.functional.scaled_dot_product_attention` is available (PyTorch ≥ 2.0), Flash Attention is used automatically. Otherwise it falls back to an explicit masked softmax.

### Weight initialisation

- All `Linear` and `Embedding` weights: `N(0, 0.02)`
- Residual projections (`c_proj.weight`): scaled by `1 / sqrt(2 × n_layer)` — the GPT-2 convention to keep residual stream variance stable at initialisation.

---

## Parameters

### `GPTConfig` — `src/models/gpt/config.py`

| Field | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | `50257` | Vocabulary size. GPT-2 tokenizer has 50 257 tokens. |
| `block_size` | `int` | `128` | Maximum context window in tokens. Sequences longer than this are rejected. |
| `n_layer` | `int` | `6` | Number of transformer blocks stacked. |
| `n_head` | `int` | `6` | Number of attention heads per block. Must divide `n_embd` evenly. |
| `n_embd` | `int` | `384` | Embedding / hidden dimension. |
| `dropout` | `float` | `0.1` | Dropout probability. Applied after embeddings, inside attention, and after MLP. Set to `0.0` for inference or small runs. |
| `bias` | `bool` | `True` | Whether to add bias terms to `Linear` and `LayerNorm` layers. |

**Constraint**: `n_embd % n_head == 0`. Validation raises `ValueError` if violated.

### Parameter count formula (approximate)

```
embedding:   vocab_size × n_embd  (shared with lm_head)
per block:   12 × n_embd²  (attention QKV + proj + MLP two layers)
total:       vocab_size × n_embd + n_layer × 12 × n_embd²
```

---

## Preset Configs

Three ready-to-use model configs are provided in `configs/miniGPT_config/model/`.

### `gpt_tiny.yaml` — debugging / CI

```yaml
model_type: gpt
model:
  vocab_size: 50257
  block_size: 16
  n_layer: 2
  n_head: 2
  n_embd: 64
  dropout: 0.0
  bias: true
```

**~3.3M parameters.** Trains in seconds on CPU. Not useful for generation quality — use for checking that code runs end-to-end.

### `gpt_small.yaml` — baseline

```yaml
model_type: gpt
model:
  vocab_size: 50257
  block_size: 128
  n_layer: 6
  n_head: 6
  n_embd: 384
  dropout: 0.1
  bias: true
```

**~30 M parameters.** Matches the original notebook. Trains on a single consumer GPU (8 GB VRAM) in a few hours on TinyStories.

### `gpt_medium.yaml` — GPT-2 small scale

```yaml
model_type: gpt
model:
  vocab_size: 50257
  block_size: 256
  n_layer: 12
  n_head: 12
  n_embd: 768
  dropout: 0.1
  bias: true
```

**~123.8M parameters** (comparable to GPT-2 small). Requires ~20 GB GPU memory for training at the default batch size.

---

## Running GPT

### Minimal experiment file

```yaml
# configs/miniGPT_config/experiments/my_gpt_run.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/gpt_small.yaml"
  - "../training/default.yaml"
```

```bash
make prep     MODEL=miniGPT_config EXP=my_gpt_run
make train    MODEL=miniGPT_config EXP=my_gpt_run
make generate MODEL=miniGPT_config EXP=my_gpt_run
```

### Changing a hyperparameter

Override only the key that differs — everything else inherits from included files:

```yaml
# configs/miniGPT_config/experiments/my_big_gpt.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/gpt_small.yaml"
  - "../training/default.yaml"

model:
  n_layer: 12     # deeper model, everything else from gpt_small.yaml

training:
  max_iters: 40000
```

Then run:

```bash
make train MODEL=miniGPT_config EXP=my_big_gpt
```

---

## Training Config Reference

Defined in `configs/miniGPT_config/training/default.yaml`, deserialized into `TrainingConfig`.

| Field | Default | Description |
|---|---|---|
| `max_iters` | `20000` | Total optimiser steps. |
| `batch_size` | `32` | Sequences per micro-batch. |
| `block_size` | `128` | Context window fed to the data loader — must match `model.block_size`. |
| `gradient_accumulation_steps` | `32` | Micro-batches accumulated before each weight update. Effective batch = `batch_size × gradient_accumulation_steps`. |
| `max_grad_norm` | `0.5` | Gradient clipping threshold. |
| `eval_interval` | `500` | Evaluate on validation set every N iterations. |
| `eval_batches` | `500` | Number of validation batches averaged per evaluation. |
| `checkpoint_path` | `outputs/miniGPT/checkpoints/` | Directory for `.pt` checkpoint files. |
| `resume_from` | `null` | Path to a checkpoint to resume from. |
| `optimizer.learning_rate` | `1e-4` | Peak learning rate (after warmup). |
| `optimizer.betas` | `[0.9, 0.95]` | AdamW momentum coefficients. |
| `optimizer.weight_decay` | `0.1` | L2 regularisation. |
| `optimizer.eps` | `1e-9` | AdamW numerical stability epsilon. |
| `scheduler.warmup_steps` | `1000` | Linear LR warmup steps. |
| `scheduler.min_lr` | `5e-4` | Minimum LR at end of cosine decay. |

---

## Outputs and Results

### Checkpoints

Written to `training.checkpoint_path` (default `outputs/miniGPT/checkpoints/`). Each `.pt` file contains:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "scaler_state_dict": ...,
    "iter": <int>,
    "val_loss": <float>,
    "config": <AppConfig>,
}
```

### Metrics

`outputs/miniGPT/metrics.json` is updated after each evaluation interval. Fields include `iter`, `train_loss`, `val_loss`, `lr`. Read by `notebooks/03_training_monitor.ipynb`.

### Interpreting validation loss

MiniGPT (gpt_small) achieved a best validation loss of **2.3921** at 20k steps in the controlled 8-architecture comparison — the lowest of all models evaluated. For reference:

| Stage | Approximate val loss |
|---|---|
| Random initialisation | ~10.8 (log 50257) |
| Early training (1 K iters) | 3.0–4.0 |
| Mid training (10 K iters) | 1.8–2.2 |
| Converged (20 K iters) | 2.39 (measured) |

Perplexity = `exp(val_loss)`. A loss of 2.39 corresponds to perplexity ≈ 10.9.

### Generation

```bash
make generate MODEL=miniGPT_config

# or explicitly:
python main.py generate \
  --config configs/miniGPT_config/experiments/exp_001_baseline.yaml \
  --prompt "Once upon a time"
```

Key generation parameters (in `InferenceConfig`):

| Field | Default | Description |
|---|---|---|
| `checkpoint_path` | `""` | Path to `.pt` file to load weights from. |
| `prompt` | `""` | Text prefix to condition on. |
| `max_new_tokens` | `200` | Tokens to generate beyond the prompt. |
| `temperature` | `1.0` | Values < 1 sharpen the distribution; > 1 flatten it. |
| `top_k` | `null` | If set, restrict sampling to the top-k highest-probability tokens. |

---

## File Locations

| Purpose | File |
|---|---|
| Config dataclass | `src/models/gpt/config.py` |
| Model implementation | `src/models/gpt/model.py` |
| Plugin registration | `src/models/gpt/__init__.py` |
| Preset configs | `configs/miniGPT_config/model/gpt_tiny.yaml`, `gpt_small.yaml`, `gpt_medium.yaml` |
| Framework primitives used | `src/core/attention.py`, `src/core/blocks.py`, `src/core/layers.py` |
| Generation loop | `src/core/generation.py` |

---

## References

Radford et al., 2019 — "Language Models are Unsupervised Multitask Learners." OpenAI Blog.

Brown et al., 2020 — "Language Models are Few-Shot Learners." arXiv:2005.14165.

Karpathy, 2022 — "nanoGPT." GitHub: karpathy/nanoGPT.
