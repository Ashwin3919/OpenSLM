# LLaMA — Architecture Reference

LlamaSLM is a decoder-only transformer that replaces every GPT-2 design choice with the LLaMA conventions: RMSNorm, Rotary Position Embeddings, Grouped Query Attention, and SwiGLU. This document covers the implementation in `src/models/llama/`, its configuration, and how to interpret training results.

---

## Architecture

```
Input token IDs  (B, T)
        │
   ┌────┴────────────────────────┐
   │  wte: token embedding        │  (B, T, n_embd)
   │  dropout                     │  (no position embedding — RoPE handles it)
   └────────────┬────────────────┘
                │
        ┌───────┴──────┐  × n_layer
        │  LlamaBlock              │
        │    RMSNorm               │
        │    GQAttention (RoPE)    │  residual connection around each
        │    RMSNorm               │
        │    SwiGLU FFN            │
        └───────────────┘
                │
        RMSNorm (final)
                │
        lm_head: Linear(n_embd → vocab_size)   ← weight-tied to wte
                │
        Logits  (B, T, vocab_size)   [training]
        Logits  (B, 1, vocab_size)   [generation — last position only]
```

### Key differences from GPT-2

| Feature | GPT-2 | LLaMA |
|---|---|---|
| Normalisation | `LayerNorm` (mean + variance) | `RMSNorm` (variance only, no mean centering) |
| Position encoding | Learned `wpe` table | Rotary Position Embeddings (RoPE), applied inside attention |
| Attention | Full Multi-Head Attention (n_head KV heads) | Grouped Query Attention (n_kv_head < n_head, shared KV) |
| FFN activation | GELU, 2 matrices | SwiGLU (`silu(xW1) * xW3`, 3 matrices, smaller hidden dim) |
| Bias terms | Yes | No — all Linear layers are `bias=False` |

### Weight tying

The token embedding matrix (`wte.weight`) and the LM head projection (`lm_head.weight`) share the same parameter tensor, identical to GPT-2. Because there is no position embedding table (`wpe`), the parameter saving relative to GPT-2 is larger at the same embedding size.

### RMSNorm

RMSNorm normalises each embedding vector by its root-mean-square, skipping the mean-centering step of LayerNorm. It has a single learnable scale vector `g` and no bias:

```
RMSNorm(x) = x / rms(x) * g,   rms(x) = sqrt(mean(x^2) + eps)
```

Applied before both the attention sub-layer and the FFN sub-layer (pre-norm). A final `RMSNorm` is applied to the output of the last block before the LM head.

### Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating the query and key vectors in pairs of dimensions. The rotation angle for dimension pair `(2i, 2i+1)` at position `t` is `t * theta^(-2i/head_dim)` where `theta = rope_theta` (default 10 000). The dot product `q_t · k_s` depends only on the relative offset `t - s`, giving the model translation-invariant position sensitivity without a stored table.

The frequencies are precomputed once at model construction and registered as a non-persistent buffer (`freqs_cis`), so they are automatically moved to the correct device without being saved in checkpoints.

### Grouped Query Attention (GQA)

Standard MHA uses `n_head` independent Q, K, V projections. GQA uses `n_kv_head` K/V projections shared across groups of `n_head / n_kv_head` query heads. This reduces the KV cache size at inference by a factor of `n_rep = n_head / n_kv_head` while retaining most of the expressive power of full MHA.

```
Q: (B, T, n_head   × head_dim)
K: (B, T, n_kv_head × head_dim)   ← n_kv_head < n_head
V: (B, T, n_kv_head × head_dim)

K, V are repeat_interleave'd to (B, T, n_head × head_dim) before attention.
```

Flash Attention (`F.scaled_dot_product_attention` with `is_causal=True`) is used when available (PyTorch >= 2.0).

### SwiGLU FFN

SwiGLU replaces the two-matrix GELU MLP with a three-matrix gated FFN:

```
SwiGLU(x) = W2( silu(W1(x)) * W3(x) )
```

Because the gate doubles the width, `intermediate_size` is set smaller than `4 * n_embd` (e.g. 1024 instead of 1536 for `n_embd=384`) so the total parameter count is comparable to GPT-2 at the same scale.

---

## Parameters

### `LlamaConfig` — `src/models/llama/config.py`

| Field | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | `50257` | Vocabulary size. GPT-2 tokenizer has 50 257 tokens. |
| `block_size` | `int` | `128` | Maximum context window in tokens. Sequences longer than this are rejected. |
| `n_layer` | `int` | `6` | Number of LlamaBlocks stacked. |
| `n_head` | `int` | `6` | Number of query attention heads per block. Must divide `n_embd`. |
| `n_kv_head` | `int` | `2` | Number of key/value heads (GQA). Must divide `n_head`. Use `n_head` for full MHA. |
| `n_embd` | `int` | `384` | Embedding / hidden dimension. |
| `intermediate_size` | `int` | `1024` | SwiGLU hidden dimension. Smaller than `4 × n_embd` to compensate for the third weight matrix. |
| `dropout` | `float` | `0.0` | Dropout probability. Applied after embeddings and inside attention. |
| `rope_theta` | `float` | `10000.0` | Base frequency for RoPE. Larger values extend effective context length. |

**Constraints**: `n_embd % n_head == 0` and `n_head % n_kv_head == 0`. Violations raise `AssertionError`.

### Parameter count formula (approximate)

```
embedding:     vocab_size × n_embd           (shared with lm_head — counted once)
per block:
  GQA Q proj:  n_embd × (n_head × head_dim)
  GQA K proj:  n_embd × (n_kv_head × head_dim)
  GQA V proj:  n_embd × (n_kv_head × head_dim)
  GQA out:     n_embd × n_embd
  SwiGLU W1:   n_embd × intermediate_size
  SwiGLU W2:   intermediate_size × n_embd
  SwiGLU W3:   n_embd × intermediate_size
  RMSNorm ×2:  2 × n_embd                    (scale vectors only)

total ≈ vocab_size × n_embd
       + n_layer × (n_embd × (n_head + 2*n_kv_head) × head_dim
                  + n_embd² + 2*n_embd*intermediate_size + n_embd)
```

For `llama_small` defaults (`n_embd=384, n_layer=6, n_head=6, n_kv_head=2, intermediate_size=1024`):

```
embedding:  50257 × 384 ≈ 19.3 M
per block:  384×(6+4)×64 + 384² + 3×384×1024 ≈ 1.57 M
            (GQA Q+K+V+out)   (SwiGLU gate+up+down)
total:      19.3 M + 6 × 1.57 M ≈ 29 M
```

---

## Preset Configs

Two ready-to-use model configs are in `configs/llama_config/model/`.

### `llama_small.yaml` — baseline (~29 M parameters)

```yaml
model_type: llama
model:
  vocab_size: 50257
  block_size: 128
  n_layer: 6
  n_head: 6
  n_kv_head: 2
  n_embd: 384
  intermediate_size: 1024
  dropout: 0.0
  rope_theta: 10000.0
```

Trains on a single consumer GPU (8 GB VRAM) in a few hours on TinyStories.

### `llama_medium.yaml` — larger variant (~60 M parameters)

```yaml
model_type: llama
model:
  vocab_size: 50257
  block_size: 256
  n_layer: 12
  n_head: 8
  n_kv_head: 2
  n_embd: 512
  intermediate_size: 1536
  dropout: 0.1
  rope_theta: 10000.0
```

Doubles depth and embedding width. Requires more GPU memory; reduce `batch_size` if needed.

---

## Running LLaMA

### Minimal experiment file

```yaml
# configs/llama_config/experiments/my_llama_run.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/llama_small.yaml"
  - "../training/default.yaml"
```

```bash
make prep     MODEL=llama_config EXP=my_llama_run
make train    MODEL=llama_config EXP=my_llama_run
make generate MODEL=llama_config EXP=my_llama_run
```

### Changing a hyperparameter

Override only the key that differs — everything else inherits from included files:

```yaml
# configs/llama_config/experiments/my_llama_deeper.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/llama_small.yaml"
  - "../training/default.yaml"

model:
  n_layer: 12     # deeper model, all other dims from llama_small.yaml

training:
  max_iters: 40000
```

Then run:

```bash
make train MODEL=llama_config EXP=my_llama_deeper
```

---

## Training Config Reference

Defined in `configs/llama_config/training/default.yaml`.

| Field | Default | Description |
|---|---|---|
| `max_iters` | `20000` | Total optimiser steps. |
| `batch_size` | `32` | Sequences per micro-batch. |
| `block_size` | `128` | Context window for the data loader — must match `model.block_size`. |
| `gradient_accumulation_steps` | `32` | Micro-batches before each weight update. Effective batch = `batch_size × grad_accum`. |
| `max_grad_norm` | `1.0` | Gradient clipping threshold. |
| `eval_interval` | `500` | Evaluate on validation set every N iterations. |
| `eval_batches` | `500` | Validation batches averaged per evaluation. |
| `checkpoint_path` | `outputs/llama/checkpoints/` | Directory for `.pt` checkpoint files. |
| `resume_from` | `null` | Path to a checkpoint to resume from. |
| `optimizer.learning_rate` | `3e-4` | Peak learning rate (after warmup). |
| `optimizer.betas` | `[0.9, 0.95]` | AdamW momentum coefficients. |
| `optimizer.weight_decay` | `0.1` | L2 regularisation. |
| `optimizer.eps` | `1e-8` | AdamW numerical stability epsilon. |
| `scheduler.warmup_steps` | `1000` | Linear LR warmup steps. |
| `scheduler.min_lr` | `3e-5` | Minimum LR at end of cosine decay. |

---

## Outputs and Results

### Checkpoints

Written to `training.checkpoint_path` (default `outputs/llama/checkpoints/`). Each `.pt` file contains:

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

`outputs/llama/metrics.json` is updated after each evaluation interval. Fields include `iter`, `train_loss`, `val_loss`, `lr`.

### Interpreting validation loss

LlamaSLM achieved a best validation loss of **2.5479** at 20k steps in the controlled 8-architecture comparison — fourth among all models, within 0.02 nats of Mamba and RetNet. Despite carrying every modern transformer upgrade (RoPE, GQA, SwiGLU, RMSNorm), LLaMA does not outperform the simpler GPT baseline at 128-token context. At this scale, the advantages of RoPE over learned position embeddings and GQA over full MHA are marginal.

### Generation

```bash
make generate MODEL=llama_config

# or explicitly:
python main.py generate \
  --config configs/llama_config/experiments/exp_001_baseline.yaml \
  --prompt "Once upon a time"
```

---

## File Locations

| Purpose | File |
|---|---|
| Config dataclass | `src/models/llama/config.py` |
| Model implementation | `src/models/llama/model.py` |
| Plugin registration | `src/models/llama/__init__.py` |
| Preset configs | `configs/llama_config/model/llama_small.yaml`, `llama_medium.yaml` |
| RMSNorm primitive | `src/core/normalization.py` |
| RoPE utilities | `src/core/rope.py` |
| SwiGLU primitive | `src/core/ffn.py` |
| Generation loop | `src/core/generation.py` |

---

## References

Touvron et al., 2023 — "LLaMA: Open and Efficient Foundation Language Models." arXiv:2302.13971.

Touvron et al., 2023 — "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288.

Su et al., 2021 — "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.

Ainslie et al., 2023 — "GQA: Training Generalised Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245.

Shazeer, 2020 — "GLU Variants Improve Transformer." arXiv:2002.05202.
