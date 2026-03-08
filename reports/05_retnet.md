# RetNet — Architecture Reference

RetNetSLM replaces softmax attention with a retention mechanism — a linear, normalisation-free operation that applies per-head exponential decay masks over past positions. Retention is parallelisable during training and admits an exact O(1)-per-step recurrent formulation at inference. This document covers the implementation in `src/models/retnet/`, its configuration, and how to interpret training results.

---

## Architecture

```
Input token IDs  (B, T)
        │
   ┌────┴────────────────────────┐
   │  wte: token embedding        │  (B, T, n_embd)
   │  dropout                     │  (no position embedding — decay mask encodes position)
   └────────────┬────────────────┘
                │
        ┌───────┴──────┐  × n_layer
        │  RetNetBlock             │
        │    RMSNorm               │  pre-norm
        │    MultiScaleRetention   │  no softmax, per-head gamma decay
        │    RMSNorm               │  pre-norm
        │    SwiGLU FFN            │  residual connection around each
        └───────────────┘
                │
        RMSNorm (final)
                │
        lm_head: Linear(n_embd → vocab_size)   ← weight-tied to wte
                │
        Logits  (B, T, vocab_size)   [training — parallel mode]
        Logits  (B, 1, vocab_size)   [generation]
```

---

## Retention Mechanism

### The causal decay matrix D

For each head `h` with decay rate `gamma_h`, the retention scores between all pairs of positions are defined by:

```
D[h, i, j] = gamma_h^(i - j)   for i >= j   (causal)
D[h, i, j] = 0                 for i <  j   (future tokens masked)
```

where `i` is the query position (current token) and `j` is the key position (past token). Position `j=i-0` (current) has weight `gamma_h^0 = 1.0`; position `j=i-1` has weight `gamma_h`; position `j=i-k` has weight `gamma_h^k`, decaying exponentially into the past.

Because `0 < gamma_h < 1`, older positions receive less weight — the model has a natural forgetting horizon. There is no softmax normalisation.

### Retention computation (parallel mode)

```
Q = q_proj(x)   # (B, T, n_embd) → (B, n_head, T, head_dim)
K = k_proj(x)
V = v_proj(x)

D = decay_mask(T)          # (1, n_head, T, T) — computed from gammas

Retention = (Q @ K^T) ⊙ D  # element-wise multiply by decay, no softmax
Y = Retention @ V          # (B, n_head, T, head_dim)
Y = GroupNorm(Y)           # normalise across heads
output = out_proj(Y)
```

The absence of softmax means retention is a **linear** function of V, enabling exact recurrent formulation at inference.

### Per-head gamma values

Each head gets a unique decay rate. The `n_head` values are log-spaced between `gamma_min` and `gamma_max`:

```python
gammas = 1.0 - exp( linspace( log(1 - gamma_max),
                               log(1 - gamma_min),
                               n_head ) )
```

For `n_head=6`, `gamma_min=0.85`, `gamma_max=0.999`:

| Head | gamma | Effective horizon (tokens) |
|---|---|---|
| 0 | ~0.999 | ~1000 tokens (long range) |
| 1 | ~0.994 | ~167 tokens |
| 2 | ~0.980 | ~50 tokens |
| 3 | ~0.953 | ~21 tokens |
| 4 | ~0.907 | ~11 tokens |
| 5 | ~0.850 | ~6 tokens (short range) |

Heads with higher gamma attend to longer temporal scales; lower-gamma heads focus on recent context. This multi-scale structure allows the model to simultaneously handle both local and global dependencies.

### GroupNorm on retention output

After the retention-weighted sum, `GroupNorm` with `n_head` groups is applied to the concatenated head output `(B, T, n_embd)`. This normalises each head's contribution independently, improving training stability. It is equivalent to applying LayerNorm per head, as described in the RetNet paper.

### Three computation modes

RetNet supports three equivalent computation modes (only the parallel mode is implemented here):

| Mode | Description | Training | Inference |
|---|---|---|---|
| **Parallel** | Full `(T × T)` decay matrix — implemented | Yes | Feasible but O(n²) memory |
| **Recurrent** | Maintain running hidden state per head — not implemented | No | O(1) per step |
| **Chunkwise** | Block-diagonal hybrid — not implemented | Efficient | Efficient |

For generation at inference time, the parallel mode is used (computing on the last token only). A full recurrent inference implementation would maintain a state tensor `h_h ∈ R^(head_dim × head_dim)` per head and update it as `h_t = gamma_h * h_{t-1} + k_t^T v_t`.

### Key differences from softmax attention

| Property | Softmax Attention | Retention |
|---|---|---|
| Score function | `softmax(QK^T / sqrt(d))` | `QK^T ⊙ D` (no softmax, no scaling) |
| Positional encoding | Absolute or RoPE | Implicit via decay matrix D |
| Normalisation | Softmax (per row) | GroupNorm (per head, post-aggregation) |
| Recurrent form | No | Yes — exact O(1) inference |
| Training | O(n²) time and memory | O(n²) time, same memory |
| Long-range decay | No forced decay | Explicit, per-head, monotone |

---

## Parameters

### `RetNetConfig` — `src/models/retnet/config.py`

| Field | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | `50257` | Vocabulary size. GPT-2 tokenizer has 50 257 tokens. |
| `block_size` | `int` | `128` | Maximum context window in tokens. |
| `n_layer` | `int` | `6` | Number of RetNetBlocks. |
| `n_head` | `int` | `6` | Number of retention heads. Each head gets a unique decay gamma. Must divide `n_embd`. |
| `n_embd` | `int` | `384` | Embedding / hidden dimension. |
| `intermediate_size` | `int` | `1024` | SwiGLU FFN hidden dimension. |
| `gamma_min` | `float` | `0.85` | Decay rate for the shortest-range head. |
| `gamma_max` | `float` | `0.999` | Decay rate for the longest-range head. |
| `dropout` | `float` | `0.0` | Dropout probability (applied inside retention). |

**Constraint**: `n_embd % n_head == 0`.

### Parameter count (approximate)

For `retnet_small` (`n_embd=384, n_layer=6, n_head=6, intermediate_size=1024`):

```
Embedding (shared with lm_head):  50257 × 384 ≈ 19.3 M

Per RetNetBlock:
  q_proj:    384 × 384             ≈ 147 K
  k_proj:    384 × 384             ≈ 147 K
  v_proj:    384 × 384             ≈ 147 K
  out_proj:  384 × 384             ≈ 147 K
  GroupNorm: 6 groups, 384 dim     ≈   1 K
  SwiGLU W1: 384 × 1024           ≈ 393 K
  SwiGLU W2: 1024 × 384           ≈ 393 K
  SwiGLU W3: 384 × 1024           ≈ 393 K
  RMSNorms:  2 × 384              ≈   1 K
  Subtotal                        ≈ 1.77 M

Retention gammas:   n_head = 6 scalars   (buffer, not parameter)

Total:
  19.3 M + 6 × 1.77 M            ≈ 29.9 M  (reported ~29 M)
```

---

## Preset Configs

Two ready-to-use model configs are in `configs/retnet_config/model/`.

### `retnet_small.yaml` — ~29 M parameters (6 layers)

```yaml
model_type: retnet
model:
  vocab_size: 50257
  block_size: 128
  n_layer: 6
  n_head: 6
  n_embd: 384
  intermediate_size: 1024
  gamma_min: 0.85
  gamma_max: 0.999
  dropout: 0.0
```

Six retention heads spanning `gamma ∈ [0.85, 0.999]` on a log scale.

### `retnet_medium.yaml` — ~60 M parameters (12 layers)

```yaml
model_type: retnet
model:
  vocab_size: 50257
  block_size: 256
  n_layer: 12
  n_head: 8
  n_embd: 512
  intermediate_size: 1536
  gamma_min: 0.85
  gamma_max: 0.999
  dropout: 0.1
```

Deeper and wider with 8 retention heads. The wider context window (`block_size=256`) is particularly relevant for RetNet since the decay mask covers more temporal scales.

---

## Running RetNet

### Minimal experiment file

```yaml
# configs/retnet_config/experiments/my_retnet_run.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/retnet_small.yaml"
  - "../training/default.yaml"
```

```bash
make prep     MODEL=retnet_config EXP=my_retnet_run
make train    MODEL=retnet_config EXP=my_retnet_run
make generate MODEL=retnet_config EXP=my_retnet_run
```

### Adjusting temporal scale coverage

To make the model focus more on local context, increase `gamma_min`:

```yaml
model:
  gamma_min: 0.92    # all heads have decay > 0.92 — shorter range
```

To extend long-range coverage, lower `gamma_min`:

```yaml
model:
  gamma_min: 0.70    # shorter-range heads decay faster
  gamma_max: 0.9999  # longer-range heads almost never forget
```

---

## Training Config Reference

Defined in `configs/retnet_config/training/default.yaml`.

| Field | Default | Description |
|---|---|---|
| `max_iters` | `20000` | Total optimiser steps. |
| `batch_size` | `32` | Sequences per micro-batch. |
| `block_size` | `128` | Context window — must match `model.block_size`. |
| `gradient_accumulation_steps` | `32` | Micro-batches before each weight update. |
| `max_grad_norm` | `1.0` | Gradient clipping threshold. |
| `eval_interval` | `500` | Evaluation frequency in iterations. |
| `eval_batches` | `500` | Validation batches per evaluation. |
| `checkpoint_path` | `outputs/retnet/checkpoints/` | Checkpoint directory. |
| `optimizer.learning_rate` | `3e-4` | Peak learning rate. |
| `optimizer.betas` | `[0.9, 0.95]` | AdamW momentum coefficients. |
| `optimizer.weight_decay` | `0.1` | L2 regularisation. |
| `scheduler.warmup_steps` | `1000` | Linear LR warmup steps. |
| `scheduler.min_lr` | `3e-5` | Minimum LR after cosine decay. |

---

## Outputs and Results

### Checkpoints

Written to `outputs/retnet/checkpoints/`. The gamma values (`self.gammas`) are registered as non-trainable buffers and are included in the checkpoint. They are deterministic from the config but their presence in the state dict avoids confusion.

### Interpreting validation loss

RetNetSLM achieved a best validation loss of **~2.56** at 20k steps — competitive with Mamba and LLaMA and within 0.02 nats of both. The GroupNorm inside the retention layer is important for stable convergence; without it, the unnormalised retention scores can grow or shrink uncontrollably. Generation quality is heavily fragmented relative to the quantitative ranking, suggesting the retention mechanism struggles to maintain grammatical span at this loss level.

---

## File Locations

| Purpose | File |
|---|---|
| Config dataclass | `src/models/retnet/config.py` |
| Model implementation | `src/models/retnet/model.py` |
| Plugin registration | `src/models/retnet/__init__.py` |
| Preset configs | `configs/retnet_config/model/retnet_small.yaml`, `retnet_medium.yaml` |
| RMSNorm primitive | `src/core/normalization.py` |
| SwiGLU primitive | `src/core/ffn.py` |
| Generation loop | `src/core/generation.py` |

---

## References

Sun et al., 2023 — "Retentive Network: A Successor to Transformer for Large Language Models." arXiv:2307.08621.
