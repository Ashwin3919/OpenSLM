# Gemma 3 — Architecture Reference

Gemma3SLM is a decoder-only transformer that combines several architectural innovations from Google DeepMind's Gemma 3 family: interleaved local/global attention with dual RoPE, per-head QK normalisation, tanh logit soft-capping in both attention and output layers, GeGLU feed-forward, and pre+post normalisation around each sub-block. This document covers the implementation in `src/models/gemma3/`, its configuration, and the motivation for each design choice.

---

## Architecture

<!--
```
Input token IDs  (B, T)
        │
   ┌────┴────────────────────────┐
   │  wte: token embedding        │  (B, T, n_embd)
   │  dropout                     │  (no position embedding — RoPE is applied per-layer)
   └────────────┬────────────────┘
                │
        ┌───────┴──────┐  × n_layer
        │  Gemma3Block             │
        │    pre_attn_norm (RMS)   │
        │    Gemma3Attention       │  local (sliding window) or global (full causal)
        │    post_attn_norm (RMS)  │     + QK norm + RoPE + attention logit soft-cap
        │    pre_ffn_norm  (RMS)   │  residual connection: x = x + post_norm(sub_block(pre_norm(x)))
        │    GeGLU FFN             │
        │    post_ffn_norm (RMS)   │
        └───────────────┘
                │
        RMSNorm (final)
                │
        lm_head: Linear(n_embd → vocab_size)   ← weight-tied to wte
                │
        Final logit soft-cap: tanh(logits / cap) * cap
                │
        Logits  (B, T, vocab_size)   [training]
        Logits  (B, 1, vocab_size)   [generation]
```
-->

**Diagram Explanation:**
- **Local vs Global layers**: Each block is either a local (sliding-window) attention layer or a global (full causal) attention layer, assigned by layer index. The default config uses a 5:1 ratio — layers 0–4 are local, layer 5 is global.
- **QK normalisation**: RMSNorm is applied to Q and K per head (over `head_dim`) after projection and before RoPE. Not the same as the block-level RMSNorm — this operates on the 64-dimensional head vectors, not the 384-dimensional hidden state.
- **Pre+post norm**: Every sub-block has both a pre-norm (normalise input before the computation) and a post-norm (normalise output before the residual add). Four RMSNorm layers per block total.
- **Logit soft-capping**: Applied twice — once on attention scores before softmax, once on output logits before CE loss or sampling.

---

## Key Innovations

### 1. Interleaved Local and Global Attention

Most layers use **sliding-window (local) attention**: each token attends only to the previous `sliding_window` tokens. Every `n_layer / (len(global_layers))` layers, a **global** layer performs full causal attention over the entire context.

```
Layer index:   0      1      2      3      4      5
               ▼      ▼      ▼      ▼      ▼      ▼
Type:        Local  Local  Local  Local  Local  Global
             (window=64)                      (full causal)
```

**Why this ratio matters:** Local attention captures local syntax and word-level patterns efficiently at O(T × W) cost instead of O(T²). Global attention layers give the model a mechanism to retrieve long-range dependencies. The 5:1 ratio is the Gemma 3 default and provides a practical trade-off between efficiency and expressiveness. At this project's context length of 128 tokens, the window of 64 covers half the context, which is still a meaningful constraint.

**Two sets of RoPE frequencies:** Local and global layers use different RoPE base frequencies:

| Layer type | `rope_theta` | Effect |
|---|---|---|
| Local | `10 000` (same as LLaMA) | Encodes relative position within the local window |
| Global | `1 000 000` | Encodes position across the full context; the large theta spreads frequency components more slowly, preserving positional distinctions at longer ranges |

Both frequency tensors are precomputed at init and registered as non-persistent buffers:

```python
freqs_local  = precompute_freqs_cis(head_dim, block_size, theta=10_000)
freqs_global = precompute_freqs_cis(head_dim, block_size, theta=1_000_000)
```

The forward loop selects the correct tensor per layer before passing it to the block.

### 2. Sliding-Window Attention Mask

The local attention mask is precomputed at init as a boolean tensor of shape `(block_size, block_size)`:

```python
i = torch.arange(block_size).unsqueeze(1)   # (T, 1)
j = torch.arange(block_size).unsqueeze(0)   # (1, T)
local_mask = (j <= i) & ((i - j) < sliding_window)
```

- `j <= i`: causal constraint — no future tokens
- `(i - j) < sliding_window`: local constraint — only the past `W` tokens

At forward time, `local_mask[:T, :T]` is sliced to the actual sequence length and broadcast over the batch and head dimensions before being applied via `masked_fill(~mask, -inf)`.

Global layers use the standard lower-triangular causal mask (all `j <= i` positions are True), also precomputed and registered as a buffer.

### 3. QK Normalisation

After the Q, K, V projections and before RoPE, the query and key tensors are normalised per-head using RMSNorm over `head_dim`:

```python
q = self.q_norm(q)   # RMSNorm(head_dim=64), applied to (B, T, n_head, 64)
k = self.k_norm(k)   # RMSNorm(head_dim=64), applied to (B, T, n_kv_head, 64)
```

**Why this is done:** Dot-product attention scores scale with the magnitude of Q and K. Without normalisation, as models grow deeper or wider, Q and K magnitudes can drift, causing attention scores to become very large and pushing softmax into near-one-hot distributions (attention collapse). QK norm prevents this without requiring aggressive gradient clipping. The normalisation operates at `head_dim` granularity (not `n_embd`) so each head's internal geometry is independently stabilised.

This is applied before RoPE so that the normalisation does not interfere with the rotational structure that RoPE imposes on the head vectors.

### 4. Attention Logit Soft-Capping

After computing raw attention scores and before softmax:

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # (B, H, T, T)
scores = torch.tanh(scores / attn_logit_cap) * attn_logit_cap
```

The tanh function maps any real number to `(-1, +1)`, so multiplying by `attn_logit_cap = 50.0` bounds all pre-softmax scores to `(-50, +50)`. This prevents extreme logit values from producing collapsed attention distributions, especially in early training when Q and K are poorly calibrated. The cap is applied after QK normalisation — they address different failure modes and are complementary.

**Note on implementation:** Because soft-capping must be applied between the raw dot-product and softmax, this implementation computes attention manually (Q @ Kᵀ → scale → tanh cap → mask → softmax → @ V) rather than using `F.scaled_dot_product_attention`. This is correct but does not benefit from FlashAttention kernel fusion. For a 30M model on TinyStories this has no practical impact.

### 5. GeGLU Feed-Forward Network

```
output = W2( GELU(W1(x)) * W3(x) )
```

Identical structure to SwiGLU (three matrices, no bias) but uses GELU instead of SiLU as the gate activation. Both are smooth approximations to ReLU; GELU has slightly heavier tails and is differentiable everywhere. The Gemma family uses GeGLU throughout; the SwiGLU vs GeGLU difference is minor in practice at this scale.

| | SwiGLU (LLaMA) | GeGLU (Gemma 3) |
|---|---|---|
| Gate activation | `SiLU(x) = x * sigmoid(x)` | `GELU(x) ≈ x * Φ(x)` |
| Parameter count | Same (3 matrices) | Same (3 matrices) |

### 6. Pre + Post Normalisation

Standard pre-norm (LLaMA-style):
```
x = x + sub_block(RMSNorm(x))
```

Gemma 3 pre+post norm:
```
x = x + RMSNorm_post( sub_block( RMSNorm_pre(x) ) )
```

The post-norm is applied to the sub-block output *before* the residual addition. This results in four RMSNorm layers per block (pre-attn, post-attn, pre-ffn, post-ffn). The motivation is additional gradient stabilisation — the post-norm prevents large activations from the sub-block from flowing directly into the residual stream.

---

## Parameters

### `Gemma3Config` — `src/models/gemma3/config.py`

| Field | Type | Default | Description |
|---|---|---|---|
| `vocab_size` | `int` | `50257` | Vocabulary size. GPT-2 tokenizer has 50 257 tokens. |
| `block_size` | `int` | `128` | Maximum context window in tokens. |
| `n_layer` | `int` | `6` | Number of Gemma3Blocks. |
| `n_head` | `int` | `6` | Number of query attention heads. Must divide `n_embd`. |
| `n_kv_head` | `int` | `2` | Number of KV heads (GQA). Must divide `n_head`. Repeat factor = `n_head / n_kv_head = 3`. |
| `n_embd` | `int` | `384` | Embedding / hidden dimension. |
| `intermediate_size` | `int` | `1024` | GeGLU hidden dim (≈ 8/3 × n_embd ≈ 1024 for n_embd=384). |
| `sliding_window` | `int` | `64` | Local attention window (tokens). |
| `local_rope_theta` | `float` | `10 000.0` | RoPE base frequency for local attention layers. |
| `global_rope_theta` | `float` | `1 000 000.0` | RoPE base frequency for global attention layers. |
| `attn_logit_cap` | `float` | `50.0` | Tanh soft-cap on raw attention scores. |
| `final_logit_cap` | `float` | `30.0` | Tanh soft-cap on output logits. |
| `global_layers` | `List[int]` | `[5]` | Layer indices using full causal attention. All others use local. |
| `dropout` | `float` | `0.0` | Dropout probability. |

### Parameter Count (approximate)

For `gemma3_small` (`n_embd=384, n_layer=6, n_head=6, n_kv_head=2`):

```
Embedding (shared with lm_head):
  50257 × 384 ≈ 19.3 M   (full precision, weight-tied)

Per Gemma3Block (all 6 layers share the same architecture):
  GQA attention:
    Q:   384 × 384   = 147 456
    K:   384 × 128   =  49 152   (n_kv_head=2, head_dim=64)
    V:   384 × 128   =  49 152
    O:   384 × 384   = 147 456
    QK norms: 2 × 64 ≈     128   (negligible)
    Subtotal                    ≈ 393 K

  GeGLU FFN:
    W1:  384 × 1024  = 393 216   (gate)
    W2: 1024 × 384   = 393 216   (down)
    W3:  384 × 1024  = 393 216   (up)
    Subtotal                    ≈ 1.18 M

  4 × RMSNorm(384): 4 × 384    ≈   1.5 K   (negligible)

  Per block total              ≈ 1.57 M

Total:
  19.3 M + 6 × 1.57 M         ≈ 28.7 M trainable parameters
```

All parameters are active for every token (dense model, no sparse routing).

---

## Preset Configs

Two ready-to-use model configs in `configs/gemma3_config/model/`.

### `gemma3_small.yaml` — ~29 M parameters

```yaml
model_type: gemma3
model:
  vocab_size: 50257
  block_size: 128
  n_layer: 6
  n_head: 6
  n_kv_head: 2
  n_embd: 384
  intermediate_size: 1024
  sliding_window: 64
  local_rope_theta: 10000.0
  global_rope_theta: 1000000.0
  attn_logit_cap: 50.0
  final_logit_cap: 30.0
  global_layers: [5]
  dropout: 0.0
```

Layers 0–4 are local (sliding-window, window=64). Layer 5 is global (full causal). 5:1 local:global ratio.

### `gemma3_medium.yaml` — ~60 M parameters

```yaml
model_type: gemma3
model:
  vocab_size: 50257
  block_size: 256
  n_layer: 12
  n_head: 8
  n_kv_head: 2
  n_embd: 512
  intermediate_size: 1536
  sliding_window: 128
  local_rope_theta: 10000.0
  global_rope_theta: 1000000.0
  attn_logit_cap: 50.0
  final_logit_cap: 30.0
  global_layers: [5, 11]
  dropout: 0.0
```

12 layers with 2 global layers (indices 5 and 11) — maintains the 5:1 ratio.

---

## Running Gemma 3

### Minimal experiment file

```yaml
# configs/gemma3_config/experiments/my_gemma3_run.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/gemma3_small.yaml"
  - "../training/default.yaml"
```

```bash
make prep     MODEL=gemma3_config
make train    MODEL=gemma3_config
make generate MODEL=gemma3_config
```

Or with explicit config path:

```bash
python main.py train --config configs/gemma3_config/experiments/exp_001_baseline.yaml
```

---

## Training Config Reference

Defined in `configs/gemma3_config/training/default.yaml`.

| Field | Default | Description |
|---|---|---|
| `max_iters` | `20000` | Total optimiser steps. |
| `batch_size` | `32` | Sequences per micro-batch. |
| `block_size` | `128` | Context window — must match `model.block_size`. |
| `gradient_accumulation_steps` | `32` | Effective batch = `32 × 32 = 1 024` tokens. |
| `max_grad_norm` | `1.0` | Gradient clipping threshold. |
| `eval_interval` | `500` | Evaluation frequency in iterations. |
| `eval_batches` | `500` | Validation batches per evaluation. |
| `checkpoint_path` | `outputs/gemma3/checkpoints/` | Checkpoint directory. |
| `optimizer.learning_rate` | `3e-4` | Peak learning rate — standard shared protocol. |
| `optimizer.betas` | `[0.9, 0.95]` | AdamW momentum coefficients. |
| `optimizer.weight_decay` | `0.1` | L2 regularisation. |
| `scheduler.warmup_steps` | `1000` | Linear LR warmup steps. |
| `scheduler.min_lr` | `3e-5` | Minimum LR after cosine decay. |

No training-protocol exceptions — Gemma 3's architecture works correctly with the standard shared hyperparameters used by GPT, LLaMA, Mamba, RetNet, and Jamba.

---

## Implementation Notes

### Why manual attention instead of `F.scaled_dot_product_attention`

`F.scaled_dot_product_attention` fuses the scale, mask, softmax, and dropout into a single kernel (FlashAttention when available). However, it does not support inserting arbitrary operations between the raw logits and the softmax. Since Gemma 3's attention logit soft-capping must be applied after `Q @ Kᵀ / sqrt(d)` and before softmax, the attention is computed manually:

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * scale
scores = torch.tanh(scores / attn_logit_cap) * attn_logit_cap  # soft-cap
scores = scores.masked_fill(~mask, float("-inf"))               # masking
probs  = F.softmax(scores, dim=-1)
y      = torch.matmul(probs, v)
```

At 30M parameters on TinyStories, the absence of FlashAttention does not meaningfully affect throughput.

### Mask precomputation

Both the sliding-window mask and the causal mask are built once at `__init__` time and registered as non-persistent buffers. This means:
- They are moved to the correct device automatically when `model.to(device)` is called.
- They are **not** saved in checkpoints (they are deterministically reconstructed from the config at load time).
- At forward time, `mask[:T, :T]` slices to the actual sequence length efficiently.

### Global layers as a Python list in config

The `global_layers` field is a `List[int]` in the dataclass and `[5]` in the YAML. The config loader preserves this as a Python list. The model checks `i in self.config.global_layers` per block, which is O(len(global_layers)) — negligible for typical values (1–3 global layers).

---

## File Locations

| Purpose | File |
|---|---|
| Config dataclass | `src/models/gemma3/config.py` |
| Model implementation | `src/models/gemma3/model.py` |
| Plugin registration | `src/models/gemma3/__init__.py` |
| Preset configs | `configs/gemma3_config/model/gemma3_small.yaml`, `gemma3_medium.yaml` |
| Training config | `configs/gemma3_config/training/default.yaml` |
| Experiments | `configs/gemma3_config/experiments/` |
| RMSNorm primitive | `src/core/normalization.py` |
| RoPE utilities | `src/core/rope.py` |
| Generation loop | `src/core/generation.py` |

---

## References

Gemma Team, Google DeepMind, 2025 — "Gemma 3 Technical Report." arXiv:2503.19786.

Gemma Team, Google DeepMind, 2024 — "Gemma 2: Improving Open Language Models at a Practical Size." arXiv:2408.00118.

Su et al., 2021 — "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.

Shazeer, 2020 — "GLU Variants Improve Transformer." arXiv:2002.05202.

Touvron et al., 2023 — "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." arXiv:2307.09288.

Ainslie et al., 2023 — "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." arXiv:2305.13245.

Beltagy et al., 2020 — "Longformer: The Long-Document Transformer." arXiv:2004.05150.
(Sliding-window attention reference)
