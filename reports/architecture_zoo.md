# SLM Architecture Zoo — Master Comparison Report

This document is the central reference for the Architecture Zoo experiment: building and benchmarking 8 SLM architectures at the same ~30–50 M parameter scale, trained on the same dataset with the same tokenizer and training budget.

For per-architecture implementation details, see the individual reports linked in the table below.

---

## The Hypothesis

Every major LLM paper benchmarks against the previous state-of-the-art at scale (7B+). Nobody runs controlled experiments at 30 M parameters where the compute budget fits on a laptop GPU. This project answers the question:

> **When all else is equal — dataset, tokenizer, parameter count, training budget — which architectural innovation actually matters at small scale?**

---

## Architecture Registry

| # | Registry Key | Report | Total Params | Active Params | Attention? | Key Innovation |
|---|---|---|---|---|---|---|
| 1 | `gpt` | [gpt.md](gpt.md) | ~29 M | 29 M | Full MHA | Vanilla transformer baseline |
| 2 | `llama` | [llama.md](llama.md) | ~31 M | 31 M | GQA + RoPE | RMSNorm + SwiGLU + GQA |
| 3 | `deepseek_moe` | [deepseek_moe.md](deepseek_moe.md) | ~48 M | ~25 M | GQA + RoPE | Shared + routed MoE experts |
| 4 | `mamba` | [mamba.md](mamba.md) | ~32 M | 32 M | **None** | Selective state spaces, O(n) |
| 5 | `rwkv` | [rwkv.md](rwkv.md) | ~33 M | 33 M | **None** | Linear attention RNN |
| 6 | `jamba` | [jamba.md](jamba.md) | ~35 M | 35 M | Partial (half layers) | Hybrid SSM + Attention |
| 7 | `bitnet` | [bitnet.md](bitnet.md) | ~30 M | 30 M | GQA + RoPE | Ternary {-1, 0, +1} weights |
| 8 | `retnet` | [retnet.md](retnet.md) | ~29 M | 29 M | Retention (no softmax) | Multi-scale exponential decay |

---

## Shared Experimental Protocol

All models are trained under identical conditions to make results comparable:

| Setting | Value |
|---|---|
| Dataset | TinyStories (`roneneldan/TinyStories`) |
| Tokenizer | GPT-2 (`tiktoken gpt2`, vocab=50 257) |
| Context length | 128 tokens |
| Training iterations | 20 000 |
| Batch size | 32 (micro) × 32 (grad accum) = 1 024 effective |
| Optimizer | AdamW, lr=3e-4, β=(0.9, 0.95), wd=0.1 |
| LR schedule | Linear warmup (1 000 steps) → cosine decay to 3e-5 |
| Gradient clipping | 1.0 |
| Evaluation interval | Every 500 iterations |
| Random seed | 42 |

---

## Quick-Start for Each Architecture

```bash
# Data prep is shared — only run once
make prep MODEL=miniGPT_config

# Train any architecture
make train MODEL=gpt             # baseline
make train MODEL=llama_config
make train MODEL=deepseek_moe_config
make train MODEL=mamba_config
make train MODEL=rwkv_config
make train MODEL=jamba_config
make train MODEL=bitnet_config
make train MODEL=retnet_config

# Generate from any trained checkpoint
make generate MODEL=llama_config
```

---

## Benchmarking Plan

After training, evaluate each model on the following metrics:

### Quality Metrics

| Metric | How to measure |
|---|---|
| **Validation perplexity** | `exp(val_loss)` on held-out TinyStories test set |
| **Generation coherence** | Score 1–5 on 10 standard prompts (see below) |
| **Cloze accuracy** | Fraction of correct next-word predictions on 1 000 test examples |

### Efficiency Metrics

| Metric | How to measure |
|---|---|
| **Training wall clock** | Seconds per 1 000 iterations |
| **Inference throughput** | Tokens/sec at batch=1, generating 100 tokens |
| **Peak inference memory** | `torch.cuda.max_memory_allocated()` in MB |
| **Active params / total params** | For MoE: efficiency ratio |

### Parameter Efficiency

```
param_efficiency = (gpt_perplexity - model_perplexity) / model_active_params
```

Higher is better: more perplexity improvement per active parameter.

---

## Standardised Evaluation Prompts

Use these exact 10 prompts for **all** models. Report the generated text verbatim alongside a coherence score.

```python
EVAL_PROMPTS = [
    "Once upon a time there was a little girl named",
    "The big brown dog liked to",
    "One day, the sun was shining and",
    "Lily went to the park with her",
    "There was a magical castle where",
    "The little boy was very sad because",
    "Mom said it was time to go to",
    "The cat and the bird were best",
    "It was a dark and stormy night when",
    "The children found a secret door that",
]
```

**Coherence scoring rubric (1–5):**

| Score | Criterion |
|---|---|
| 1 | Incoherent, grammatically broken |
| 2 | Mostly incoherent with occasional valid phrases |
| 3 | Mostly coherent but off-topic or repetitive |
| 4 | Coherent and on-topic, minor errors |
| 5 | Fluent, creative, child-story appropriate |

---

## Shared Core Primitives

The following framework-level building blocks are shared across architectures to avoid duplication:

| Module | File | Used by |
|---|---|---|
| `RMSNorm` | `src/core/normalization.py` | LLaMA, MoE, Mamba, RWKV, Jamba, BitNet, RetNet |
| `precompute_freqs_cis`, `apply_rotary_emb` | `src/core/rope.py` | LLaMA, MoE, Jamba, BitNet |
| `SwiGLU` | `src/core/ffn.py` | LLaMA, MoE, Jamba, RetNet |
| `MambaBlock` | `src/core/mamba_block.py` | Mamba, Jamba |

---

## Architectural Decision Map

```
                   ATTENTION-BASED
                        │
         ┌──────────────┼──────────────┐
         │              │              │
       GPT-2         LLaMA          RetNet
    (baseline)    (GQA+RoPE+    (no softmax,
                   SwiGLU)      decay mask)
         │
         ├──────── BitNet (ternary weights)
         │
         └──────── DeepSeek MoE (sparse experts)

                   NO-ATTENTION
                        │
         ┌──────────────┼
         │              │
       Mamba           RWKV
    (selective      (linear RNN,
      SSM, O(n))    token shift)

                   HYBRID
                        │
                      Jamba
                  (Mamba even +
                  Attn odd layers)
```

---

## Expected Results (Hypothesis)

Based on prior literature, rough expectations for TinyStories perplexity:

| Architecture | Expected Val Loss | Rationale |
|---|---|---|
| GPT-2 baseline | 1.4–1.7 | Known result from training |
| LLaMA-style | 1.3–1.6 | RoPE + SwiGLU should improve slightly |
| DeepSeek MoE | 1.2–1.5 | More total capacity with same active params |
| Mamba | 1.5–1.8 | SSM may struggle on short context where attention shines |
| RWKV | 1.5–1.9 | Token-shift recurrence may under-use short context |
| Jamba Hybrid | 1.3–1.6 | Best of both worlds, but more complex to optimise |
| BitNet | 1.6–2.0 | Quantization loss, especially at 30M scale |
| RetNet | 1.4–1.7 | Should be competitive with GPT, retention ≈ softmax |

These are hypotheses — the experiment will determine the truth.

---

## File Locations

| Resource | Path |
|---|---|
| GPT baseline | `src/models/gpt/` |
| LLaMA model | `src/models/llama/` |
| DeepSeek MoE model | `src/models/deepseek_moe/` |
| Mamba model | `src/models/mamba/` |
| RWKV model | `src/models/rwkv/` |
| Jamba model | `src/models/jamba/` |
| BitNet model | `src/models/bitnet/` |
| RetNet model | `src/models/retnet/` |
| Shared primitives | `src/core/normalization.py`, `rope.py`, `ffn.py`, `mamba_block.py` |
| Configs | `configs/<model>_config/` |
| Scripts | `scripts/<model>_scripts/` |
| Tests | `tests/test_models/` |
