# SLM Architecture Zoo — Master Reference

The dominant paradigm for comparing language model architectures is evaluation at 7B+ parameters, where differences in hardware utilisation and dataset composition confound architectural comparisons. No controlled study consistently benchmarks diverse sequence modelling architectures at 30M parameters — the scale reachable on a single consumer GPU — under a fixed training budget and shared data pipeline. This gap matters: small-scale deployments (edge inference, embedded systems, on-device NLP) require architecture decisions that cannot be extrapolated from 7B-scale results.

This document is the central reference for a controlled experiment benchmarking 8 SLM architectures at 28–53M parameters on TinyStories. All models share the same dataset, tokenizer, training budget, optimizer, and learning rate schedule. The central question is: **when all else is equal — dataset, tokenizer, parameter count, training budget — which architectural innovation actually matters at small scale?** Individual architecture reports are linked at the end. Measured results are in `reports/10_compare.md`.

---

## Architecture Registry

| # | Key | Params (total) | Active params | Attention? | Distinguishing mechanism |
|---|---|---|---|---|---|
| 1 | `gpt` | 30.0 M | 30.0 M | Full MHA | Vanilla decoder |
| 2 | `llama` | 28.7 M | 28.7 M | GQA + RoPE | RMSNorm + SwiGLU + GQA |
| 3 | `deepseek_moe` | 34.7 M | ~28 M | GQA + RoPE | Shared + routed MoE experts |
| 4 | `mamba` | 30.4 M | 30.4 M | None | Selective SSMs |
| 5 | `rwkv` | 53.0 M | 53.0 M | None | WKV recurrence |
| 6 | `jamba` | 34.8 M | 34.8 M | Half layers | Hybrid SSM-attention |
| 7 | `bitnet` | 28.8 M | 28.8 M | GQA + RoPE | Ternary {-1, 0, +1} weights |
| 8 | `retnet` | 29.9 M | 29.9 M | Retention (no softmax) | Multi-scale exponential decay |

---

## Shared Experimental Protocol

| Setting | Value |
|---|---|
| Dataset | TinyStories (`roneneldan/TinyStories`) |
| Tokenizer | GPT-2 (`tiktoken gpt2`, vocab = 50 257) |
| Context length | 128 tokens |
| Training iterations | 20 000 |
| Batch size | 32 (micro) × 32 (grad accum) = 1 024 effective |
| Optimizer | AdamW, lr = 3e-4, β = (0.9, 0.95), wd = 0.1 (baseline; see exceptions below) |
| LR schedule | Linear warmup (1 000 steps) → cosine decay to 3e-5 |
| Gradient clipping | 1.0 |
| Evaluation interval | Every 500 iterations |
| Random seed | 42 |

**Exceptions to shared protocol:**
- **MiniGPT**: lr = 1e-4, min_lr = 5e-4, max_grad_norm = 0.5, eps = 1e-9 (original nanoGPT notebook settings preserved).
- **RWKV**: β = (0.9, 0.99), wd = 0.01 (RWKV-specific optimizer tuning).
- **BitNet**: wd = 0.0, warmup_steps = 2000 (adjusted for ternary weight training dynamics).
- **DeepSeek MoE**: batch_size = 16, gradient_accumulation_steps = 64 (same effective batch of 1 024; micro-batch halved to reduce memory from expert activations).

---

## Taxonomy

**(a) Attention-based — GPT, LLaMA, RetNet.** These models compute token interactions via explicit query-key similarity; their inductive bias is that any two positions in the sequence can interact directly, with the attention matrix encoding relative importance.

**(b) Attention-free SSM/RNN — Mamba, RWKV.** These models encode history in a fixed-size hidden state updated recurrently; their inductive bias is a compression of the past, with no direct pairwise token interaction. Mamba's state update is input-dependent (selective); RWKV's uses exponential decay with a learned bonus for the current token.

**(c) Hybrid — Jamba.** Interleaves SSM layers (O(n) context compression) with attention layers (O(n²) precise retrieval), aiming to inherit the strengths of both families within a single stack.

**(d) Structural innovation on attention-based — BitNet (quantization), DeepSeek (conditional compute).** BitNet applies ternary weight quantization to a standard attention-based architecture; DeepSeek replaces dense FFNs with a mixture-of-experts layer to decouple total parameter capacity from per-token compute cost.

---

## Shared Core Primitives

| Module | File | Used by |
|---|---|---|
| `RMSNorm` | `src/core/normalization.py` | LLaMA, MoE, Mamba, RWKV, Jamba, BitNet, RetNet |
| `precompute_freqs_cis`, `apply_rotary_emb` | `src/core/rope.py` | LLaMA, MoE, BitNet |
| `SwiGLU` | `src/core/ffn.py` | LLaMA, MoE, Jamba, RetNet |
| `MambaBlock` | `src/core/mamba_block.py` | Mamba, Jamba |

---

## Quick-Start

```bash
# Data prep — run once
make prep MODEL=miniGPT_config

# Train any architecture
make train MODEL=gpt
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

## Individual Architecture Reports

- [02_gpt.md](02_gpt.md) — GPT (vanilla decoder, Full MHA)
- [03_llama.md](03_llama.md) — LLaMA (RMSNorm, RoPE, GQA, SwiGLU)
- [04_bitnet.md](04_bitnet.md) — BitNet (ternary weight quantization)
- [05_retnet.md](05_retnet.md) — RetNet (retention mechanism, no softmax)
- [06_deepseek_moe.md](06_deepseek_moe.md) — DeepSeek MoE (shared + routed experts)
- [07_jamba.md](07_jamba.md) — Jamba (hybrid SSM-attention)
- [08_mamba.md](08_mamba.md) — Mamba (selective state spaces)
- [09rwkv.md](09rwkv.md) — RWKV (WKV recurrence, O(1) inference)
- [10_compare.md](10_compare.md) — Controlled comparison: results and analysis
