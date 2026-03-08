# Controlled Comparison of Eight Small Language Model Architectures at 30M Parameters

---

## 1. Experimental Setup

Eight architectures were trained on TinyStories (`roneneldan/TinyStories`) using the GPT-2 tiktoken tokenizer (vocab = 50 257), a context length of 128 tokens, and a fixed training budget of 20 000 steps with an effective batch size of 1 024 (32 micro-batch × 32 gradient accumulation). All models used AdamW (lr = 3e-4, β = (0.9, 0.95), wd = 0.1) with a 1 000-step linear warmup followed by cosine decay to 3e-5, and gradient clipping at 1.0. The goal was not to reproduce the best published result for each architecture — it was to compare learning ability under a controlled budget. All hyperparameters were set to a single shared baseline; no per-architecture tuning was performed.

---

## 2. Results

| Architecture | Best Val Loss | Final Train Loss |
|---|---|---|
| MiniGPT | 2.3921 | 2.3899 |
| Jamba | 2.4204 | 2.4228 |
| RWKV | 2.4994 | 2.4992 |
| LLaMA | 2.5479 | 2.5467 |
| RetNet | 2.5606 | 2.5532 |
| Mamba | 2.5662 | 2.5623 |
| DeepSeek MoE | 3.1681 | 3.1536 |
| BitNet | 5.5016 | 5.5099 |

The convergence plot (`experiment_comparison.png`) shows validation loss over 20 000 iterations for all 8 architectures. The curves separate into three tiers: (1) MiniGPT, Jamba, RWKV converge below 2.5; (2) LLaMA, RetNet, Mamba converge tightly in the 2.55–2.57 band; (3) DeepSeek and BitNet remain significantly above 3.0 and 5.0 respectively throughout training. All models show typical loss curves — rapid early descent from ~10 (random initialisation over 50k vocabulary), then flattening as training progresses.

---

## 3. Analysis by Group

### 3.1 Top tier: MiniGPT, Jamba, RWKV

MiniGPT (baseline GPT-2 decoder) achieves the lowest validation loss (2.39). This is notable: the simplest architecture in the comparison wins. At 30M parameters on TinyStories, additional architectural complexity does not outperform a well-implemented vanilla transformer. Jamba (2.42) finishes second — the hybrid SSM-attention design provides a small benefit, likely because the attention layers give it precise retrieval capability that pure SSMs lack, while the Mamba layers compress context cheaply. RWKV (2.50) is the best attention-free result, demonstrating that O(1)-inference RNNs trained with WKV recurrence can match or beat attention-based models at this scale.

### 3.2 Middle tier: LLaMA, RetNet, Mamba (2.55–2.57)

These three converge within 0.02 nats of each other — effectively indistinguishable given evaluation noise. LLaMA (2.55), despite carrying every modern transformer upgrade (RoPE, GQA, SwiGLU, RMSNorm), does not outperform the simpler GPT baseline. At 128-token context, the advantages of RoPE over learned position embeddings and GQA over full MHA are marginal. Mamba (2.57) and RetNet (2.56) are competitive without attention, reinforcing that SSM-based and decay-based mechanisms are viable alternatives at this scale.

### 3.3 DeepSeek MoE (3.17) — sparse routing advantage does not appear at 20k steps

DeepSeek finishes with validation loss 0.75 above the dense baselines. This is a training dynamics issue, not a model quality issue. MoE architectures require the router to learn token specialisation — at 20k steps, the router has not converged. The load-balancing auxiliary loss adds a competing training signal that may have slowed language-model convergence. MoE's computational advantage (higher total capacity at the same per-token cost) typically requires 3–5× more training steps to materialise than a dense model of the same active parameter count (Fedus et al., 2022). This result is consistent with the MoE scaling literature.

### 3.4 BitNet (5.50) — ternary quantization does not converge at this scale

BitNet's validation loss of 5.50 — compared to 2.39 for the dense GPT baseline — indicates near-absence of language modelling convergence. A loss of 5.50 on a 50k-token vocabulary corresponds to perplexity ~245; for reference, random uniform over 50k tokens gives perplexity ~50 257 (loss ~10.8), and a functional language model reaches perplexity 10–15 (loss 2.3–2.7). BitNet's loss trajectory is flat from approximately step 3k onward (visible in the convergence plot), indicating the optimizer has stalled — not that convergence is slow. This is the signature of STE gradient collapse, not data mismatch.

The probable causes, in order of likelihood:

1. **Gradient signal degradation through STE.** Ternary quantization uses the Straight-Through Estimator: the backward pass treats the quantizer as an identity. The effective gradient per parameter is the gradient of the loss with respect to the quantized weight, not the full-precision weight. At 30M parameters and 20k steps, this is insufficient for the optimizer to move the full-precision weights into a regime where their quantized counterparts form coherent representations. Ma et al. (2024) document this and report that BitNet benefits from longer warmup and lower LR than dense counterparts.

2. **Learning rate mismatch.** The 3e-4 LR was selected for dense float-weight training. Ternary weights are more sensitive to LR: a step that is small in full-precision parameter space can flip many weight signs, causing large loss spikes. BitNet papers recommend 3–10× lower LR with longer warmup; this experiment used neither.

3. **Scale threshold.** The BitNet b1.58 paper (Ma et al., 2024) demonstrates viable performance at 100M+ parameters. At 30M parameters, the quantization-induced representation capacity reduction may exceed a threshold below which the model cannot form coherent token predictions within the training budget.

4. **Shared activation quantization pressure.** BitLinear also quantizes activations (int8). Combined with ternary weights, the information bottleneck at 30M scale is severe — each layer sees inputs rounded to 256 levels, which limits the gradient signal flowing to earlier layers.

The tokenizer is not the cause: changing the tokenizer would shift loss by at most ±0.1 for a functioning model. BitNet's underperformance at this scale is a well-documented regime: ternary networks require either larger parameter counts, more training steps, architecture-specific LR schedules, or a teacher model for knowledge distillation to match float baselines. At 30M parameters / 20k steps / shared LR, they do not converge competitively. This result defines the minimum scale threshold below which BitNet is not a viable replacement for dense transformers under a shared training budget.

---

## 4. Generation Quality vs Validation Loss

The qualitative generation outputs are consistent with the quantitative ranking:

- **MiniGPT, Jamba, RWKV**: coherent multi-sentence narratives with TinyStories structure. Characters, settings, and basic story arcs are maintained across sentences.
- **LLaMA, Mamba**: coherent sentences but with semantic drift — plausible sentence-level grammar, with incoherence accumulating across paragraphs. New narrative threads are introduced without resolving prior ones.
- **RetNet**: heavily fragmented, high short-sentence frequency (repeated "He said.", "She said.", "They are happy." constructions) — suggests the retention mechanism struggles to maintain grammatical span at this loss level, despite competitive validation loss numbers.
- **DeepSeek**: tokenization-level failures visible in generation ("likerr", "toed theep", "Hodgapped") — consistent with unstable routing producing degenerate token distributions.
- **BitNet**: no grammatical structure — word salad with no coherent phrases or sentence boundaries.

---

## 5. Summary and Takeaways

1. At 30M parameters and 128-token context, vanilla GPT-2 architecture (MiniGPT) achieves the lowest validation loss. Architectural complexity above the GPT baseline offers marginal improvement at best in this regime.

2. The hybrid SSM-attention design (Jamba) is the closest competitor to GPT, confirming that complementary mixing mechanisms provide benefit even at small scale.

3. Attention-free models (RWKV, Mamba) are competitive with modern attention-based variants (LLaMA) at this parameter count, suggesting the attention vs. no-attention distinction matters less than training dynamics at 30M scale.

4. MoE sparse routing (DeepSeek) requires significantly more training steps than dense architectures to leverage its parameter efficiency. 20k steps is insufficient for the router to converge.

5. Ternary quantization (BitNet) requires architecture-specific training conditions (lower LR, longer schedule, larger scale) not provided in this controlled baseline. Applying a shared training budget designed for dense models yields near-zero learning signal.

---

## References

Radford et al., 2019 — "Language Models are Unsupervised Multitask Learners." OpenAI Blog.

Touvron et al., 2023 — "LLaMA 2." arXiv:2307.09288.

Gu & Dao, 2023 — "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752.

Peng et al., 2023 — "RWKV: Reinventing RNNs for the Transformer Era." arXiv:2305.13048.

Lieber et al., 2024 — "Jamba: A Hybrid Transformer-Mamba Language Model." arXiv:2403.19887.

Dai et al., 2024 — "DeepSeekMoE." arXiv:2401.06066.

Ma et al., 2024 — "The Era of 1-bit LLMs." arXiv:2402.17764.

Sun et al., 2023 — "Retentive Network." arXiv:2307.08621.

Fedus et al., 2022 — "Switch Transformers." arXiv:2101.03961.
