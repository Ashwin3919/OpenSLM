# 📄 PRD v5.0 — SLM Architecture Benchmark & On-Device Personal Assistant
**Author:** You | **Date:** March 2026 | **Status:** Final

***

## 1. Executive Summary

Train and benchmark 8 small language model architectures (~30M parameters each) using a 3-stage continual training pipeline: **TinyStories → Domain Pretraining → Curriculum SFT**. Deploy the winning model as an on-device personal assistant via **FastAPI + OpenClaw + n8n + Google ADK**.

**Total budget: ~$10–15 | Total time: ~33 days**

***

## 2. Progressive Structure Philosophy

> **You only build what the current phase needs. Each phase adds files to the repo — nothing is built ahead of time.**

This is how real engineering projects work. You don't build the serving layer before you have a model. You don't build evaluation charts before you have results. The repo grows phase by phase, each addition motivated by a concrete need.

***

## 3. Phase-by-Phase Repo Evolution

***

### ⚙️ Phase 0 — Project Skeleton
**Day 1–2 | Cost: $0**
**Goal:** Bare minimum to run one architecture through one training step without errors.

```
slm_project/
│
├── core/
│   ├── __init__.py
│   ├── norms.py              # RMSNorm only — needed immediately by GPT-2
│   ├── embeddings.py         # TokenEmbedding + learned positions — GPT-2 uses these
│   ├── activations.py        # GELU only — GPT-2 uses GELU
│   ├── attention.py          # MHA only — GPT-2 uses standard MHA
│   ├── ffn.py                # Standard FFN only
│   └── config.py             # Pydantic ModelConfig + TrainingConfig + load_config()
│
├── architectures/
│   ├── base.py               # BaseLM: generate(), count_params(), get_loss()
│   └── gpt2.py               # GPT-2 only — assembles from core/
│
├── data/
│   ├── tokenizer.py          # BPE tokenizer, vocab 8192
│   └── dataloader.py         # streaming loader for TinyStories + .jsonl
│
├── training/
│   ├── trainer.py            # Trainer class: loop, grad clip, logging, checkpointing
│   ├── scheduler.py          # cosine + warmup
│   └── optimizer.py          # AdamW param groups
│
├── configs/
│   ├── base/
│   │   ├── training.yaml     # all training defaults
│   │   ├── model.yaml        # all model defaults
│   │   └── data.yaml         # dataloader defaults
│   ├── stages/
│   │   └── stage1_tinystories.yaml   # stage1 overrides only
│   ├── archs/
│   │   └── gpt2.yaml         # gpt2 delta only
│   └── experiments/          # empty — auto-populated on first run
│
├── results/
│   └── (empty)
│
├── checkpoints/              # gitignored
│
├── train.py                  # thin CLI → training/trainer.py
├── generate.py               # thin CLI → quick inference
└── requirements.txt
```

**What does NOT exist yet:** `core/scan.py`, `core/linear.py`, `serving/`, `evaluation/`, `scripts/`, `n8n_workflows/`, all arch files except `gpt2.py`

#### Acceptance Criteria
- [ ] `python train.py --arch gpt2 --stage stage1` runs 10 steps, no error
- [ ] `python generate.py --arch gpt2 --checkpoint random` produces 50 tokens
- [ ] `python train.py --arch gpt2 --stage stage1 --dry-run` prints resolved config and exits
- [ ] `torch.manual_seed(42)` set globally, logged per run

***

### 🧪 Phase 1 — miniGPT: Full Pipeline Validation
**Day 3–5 | Cost: $0**
**Goal:** Train GPT-2 end-to-end on TinyStories. Validate every piece works before touching other architectures or spending money.

**No new files added.** You run what Phase 0 built.

```
# What you run:
python train.py --arch gpt2 --stage stage1
# → trains 10K steps on TinyStories
# → saves checkpoints/gpt2_stage1.pt
# → saves results/gpt2_stage1_logs.csv
# → auto-saves configs/experiments/gpt2_stage1.yaml
```

#### Validation Checklist
| Question | Pass Condition |
|---|---|
| Does dataloader stream without OOM? | 5K steps, no crash |
| Does loss decrease? | ~10.5 → <2.8 at 10K steps |
| Is tokenizer consistent? | Same token IDs on re-encode |
| Does generation make sense? | 200 tokens pass human "does this make sense?" test |
| Are logs saving? | `results/gpt2_stage1_logs.csv` exists with step/loss/lr |
| Is config audit saved? | `configs/experiments/gpt2_stage1.yaml` exists |

#### Acceptance Criteria
- [ ] Val loss < 2.8 at 10K steps
- [ ] `checkpoints/gpt2_stage1.pt` saved to Drive
- [ ] `results/gpt2_stage1_logs.csv` committed
- [ ] **Do not proceed to Phase 2 until this passes** — every other arch depends on this pipeline

***

### 🏗️ Phase 2 — Build All 8 Architectures
**Day 6–11 | Cost: $0**
**Goal:** Implement remaining 7 architectures. Each one adds only the new `core/` components it needs.

**New files added — in strict build order:**

#### Step 1: Add LLaMA → adds `RoPE`, `SwiGLU`, `GQA` to `core/`
```
core/
├── embeddings.py         # ADD: RoPE alongside existing learned positions
├── activations.py        # ADD: SwiGLU alongside existing GELU
└── attention.py          # ADD: GQA (parameterized by n_kv_heads) alongside MHA

architectures/
└── llama.py              # NEW: imports RoPE + GQA + SwiGLU from core/

configs/archs/
└── llama.yaml            # NEW: rope, n_kv_heads: 2, swiglu
```

#### Step 2: Add RetNet → adds `retention kernel` to `core/`
```
core/
└── retention.py          # NEW: exponential decay retention — parallel + recurrent modes

architectures/
└── retnet.py             # NEW: imports retention from core/

configs/archs/
└── retnet.yaml           # NEW: gamma decay param
```

#### Step 3: Add BitNet → adds `BitLinear` to `core/`
```
core/
└── linear.py             # NEW: BitLinear (ternary weights, STE gradients)

architectures/
└── bitnet.py             # NEW: ~20 lines — LLaMA + swap all Linear → BitLinear

configs/archs/
└── bitnet.yaml           # NEW: quantization block
```

#### Step 4: Add DeltaNet → adds `delta rule kernel` to `core/`
```
core/
└── delta.py              # NEW: delta rule update: Wt = Wt-1 + β(v - Wt-1 k)kᵀ

architectures/
└── deltanet.py           # NEW: imports delta rule from core/

configs/archs/
└── deltanet.yaml         # NEW: beta schedule params
```

#### Step 5: Add RWKV → adds `WKV kernel` to `core/`
```
core/
└── wkv.py                # NEW: WKV recurrence + token-shift mixing

architectures/
└── rwkv.py               # NEW: imports WKV from core/

configs/archs/
└── rwkv.yaml             # NEW: channel mixing ratio
```

#### Step 6: Add Mamba → adds `SSM + scan` to `core/`
```
core/
└── scan.py               # NEW: parallel associative scan (training) + sequential (inference)

architectures/
└── mamba.py              # NEW: selective SSM, imports scan from core/

configs/archs/
└── mamba.yaml            # NEW: d_state, d_conv, expand
```

#### Step 7: Add Jamba → zero new `core/` files needed
```
architectures/
└── jamba.py              # NEW: ~30 lines — interleaves MambaBlock + LLaMA AttentionBlock

configs/archs/
└── jamba.yaml            # NEW: attn_every (how often to insert attention layer)
```

#### Full `core/` after Phase 2 completes
```
core/
├── __init__.py
├── norms.py              # Phase 0
├── embeddings.py         # Phase 0 + RoPE added in Phase 2
├── activations.py        # Phase 0 + SwiGLU added in Phase 2
├── attention.py          # Phase 0 + GQA added in Phase 2
├── ffn.py                # Phase 0
├── config.py             # Phase 0
├── retention.py          # Phase 2 (RetNet)
├── linear.py             # Phase 2 (BitNet)
├── delta.py              # Phase 2 (DeltaNet)
├── wkv.py                # Phase 2 (RWKV)
├── scan.py               # Phase 2 (Mamba)
└── sampling.py           # Phase 2 — added now, needed for generate() on all 8 archs
```

#### Per-Architecture Gate (must pass before next arch)
```
□ python train.py --arch X --stage stage1 → 500 steps, loss decreasing
□ python generate.py --arch X --checkpoint random → 50 tokens, no error
□ model.count_params() → 28–35M
□ No shape errors at context_length = 64, 128, 256
□ Docstring in arch file explains the key mechanism in 2 sentences
```

***

### 📦 Phase 3 — Generate All Datasets
**Day 9–13 | Parallel with Phase 2 | Cost: ~$5.50**
**Goal:** Generate all data needed for Phases 4–7. Run async while finishing architectures.

**New files added:**
```
scripts/                           # NEW directory
├── generate_domain_corpus.py      # 250M token domain pretraining corpus
├── generate_sft_datasets.py       # 4 SFT datasets, 22K examples total
├── generate_eval_datasets.py      # 7 eval sets, 3,500 examples total
└── generation_cost_log.txt        # running API cost log

data/
└── preprocessing.py               # NEW: MinHash dedup, length filter, quality filter

configs/stages/
└── stage2_domain.yaml             # NEW: stage2 overrides
```

**Data produced:**
```
data/
├── pretraining_domain.jsonl       # 250M tokens, deduplicated
├── sft_summarization.jsonl        # 3,000 examples
├── sft_multiturn.jsonl            # 5,000 examples
├── sft_intent_json.jsonl          # 12,000 examples (2K × 6 intents)
├── sft_cot_planning.jsonl         # 2,000 examples
└── eval/
    ├── eval_grammar.json
    ├── eval_coherence.json
    ├── eval_summarization.json
    ├── eval_instruction.json
    ├── eval_factual_qa.json
    ├── eval_tool_call.json        # ⭐ novel — no public SLM benchmark has this at 30M scale
    └── eval_multiturn.json        # ⭐ novel
```

#### Acceptance Criteria
- [ ] `data/pretraining_domain.jsonl` — ~250M tokens, deduplicated
- [ ] All 4 SFT datasets generated
- [ ] All 7 eval datasets in `data/eval/`
- [ ] Total API cost < $6 in `scripts/generation_cost_log.txt`

***

### 🏋️ Phase 4 — 3-Stage Training: All 8 Models
**Day 14–22 | Cost: ~$5–8 Colab Pro**
**Goal:** Train all 8 architectures through Stage 1 + Stage 2. No new files — you run existing code with new configs.

**New files added:**
```
configs/stages/
└── stage2_domain.yaml             # if not added in Phase 3 already
```

**Training commands:**
```bash
# Stage 1 — TinyStories
python train.py --arch gpt2   --stage stage1
python train.py --arch llama  --stage stage1
python train.py --arch retnet --stage stage1
# ... all 8

# Stage 2 — Domain (auto-resumes from stage1 checkpoint via config)
python train.py --arch gpt2   --stage stage2
python train.py --arch llama  --stage stage2
# ... all 8
```

**Training schedule (Colab Pro, 2 models/day):**
| Day | Models | Notes |
|---|---|---|
| 14 | GPT-2, LLaMA | Simplest — run first |
| 15 | RetNet, BitNet | BitNet slower due to quantization |
| 16 | DeltaNet, RWKV | RWKV needs careful LR |
| 17 | Mamba, Jamba | Most complex — run last |
| 18 | Buffer | Re-run any failures |

**Checkpoints produced:**
```
checkpoints/
├── gpt2_stage1.pt     ├── gpt2_stage2.pt
├── llama_stage1.pt    ├── llama_stage2.pt
├── retnet_stage1.pt   ├── retnet_stage2.pt
├── bitnet_stage1.pt   ├── bitnet_stage2.pt
├── deltanet_stage1.pt ├── deltanet_stage2.pt
├── rwkv_stage1.pt     ├── rwkv_stage2.pt
├── mamba_stage1.pt    ├── mamba_stage2.pt
└── jamba_stage1.pt    └── jamba_stage2.pt
# 16 total — all synced to Google Drive
```

#### Acceptance Criteria
- [ ] 16 checkpoints on Drive
- [ ] 16 CSV log files in `results/`
- [ ] All Stage 2 models reach val loss < 3.5 on domain data
- [ ] Stage 1 → Stage 2 loss gap documented per arch

***

### 📊 Phase 5 — Standard Benchmarks (No Agents)
**Day 23–24 | Cost: $0**
**Goal:** Benchmark all 16 checkpoints. Select the winner. Build the evaluation infrastructure.

**New files added:**
```
evaluation/                        # NEW directory
├── metrics.py                     # ROUGE-L, BERTScore, JSON validity, exact match, perplexity
├── benchmark.py                   # runs all 7 eval sets on a checkpoint
└── charts.py                      # auto-generates all comparison charts

evaluate.py                        # NEW: thin CLI → evaluation/benchmark.py
```

```bash
python evaluate.py --checkpoint checkpoints/{arch}_stage1.pt
python evaluate.py --checkpoint checkpoints/{arch}_stage2.pt
# results appended to results/results.json
```

**Winner selection formula:**
```
Score = 0.35 × tool_call_accuracy
      + 0.25 × multiturn_recall
      + 0.20 × ROUGE_L
      + 0.10 × perplexity_normalized
      + 0.10 × inference_tokens_per_sec_normalized
```

**Auto-generated charts (saved to `results/charts/`):**
- Overlaid loss curves: all 8 models, Stage 1 + Stage 2
- Bar charts: per metric, all 8 architectures side by side
- Radar chart: quality vs. efficiency tradeoff
- Stage 1 vs Stage 2 delta chart — shows what domain training added

#### Acceptance Criteria
- [ ] `results/results.json` — all metrics for all 16 checkpoints
- [ ] All charts saved
- [ ] Winner identified and documented with justification

***

### 🎓 Phase 6 — Curriculum SFT on Winner
**Day 25–26 | Cost: $0**
**Goal:** Fine-tune winner through 4 SFT stages in curriculum order.

**New files added:**
```
training/
└── curriculum.py                  # NEW: multi-stage SFT runner, reads stage list from config

configs/stages/
└── stage3_sft.yaml                # NEW: curriculum list + dropout + early stopping
```

```bash
python train.py --arch {winner} --stage stage3
# curriculum.py runs all 4 stages sequentially:
# summarization → multiturn → intent_json → cot_planning
# saves checkpoint after each stage, saves winner_stage3_final.pt at end
```

**Expected improvement:**
| Stage | Tool-call accuracy | Multi-turn accuracy |
|---|---|---|
| Stage 2 baseline | ~20–30% | ~25–35% |
| After summarization SFT | ~25–35% | ~35–45% |
| After multi-turn SFT | ~30–40% | ~55–70% |
| After intent→JSON SFT | ~65–80% | ~60–75% |
| After CoT planning SFT | ~70–85% | ~65–78% |

#### Acceptance Criteria
- [ ] `checkpoints/winner_stage3_final.pt` saved
- [ ] Tool-call accuracy improved ≥15% over Stage 2 baseline
- [ ] Per-stage eval results logged

***

### 🤖 Phase 7 — Agent Integration & Agentic Benchmarks
**Day 27–29 | Cost: $0**
**Goal:** Deploy winner into full agent stack. Run 50-task agentic benchmark.

**New files added:**
```
serving/                           # NEW directory
├── server.py                      # FastAPI OpenAI-compatible /v1/chat/completions
└── model_loader.py                # loads any arch + checkpoint by name

n8n_workflows/                     # NEW directory
├── reminder.json
├── add_todo.json
├── query_schedule.json
├── summarize_note.json
├── web_search.json
└── set_timer.json

docker-compose.yml                 # NEW: slm-server + openclaw + n8n
Dockerfile                         # NEW: builds serving/server.py container
```

**Full deployment stack:**
```
User Input
    ↓
OpenClaw (semantic memory + tool routing)
    ↓
serving/server.py  ← pure PyTorch, no GGUF needed
  POST /v1/chat/completions (OpenAI-compatible)
  loads winner_stage3_final.pt directly
    ↓
n8n (6 workflows, one per intent)
    ↓
Google ADK (multi-step orchestration)
```

**One command to start everything:**
```bash
docker-compose up
# starts: slm-server (port 8000) + openclaw (8080) + n8n (5678)
```

**50-task agentic benchmark:**
| Category | Tasks | Success Criterion |
|---|---|---|
| Reminders | 10 | Correct time + recurrence, n8n triggered |
| Todos | 10 | All items added to correct list |
| Schedule queries | 10 | Correct calendar entries returned |
| Summarization | 10 | ROUGE-L > 0.4 vs. teacher reference |
| Multi-step planning | 10 | All events scheduled, no conflicts |

**Scoring:** 0 / 0.5 / 1.0 per task
**Target:** SLM >70% vs. GPT-4o-mini ~95%
**Additional:** latency <500ms CPU, RAM <2GB, cost $0 vs $0.30/1K tasks

#### Acceptance Criteria
- [ ] `docker-compose up` starts full stack on clean machine
- [ ] 50-task benchmark complete, saved to `results/agentic_benchmark.json`
- [ ] Latency + RAM measurements logged

***

### 🚀 Phase 8 — Polish & Ship
**Day 30–33 | Cost: $0**

**New files added:**
```
notebooks/
└── reproduce_colab.ipynb          # end-to-end pipeline, one-click Colab

REPRODUCE.md                       # clone → docker-compose up → working assistant
README.md                          # comparison table, architecture diagrams, results
```

**Blog post title:**
> *"I trained 8 SLM architectures from scratch and deployed the winner as a real AI assistant for $12. Here's what I found."*

**Publish to:** HuggingFace blog, dev.to, r/LocalLLaMA, r/MachineLearning, LinkedIn, X

***

## 4. Final Repo at Completion

```
slm_project/
├── core/                 # 13 files — shared components, written once
├── architectures/        # 9 files — thin assembly (base + 8 archs)
├── training/             # 4 files — arch-agnostic training logic
├── data/                 # 3 files — tokenizer, loader, preprocessing
├── evaluation/           # 3 files — metrics, benchmark runner, charts
├── serving/              # 2 files — FastAPI server, model loader
├── scripts/              # 4 files — one-shot data generation
├── configs/              # base/ + stages/ + archs/ + experiments/
├── n8n_workflows/        # 6 workflow JSON files
├── notebooks/            # 1 Colab notebook
├── results/              # results.json + charts/
├── checkpoints/          # 17 .pt files (gitignored)
├── train.py              # thin CLI
├── evaluate.py           # thin CLI
├── generate.py           # thin CLI
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── REPRODUCE.md
└── README.md
```

***

## 5. Complete Timeline

| Days | Phase | What Gets Added | Key Deliverable |
|---|---|---|---|
| 1–2 | Phase 0 | `core/` (partial), `architectures/gpt2.py`, `training/`, `data/`, `configs/` | Skeleton runs 10 steps |
| 3–5 | Phase 1 | Nothing new — run Phase 0 code | GPT-2 val loss < 2.8 |
| 6–11 | Phase 2 | Remaining `core/` components + 7 arch files | All 8 archs pass gate |
| 9–13 | Phase 3 | `scripts/`, `data/preprocessing.py`, all datasets | 250M tokens + 22K SFT examples |
| 14–18 | Phase 4 | Nothing new — run existing code | 16 checkpoints saved |
| 23–24 | Phase 5 | `evaluation/`, `evaluate.py` | Winner selected, all charts |
| 25–26 | Phase 6 | `training/curriculum.py`, `configs/stages/stage3_sft.yaml` | winner_stage3_final.pt |
| 27–29 | Phase 7 | `serving/`, `n8n_workflows/`, `docker-compose.yml`, `Dockerfile` | 50-task benchmark |
| 30–33 | Phase 8 | `notebooks/`, `REPRODUCE.md`, `README.md` | Live on GitHub + HuggingFace + blog |

**Total: ~33 days | ~$10–15**

***

## 6. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Architecture bug wastes compute | High | Phase 1 + 500-iter gate before each arch in Phase 2 |
| Colab timeout loses training | Medium | Checkpoint every 2K steps to Drive |
| Domain corpus too repetitive | Medium | MinHash dedup + 30% TinyStories mix-in |
| GGUF conversion fails | **Eliminated** | `serving/server.py` serves raw PyTorch — no conversion |
| OpenClaw rejects server | Low | `server.py` uses OpenAI-compatible `/v1/chat/completions` |
| SFT overfitting | Medium | Dropout 0.1 + early stopping patience=3 |
| Results not reproducible | Prevented | `seed: 42` in every config, logged per run |

***

## 7. Definition of Done

- [ ] All 8 Stage 1 + Stage 2 checkpoints public on HuggingFace
- [ ] `winner_stage3_final.pt` on HuggingFace with full model card
- [ ] `docker-compose up` tested on a clean machine — works first try
- [ ] 50-task agentic benchmark results public
- [ ] Blog post live on at least 3 platforms
- [ ] `REPRODUCE.md` verified by running it cold
- [ ] At least 3 surprising findings identified, documented, backed by data