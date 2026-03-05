# 🏗️ SLM Architecture Zoo: Build & Benchmark Every Architecture from Scratch

## The Big Idea

You already have a working **GPT-2 style SLM** (~29M params) trained on TinyStories. The project is to build **7 different architectures** at the **same ~30-50M parameter scale**, train them on the **same dataset** with the **same tokenizer and training budget**, and produce an **apples-to-apples comparison** that nobody in the SLM space has done properly.

**Why this gets attention:** Everyone builds one model. Nobody systematically compares architectures at small scale with controlled experiments. This is the kind of work that gets shared on Twitter/X, cited in papers, and bookmarked by researchers.

---

## Your Current Model: The Baseline

```
Architecture:  GPT-2 (Vanilla Transformer Decoder)
Parameters:    ~29.2M (with weight tying)
Config:        6 layers, 6 heads, 384 dim, 128 ctx
Dataset:       TinyStories (GPT2 tokenizer, vocab=50257)
Key features:  Learned positional embeddings, GELU MLP, LayerNorm, weight tying
```

### Parameter Breakdown of Your Current Model
| Component              | Calculation                        | Params    |
|------------------------|------------------------------------|-----------|
| Token Embedding (tied) | 50257 × 384                        | 19.3M     |
| Position Embedding     | 128 × 384                          | 49K       |
| Per-layer Attention    | 4 × (384 × 384) × 6 layers        | 3.5M      |
| Per-layer MLP          | 2 × (384 × 1536) × 6 layers       | 7.1M      |
| LayerNorms             | 2 × 384 × 6 + 384                 | 5K        |
| LM Head (tied)         | (shared with embedding)            | 0         |
| **Total**              |                                    | **~29.2M**|

---

## The 7 Architectures to Build

Here's what to build, in order of implementation difficulty:

---

### 1. 🦙 LLaMA-Style SLM (Difficulty: ⭐⭐)
**Why it matters:** This is what modern LLMs actually use. Shows the effect of RoPE + RMSNorm + SwiGLU + GQA vs vanilla transformer.

```
Changes from your GPT-2:
├── Replace LayerNorm → RMSNorm (no bias, no mean subtraction)
├── Replace learned pos embeddings → RoPE (Rotary Position Embeddings)
├── Replace GELU MLP → SwiGLU FFN (gate mechanism)
├── Replace MHA → GQA (Grouped Query Attention, e.g. 6 heads, 2 KV heads)
└── Pre-norm stays the same

Config to match ~30M params:
  n_layer=6, n_head=6, n_kv_head=2, n_embd=384
  intermediate_size=1024 (SwiGLU has 3 weight matrices so you shrink this)
  block_size=128, vocab_size=50257
```

**Key code changes:**

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape to complex, apply rotation, reshape back
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:xq.shape[-2]]  # slice to seq len
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```

**Expected params:** ~31M (GQA saves params, SwiGLU adds a third matrix but uses smaller hidden dim)

---

### 2. 🔀 DeepSeek-Style MoE SLM (Difficulty: ⭐⭐⭐)
**Why it matters:** Mixture-of-Experts is the hottest scaling paradigm. DeepSeek's innovation is the **shared expert** + **fine-grained experts**. Shows how MoE gets you more total capacity with the same active params.

```
Architecture:
├── Base: LLaMA-style (RMSNorm, RoPE, SwiGLU)
├── Replace dense FFN in layers 2-5 → MoE layer
│   ├── 1 shared expert (always active)
│   ├── 8 routed experts (top-2 selected per token)
│   └── Each expert is a small SwiGLU FFN
├── Layers 0, 1 stay dense (like DeepSeek)
└── Router: linear layer → softmax → top-k selection

Config to match ~50M total params, ~25M active params:
  n_layer=6, n_head=6, n_embd=384
  n_shared_experts=1, n_routed_experts=8, top_k=2
  expert_hidden_dim=256 (smaller per expert)
  dense layers: 0, 1 (first two)
```

**Key code - Router & MoE Layer:**

```python
class TopKRouter(nn.Module):
    def __init__(self, n_embd, n_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, n_experts, bias=False)

    def forward(self, x):
        # x: (B, T, C)
        logits = self.gate(x)  # (B, T, n_experts)
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        return weights, top_k_indices, logits


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shared_expert = SwiGLU(config.n_embd, config.expert_hidden_dim)
        self.experts = nn.ModuleList([
            SwiGLU(config.n_embd, config.expert_hidden_dim)
            for _ in range(config.n_routed_experts)
        ])
        self.router = TopKRouter(config.n_embd, config.n_routed_experts, config.top_k)

    def forward(self, x):
        shared_out = self.shared_expert(x)

        weights, indices, router_logits = self.router(x)
        # Simple loop implementation (fine for small models)
        B, T, C = x.shape
        expert_out = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (indices == i).any(dim=-1)  # (B, T)
            if mask.any():
                weight = (weights * (indices == i).float()).sum(dim=-1)
                expert_out[mask] += weight[mask].unsqueeze(-1) * expert(x[mask])

        return shared_out + expert_out, router_logits  # return logits for load balancing loss
```

**Load balancing loss (critical for MoE training):**
```python
def load_balancing_loss(router_logits, top_k_indices, n_experts):
    # Encourages uniform expert utilization
    probs = F.softmax(router_logits, dim=-1)
    avg_probs = probs.mean(dim=[0, 1])
    freq = torch.zeros(n_experts, device=router_logits.device)
    freq.scatter_add_(0, top_k_indices.reshape(-1),
                      torch.ones_like(top_k_indices.reshape(-1), dtype=torch.float))
    freq = freq / top_k_indices.numel()
    return (avg_probs * freq).sum() * n_experts
```

**Expected params:** ~48M total, ~25M active per token

---

### 3. 🐍 Mamba SSM SLM (Difficulty: ⭐⭐⭐⭐)
**Why it matters:** Mamba is a **completely different paradigm** — no attention at all. O(n) instead of O(n²). This is the most exciting comparison because it tests whether attention is even necessary at small scale.

```
Architecture:
├── No attention mechanism at all
├── Each layer: Norm → Mamba Block → Residual
├── Mamba Block:
│   ├── Linear projection: dim → 2*expand*dim
│   ├── Split into x and z branches
│   ├── x → 1D depthwise conv → SiLU → SSM
│   ├── z → SiLU (gate)
│   ├── y = SSM(x) * gate(z)
│   └── Output projection: expand*dim → dim
└── SSM: Selective State Space with input-dependent Δ, B, C

Config for ~30M params:
  n_layer=12 (more layers since each is cheaper)
  d_model=384
  d_state=16, d_conv=4, expand=2
  vocab_size=50257
```

**Key code - Simplified Mamba Block (pure PyTorch, no custom CUDA):**

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)

        # 1D depthwise convolution
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )

        # SSM parameters (input-dependent — this is the "selective" part)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_inner, bias=True)  # broadcast dt

        # Learnable SSM parameters
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))  # log for stability
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        # Conv
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, T)
        x_branch = self.conv1d(x_branch)[:, :, :T]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        # SSM
        y = self.ssm(x_branch)

        # Gate and output
        z = F.silu(z)
        output = y * z
        return self.out_proj(output)

    def ssm(self, x):
        B, T, d_inner = x.shape
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        D = self.D

        # Input-dependent params
        x_proj = self.x_proj(x)  # (B, T, d_state*2 + 1)
        d_state = A.shape[-1]
        B_param = x_proj[:, :, :d_state]
        C_param = x_proj[:, :, d_state:2*d_state]
        dt = F.softplus(self.dt_proj(x_proj[:, :, -1:]))  # (B, T, d_inner)

        # Discretize: A_bar = exp(dt * A), B_bar = dt * B
        # Selective scan (sequential for simplicity — still fast at small scale)
        h = torch.zeros(B, d_inner, d_state, device=x.device)
        ys = []
        for t in range(T):
            dt_t = dt[:, t]  # (B, d_inner)
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A)  # (B, d_inner, d_state)
            B_bar = dt_t.unsqueeze(-1) * B_param[:, t].unsqueeze(1)

            h = A_bar * h + B_bar * x[:, t].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y = y + D * x  # skip connection
        return y
```

**⚠️ Note on Mamba training speed:** The sequential scan is slow in pure PyTorch. For the blog, this is fine — you're comparing quality, not speed. But mention that real Mamba uses custom CUDA kernels for the parallel associative scan. At ~30M params and 128 context length, the sequential version is totally tractable.

**Expected params:** ~32M

---

### 4. 🔄 RWKV-Style SLM (Difficulty: ⭐⭐⭐)
**Why it matters:** RWKV is the "linear attention RNN" — it processes like an RNN at inference (O(1) per step) but trains like a transformer (parallelizable). It's a third paradigm alongside attention and SSM.

```
Architecture:
├── Each layer has two sub-blocks:
│   ├── Time-Mixing (replaces attention) — linear recurrence with decay
│   └── Channel-Mixing (replaces FFN) — similar gated structure
├── Token shift: mix current token with previous token
├── WKV mechanism: weighted key-value accumulation with decay
└── No position embeddings needed (recurrence is inherently positional)

Config for ~30M params:
  n_layer=8, n_embd=512
  vocab_size=50257
```

**Key code - RWKV Block:**

```python
class RWKV_TimeMix(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        head_dim = n_embd // n_head

        # Learnable mixing coefficients (token shift)
        self.mix_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.mix_v = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        # Per-head decay (learned)
        self.time_decay = nn.Parameter(torch.randn(n_head) - 5)  # starts ~exp(-5)
        self.time_first = nn.Parameter(torch.randn(n_head))  # bonus for current

    def forward(self, x):
        B, T, C = x.shape
        # Token shift: mix with previous timestep
        x_prev = F.pad(x, (0, 0, 1, -1))  # shift right
        k = self.key(x * self.mix_k + x_prev * (1 - self.mix_k))
        v = self.value(x * self.mix_v + x_prev * (1 - self.mix_v))
        r = self.receptance(x * self.mix_r + x_prev * (1 - self.mix_r))
        r = torch.sigmoid(r)  # receptance gate

        # WKV computation (sequential for clarity)
        wkv = self.wkv_sequential(k, v, B, T, C)
        return self.output(r * wkv)

    def wkv_sequential(self, k, v, B, T, C):
        H = self.n_head
        head_dim = C // H
        k = k.view(B, T, H, head_dim)
        v = v.view(B, T, H, head_dim)
        w = -torch.exp(self.time_decay)  # negative decay rates
        u = self.time_first

        out = torch.zeros(B, T, H, head_dim, device=k.device)
        state_a = torch.zeros(B, H, head_dim, device=k.device)
        state_b = torch.zeros(B, H, 1, device=k.device)

        for t in range(T):
            kt = k[:, t]  # (B, H, head_dim)
            vt = v[:, t]
            # Current contribution with bonus
            wkv_t = (state_a + torch.exp(u.view(1, H, 1) + kt) * vt) / \
                     (state_b + torch.exp(u.view(1, H, 1) + kt))
            out[:, t] = wkv_t
            # Update state with decay
            state_a = torch.exp(w.view(1, H, 1)) * state_a + torch.exp(kt) * vt
            state_b = torch.exp(w.view(1, H, 1)) * state_b + torch.exp(kt)

        return out.view(B, T, C)


class RWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * n_embd
        self.mix_k = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.mix_r = nn.Parameter(torch.ones(1, 1, n_embd) * 0.5)
        self.key = nn.Linear(n_embd, hidden_dim, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_dim, n_embd, bias=False)

    def forward(self, x):
        x_prev = F.pad(x, (0, 0, 1, -1))
        k = self.key(x * self.mix_k + x_prev * (1 - self.mix_k))
        r = self.receptance(x * self.mix_r + x_prev * (1 - self.mix_r))
        return torch.sigmoid(r) * self.value(F.relu(k) ** 2)  # squared ReLU
```

**Expected params:** ~33M

---

### 5. 🔀 Hybrid Mamba-Attention SLM (Jamba-style) (Difficulty: ⭐⭐⭐⭐)
**Why it matters:** This tests whether combining SSM + Attention is better than either alone. Jamba (AI21) and Zamba (Zyphra) showed this works at scale. Does it work at ~30M? Nobody knows.

```
Architecture:
├── 8 total layers
├── Layers 0,2,4,6: Mamba SSM blocks (cheap, long-range)
├── Layers 1,3,5,7: Standard attention blocks (expensive, precise)
├── Both use RMSNorm + residual connections
└── Idea: SSM handles broad context, attention handles precise lookups

Config for ~35M params:
  n_layer=8 (4 Mamba + 4 Attention)
  n_embd=384, n_head=6
  mamba_d_state=16, mamba_expand=2
```

**Key code:**
```python
class HybridBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)

        # Alternate: even layers = Mamba, odd layers = Attention
        if layer_idx % 2 == 0:
            self.mixer = MambaBlock(config.n_embd, d_state=16, expand=2)
        else:
            self.mixer = CausalSelfAttention(config)  # reuse your existing class

        self.ffn = SwiGLU(config.n_embd, config.n_embd * 2)

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

**Expected params:** ~35M

---

### 6. 🧮 BitNet-Style 1.58-bit SLM (Difficulty: ⭐⭐⭐)
**Why it matters:** BitNet shows you can constrain weights to {-1, 0, +1} and still get decent performance. At small scale, this would be a provocative result — can a ~30M param model work with ternary weights?

```
Architecture:
├── Base: LLaMA-style
├── Replace all nn.Linear with BitLinear
│   ├── Weights quantized to {-1, 0, +1} during forward pass
│   ├── Activations quantized to 8-bit
│   └── Full precision kept for gradient computation (STE)
└── RMSNorm before each quantization step

Key: The model has "30M params" but each param is 1.58 bits instead of 16/32 bits.
     This means ~6MB model size instead of ~60MB.
```

**Key code:**
```python
class BitLinear(nn.Module):
    """1.58-bit linear layer from BitNet b1.58"""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.rms_norm = RMSNorm(in_features)

    def ste_ternary(self, w):
        """Straight-Through Estimator for {-1, 0, +1} quantization"""
        scale = w.abs().mean()
        w_ternary = torch.clamp(torch.round(w / (scale + 1e-8)), -1, 1)
        return w + (w_ternary - w).detach()  # STE: gradient flows through as if no quant

    def activation_quant(self, x, bits=8):
        """Quantize activations to int8 range"""
        Qn = -(2 ** (bits - 1))
        Qp = 2 ** (bits - 1) - 1
        scale = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_quant = (x * scale).round().clamp(Qn, Qp) / scale
        return x + (x_quant - x).detach()  # STE

    def forward(self, x):
        x = self.rms_norm(x)
        x = self.activation_quant(x)
        w = self.ste_ternary(self.weight)
        return F.linear(x, w, self.bias)
```

**Expected params:** ~30M (but only 6MB on disk with proper encoding)

---

### 7. 🌀 RetNet-Style SLM (Difficulty: ⭐⭐⭐)
**Why it matters:** RetNet (Microsoft) uses **multi-scale exponential decay** as a replacement for softmax attention. It can run in parallel (like a transformer) for training but recurrently for inference. It's the "best of both worlds" candidate.

```
Architecture:
├── Replace softmax attention with retention mechanism
│   ├── Each head gets a different decay rate γ
│   ├── Parallel mode: attention-like matrix with causal decay mask
│   └── No softmax! Just linear + decay
├── SwiGLU FFN
└── RMSNorm

Config for ~30M params:
  n_layer=6, n_head=6, n_embd=384
  gammas per head: evenly spaced in [0.85, 0.999]
```

**Key code - Retention:**
```python
class MultiScaleRetention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.group_norm = nn.GroupNorm(n_head, n_embd)

        # Different decay rate per head
        gammas = 1 - torch.exp(torch.linspace(
            math.log(1 - 0.85), math.log(1 - 0.999), n_head
        ))
        self.register_buffer('gammas', gammas)

    def forward(self, x):
        B, T, C = x.shape
        H = self.n_head
        d = self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # Build causal decay mask D[i,j] = gamma^(i-j) for i >= j, else 0
        decay = self.gammas.view(1, H, 1, 1)
        positions = torch.arange(T, device=x.device)
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        D = (decay ** distance.float().unsqueeze(0).unsqueeze(0))   # (1, H, T, T)
        D = D * (distance >= 0).float().unsqueeze(0).unsqueeze(0)   # causal mask

        # Retention = (Q @ K^T) * D @ V  (no softmax!)
        retention = (q @ k.transpose(-2, -1)) * D
        y = retention @ v  # (B, H, T, d)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.group_norm(y.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(y)
```

**Expected params:** ~29M

---

## Parameter Comparison Table

| #  | Architecture          | Total Params | Active Params | Layers | Dim  | Attention? | Key Innovation              |
|----|----------------------|-------------|---------------|--------|------|------------|-----------------------------|
| 1  | GPT-2 (baseline)     | ~29M        | 29M           | 6      | 384  | Full MHA   | Vanilla transformer         |
| 2  | LLaMA-style          | ~31M        | 31M           | 6      | 384  | GQA        | RoPE + SwiGLU + RMSNorm     |
| 3  | DeepSeek MoE         | ~48M        | 25M           | 6      | 384  | GQA        | Shared + routed experts      |
| 4  | Mamba SSM            | ~32M        | 32M           | 12     | 384  | None       | Selective state spaces       |
| 5  | RWKV                 | ~33M        | 33M           | 8      | 512  | None       | Linear attention RNN         |
| 6  | Jamba Hybrid         | ~35M        | 35M           | 8      | 384  | Partial    | SSM + Attention interleaved  |
| 7  | BitNet 1.58b         | ~30M        | 30M           | 6      | 384  | GQA        | Ternary weights              |
| 8  | RetNet               | ~29M        | 29M           | 6      | 384  | Retention  | Multi-scale decay, no softmax|

---

## Benchmarking Plan

### Metrics to Track (for each model)

**During Training:**
- Training loss curve
- Validation loss curve (same eval set for all!)
- Wall-clock time per 1000 iterations
- Peak GPU memory usage
- Training throughput (tokens/sec)

**After Training (Quality):**
- **Perplexity** on held-out TinyStories test set
- **Generation quality** (same 10 prompts for all models, score coherence/creativity 1-5)
- **Completion accuracy** on simple TinyStories-style cloze tasks

**Efficiency Metrics:**
- Parameters per unit of perplexity improvement
- Inference speed (tokens/sec at batch=1)
- Memory footprint at inference
- For MoE: active params vs total params efficiency

### Benchmark Code Template

```python
import time
import torch

def benchmark_model(model, config, name, test_data, prompts, device='cuda'):
    results = {'name': name, 'total_params': sum(p.numel() for p in model.parameters())}

    # Perplexity on test set
    model.eval()
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for X, Y in test_data:  # iterate through test batches
            logits, loss = model(X.to(device), Y.to(device))
            total_loss += loss.item() * Y.numel()
            total_tokens += Y.numel()
    results['perplexity'] = math.exp(total_loss / total_tokens)

    # Inference speed
    prompt = torch.tensor(enc.encode_ordinary("Once upon a time")).unsqueeze(0).to(device)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model.generate(prompt, max_new_tokens=100, temperature=0.8)
    torch.cuda.synchronize()
    results['tokens_per_sec'] = (10 * 100) / (time.time() - start)

    # Memory
    torch.cuda.reset_peak_memory_stats()
    _ = model.generate(prompt, max_new_tokens=200)
    results['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 1e6

    # Generation samples (same prompts for all models)
    results['samples'] = []
    for p in prompts:
        ctx = torch.tensor(enc.encode_ordinary(p)).unsqueeze(0).to(device)
        out = model.generate(ctx, max_new_tokens=150, temperature=0.8, top_k=40)
        results['samples'].append(enc.decode(out.squeeze().tolist()))

    return results
```

---

## Standardized Test Prompts

Use the same 10 prompts for ALL models to enable fair comparison:

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

---

## Unified Training Framework

To make the comparison fair, use a single training script:

```python
# train_any_slm.py

import argparse

ARCHITECTURES = {
    'gpt2': (GPTConfig, GPT),
    'llama': (LlamaConfig, LlamaSLM),
    'deepseek_moe': (MoEConfig, DeepSeekMoESLM),
    'mamba': (MambaConfig, MambaSLM),
    'rwkv': (RWKVConfig, RWKVSLM),
    'jamba': (JambaConfig, JambaSLM),
    'bitnet': (BitNetConfig, BitNetSLM),
    'retnet': (RetNetConfig, RetNetSLM),
}

parser = argparse.ArgumentParser()
parser.add_argument('--arch', choices=ARCHITECTURES.keys(), required=True)
parser.add_argument('--max_iters', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--block_size', type=int, default=128)
args = parser.parse_args()

# Same dataset, same tokenizer, same training budget for all
ConfigClass, ModelClass = ARCHITECTURES[args.arch]
config = ConfigClass(...)  # architecture-specific defaults
model = ModelClass(config)

print(f"Architecture: {args.arch}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ... identical training loop for all architectures ...
```

