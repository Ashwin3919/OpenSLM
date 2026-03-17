# OpenSLM — Technical Design Document

**Author**: Ashwin Shirke

---

## 1. System Architecture

The codebase is structured in four layers with a strict one-directional dependency rule: dependencies only flow downward. Pipelines call core; core never calls pipelines. Models are isolated plugins — they cannot import from pipelines or from each other.

```
main.py
  └── src/pipelines/        orchestration: train, evaluate, generate, data prep
        ├── src/core/        model primitives, registry, generation loop
        ├── src/infra/       I/O: YAML loading, checkpoints, device setup, logging
        └── src/utils/       stateless helpers: optimizer factory, scheduler, scaler

src/models/<name>/           one self-contained plugin per architecture
configs/<name>_config/       YAML config tree per model
```

**Enforced invariants:**
- `src/core/` contains zero model-specific code.
- Every model lives entirely within `src/models/<name>/` — no model imports bleed outside that package.
- Pipelines instantiate models exclusively via `create_model(key, config)`. No pipeline imports a concrete model class.

---

## 2. Design Decisions and Rationale

### 2.1 Plugin Registry (Dependency Inversion over a Factory Switch)

```python
# src/core/registry.py
MODEL_REGISTRY: dict[str, Type] = {}

def register_model(key: str):
    def decorator(cls):
        MODEL_REGISTRY[key] = cls
        return cls
    return decorator

def create_model(key: str, config) -> BaseSLM:
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{key}' not found. Registered: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](config)
```

A `switch/elif` factory in the training pipeline violates the Open/Closed Principle: adding a new architecture requires modifying core code. The registry inverts this dependency — new models self-register on import, and core code is never touched. Auto-discovery fires when `importlib.import_module("src.models")` executes each `src/models/<name>/__init__.py`. No manifest or explicit list of models is maintained anywhere.

**Trade-off acknowledged:** The registry introduces implicit module loading. A model that fails to import silently disappears from the registry, which makes the failure mode non-obvious. The `create_model` guard (`KeyError` with a list of known keys) surfaces this immediately at training start rather than at model instantiation time deep in the pipeline.

### 2.2 Config Composition via `_includes_`

```yaml
# configs/llama_config/experiments/exp_001_baseline.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/llama_small.yaml"
  - "../training/default.yaml"

# Any key written here overrides its included value.
```

Merging is deep (nested dicts merge key-by-key); later keys override earlier ones. This was chosen over Hydra/OmegaConf after evaluating the actual composition patterns needed:

| Pattern | `_includes_` | Hydra |
|---|---|---|
| Base + override | Yes | Yes |
| Conditional sweeps | No | Yes |
| Type coercion and validation | Manual | Built-in |
| Dependency to audit | 60 lines | External library + release cycle |

For this project's requirements, `_includes_` covers every composition pattern. Hydra's sweep and override features are not needed; adding its dependency creates an external release cycle risk with no upside.

### 2.3 Typed Config Dataclasses (not raw dicts, not Pydantic)

`src/infra/config.load_config()` deserialises the merged YAML dict into `AppConfig`, a tree of `@dataclass` objects. Each model declares its own config class via `config_class` on `BaseSLM`:

```python
class LlamaSLM(BaseSLM):
    config_class = LlamaConfig   # registry uses this to parse the `model:` YAML section
```

Dataclasses were preferred over Pydantic for three reasons:

1. **No runtime metaclass magic.** Pydantic v1 and v2 have incompatible APIs; dataclasses have none.
2. **Direct test construction.** `LlamaConfig(n_embd=64, n_head=2, ...)` works in a unit test without mocking a validation layer.
3. **Single validation site.** `validate_config(config)` raises `ValueError` on constraint violations before any model is built. Pydantic validators scatter validation across field definitions, making constraint interactions harder to reason about.

### 2.4 `generate()` Decoupled from Models

`src/core/generation.py` implements the autoregressive loop. It works for any model satisfying the `(logits, None)` forward contract:

```python
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        logits, _ = model(idx_cropped)   # model must return (logits, loss_or_None)
        # sample from logits
        idx = torch.cat([idx, idx_next], dim=1)
    return idx
```

Keeping generation in core has two properties that matter:

- **One bug surface.** Sampling strategies (top-k, temperature) are tested once, not per model.
- **No accidental override.** A model author cannot break generation by implementing `forward` incorrectly. The only contract is `(logits, None)`.

Override is available but requires an explicit subclass override of `generate()`. This is used only for architectures with genuinely different inference modes: Mamba's recurrent O(1) path, RWKV's stateful WKV recurrence.

### 2.5 `BatchLoader` Encapsulates State (not a Global Function)

```python
class BatchLoader:
    def __init__(self, data_path, block_size, batch_size, device, device_type):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i+self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i+1:i+1+self.block_size].astype(np.int64)) for i in ix])
        return x.to(self.device), y.to(self.device)
```

The notebook-derived alternative — a module-level `get_batch(block_size, batch_size, device, data)` with implicit globals — is un-testable and couples the data loader to global state. A class encapsulates all state explicitly, is injectable in tests with a mock or a small array, and makes the dependency structure visible in the constructor signature.

### 2.6 Single Device Resolution Call

`get_device_context(config.device)` returns `(device, device_type, dtype_str, pt_dtype, autocast_ctx)` and is called once per pipeline at startup. Nothing else in the system detects or queries the device. This eliminates hidden device-detection branches scattered across model and training code — a common source of `device` mismatch bugs in multi-component systems.

### 2.7 Metrics Written to Disk, Not Held in Process Memory

`TrainingPipeline.save_results()` writes `outputs/<model>/metrics.json` after every evaluation interval. Notebooks read from this file; they never call training code inline.

This is a resilience property, not an aesthetic choice. If a notebook kernel is restarted during a multi-hour training run, no metrics are lost. The training pipeline and the analysis notebook are decoupled in both directions — the notebook has no import dependency on the training pipeline.

---

## 3. The `BaseSLM` Contract

Every registered model must satisfy this interface:

```python
class BaseSLM(nn.Module):
    config_class: Type            # required: links the `model:` YAML section to this class

    def forward(
        self,
        idx: torch.Tensor,        # (B, T) int64 token IDs
        targets: Optional[torch.Tensor] = None   # (B, T) int64 targets
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Must return (logits: FloatTensor[B, T, V], loss: scalar) when targets is not None
        # Must return (logits: FloatTensor[B, T, V], None) when targets is None
        ...
```

The contract is deliberately minimal. The framework asks only for logits and an optional scalar loss. MoE auxiliary losses, SSM state management, quantization, positional embedding strategies — all of these are internal to the model. The generation loop and training loop have zero per-architecture branches.

**Why a scalar loss, not raw logits?** Some architectures (DeepSeek MoE) add auxiliary losses (load-balancing) to the cross-entropy loss. If the framework computed loss externally from logits, these auxiliary losses would be invisible to the training loop. Returning loss from `forward` keeps auxiliary loss accounting entirely inside the model where it belongs.

---

## 4. Adding a New Architecture

The full procedure, with no changes required outside the new model's directory:

**Step 1**: Copy the scaffold.
```bash
cp -r src/models/_template src/models/<name>
cp -r configs/config_template configs/<name>_config
```

**Step 2**: Define the config dataclass (`src/models/<name>/config.py`):
```python
@dataclass
class <Name>Config:
    vocab_size: int = 50257
    block_size: int = 128
    n_embd: int = 384
    # ... architecture-specific fields
```

**Step 3**: Implement the model (`src/models/<name>/model.py`):
```python
class <Name>SLM(BaseSLM):
    config_class = <Name>Config

    def __init__(self, config: <Name>Config) -> None:
        super().__init__()
        ...

    def forward(self, idx, targets=None):
        # return (logits, loss) or (logits, None)
        ...
```

**Step 4**: Register (`src/models/<name>/__init__.py`):
```python
from src.core.registry import register_model
from .model import <Name>SLM

register_model("<name>")(<Name>SLM)
```

**Step 5**: Write the experiment YAML:
```yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/<name>_small.yaml"
  - "../training/default.yaml"
```

**Step 6**: Run.
```bash
make train MODEL=<name>_config
```

No other file changes. The registry discovers the new model automatically on `importlib.import_module("src.models")`.

---

## 5. Config Validation Rules

`validate_config(config)` is called immediately after loading and raises `ValueError` before any model or data loader is constructed:

| Check | Condition | Scope |
|---|---|---|
| Head dimension divisibility | `n_embd % n_head != 0` | Central |
| Positive batch size | `batch_size <= 0` | Central |
| Positive iteration count | `max_iters <= 0` | Central |
| Positive gradient accumulation | `gradient_accumulation_steps <= 0` | Central |
| Warmup precedes training end | `warmup_steps >= max_iters` | Central |
| GQA head divisibility | `n_head % n_kv_head != 0` | LLaMA, BitNet, DeepSeek |
| MoE top-k valid | `top_k > n_routed_experts` | DeepSeek |

Central validations go in `validate_config`. Architecture-specific constraints go in the model constructor as `assert` statements. This keeps central validation focused on training-rig consistency; the model is responsible for its own architectural consistency.

---

## 6. Shared Core Primitives

Primitives in `src/core/` are shared across architectures to enforce consistency and eliminate duplication. Each primitive is stateless with respect to architecture — no model-specific logic, no registry coupling.

| Primitive | File | Used by |
|---|---|---|
| `LayerNorm` | `src/core/layers.py` | GPT |
| `MLP` | `src/core/layers.py` | GPT |
| `RMSNorm` | `src/core/normalization.py` | LLaMA, MoE, Mamba, RWKV, Jamba, BitNet, RetNet |
| `precompute_freqs_cis`, `apply_rotary_emb` | `src/core/rope.py` | LLaMA, MoE, BitNet |
| `SwiGLU` | `src/core/ffn.py` | LLaMA, MoE, Jamba, RetNet |
| `MambaBlock` | `src/core/mamba_block.py` | Mamba, Jamba |
| `CausalSelfAttention` | `src/core/attention.py` | GPT, Jamba |
| `generate` | `src/core/generation.py` | All models via `BaseSLM` |

**Why not one shared `Attention` for LLaMA, BitNet, and DeepSeek?** LLaMA uses standard GQA with float projections; BitNet uses GQA with `BitLinear` projections; DeepSeek uses GQA with an additional router. The attention **pattern** (GQA + RoPE) is shared via `src/core/rope.py` utilities, but the attention **module** is architecture-local because the projection layer type differs. Forcing a shared attention module would require per-call injection of the linear type, which adds complexity without measurable benefit at this scale.

---

## 7. Testing Philosophy

Tests are co-located by concern, not by file:

```
tests/
  test_models/          smoke tests: instantiate, run forward, check output shape
  test_core/            unit tests for registry, generation, normalization, RoPE
  test_rwkv_wkv_scan/   regression tests for the parallel WKV scan equivalence
  test_infra/           config loading, checkpoint roundtrip, YAML composition
```

**What is tested:**
- Every registered model: `forward(x, targets)` returns `(logits[B,T,V], loss)` with finite values.
- Every registered model: `forward(x, None)` returns `(logits[B,1,V], None)` at inference.
- Config validation: all six constraint violations raise `ValueError` with an informative message.
- Registry: `create_model` with an unknown key raises `KeyError` listing registered keys.
- `BatchLoader`: batches have the correct shape, device, and dtype.
- WKV parallel scan: 16 test cases covering gradient equivalence, edge cases (`T=1`, strong/weak decay), and full-model smoke tests.

**What is not tested:**
- Training convergence or final loss values. These are validated by the experiment run itself.
- CUDA kernel behaviour. The `mamba-ssm` fast path is tested at the integration level (output equivalence), not at the CUDA level.

---

## 8. Key System Invariants

These invariants must hold for the system to function correctly. They are enforced by the review process, the test suite, and `validate_config`.

1. **No model imports from another model.** Cross-model sharing goes through `src/core/`.
2. **No pipeline imports a concrete model class.** Pipelines use `create_model(key, config)` exclusively.
3. **`validate_config` fires before any model or data loader is constructed.** Config errors are cheap to diagnose.
4. **`get_device_context` is called once per pipeline entry point.** Device detection does not repeat inside model or utility code.
5. **Metrics are written to disk after every evaluation interval.** Notebooks never run training inline.
6. **Every registered model satisfies the `(logits, loss)` forward contract.** The generation loop and training loop have no per-architecture branches.

---

## 9. Checkpoint Format

Every checkpoint is a self-contained restoration point:

```python
{
    "model_state_dict":     ...,   # nn.Module.state_dict()
    "optimizer_state_dict": ...,   # AdamW state
    "scheduler_state_dict": ...,   # cosine LR scheduler state
    "scaler_state_dict":    ...,   # AMP GradScaler state (None if not using AMP)
    "iter":                 int,   # iteration number at checkpoint time
    "val_loss":             float, # validation loss at checkpoint time
    "config":               AppConfig,  # full experiment config, serialised
}
```

Training resumed from a checkpoint reconstructs optimizer and scheduler state exactly, continuing from the stored iteration number. The stored `config` is a sanity reference; the live config (from YAML at resume time) takes precedence.

**BitNet checkpoint note:** BitNet checkpoints store the full-precision `self.weight` tensors from each `BitLinear` layer. The ternary weights at any checkpoint can be recovered by calling `_ternary_weight(layer.weight)` on each layer. No ternary weights are persisted directly.

---

## 10. Notebook Layout

Notebooks are read-only consumers of training outputs. They do not call model code directly.

| Notebook | Input | Purpose |
|---|---|---|
| `01_data_exploration` | `.bin` files | Dataset statistics, token distributions |
| `02_architecture_lab` | `src/core/` imports | Interactive layer prototyping |
| `03_training_monitor` | `outputs/<model>/metrics.json` | Loss curve visualisation |
| `04_generation_demo` | Checkpoint `.pt` file | Load weights and sample text |

`notebooks/00_template/` contains blank versions.

---

*Individual architecture references: `reports/02_gpt.md` through `reports/09rwkv.md`.*
*Experimental results and cross-architecture comparison: `reports/10_compare.md`.*
