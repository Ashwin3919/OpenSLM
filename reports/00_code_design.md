# Technical Design — OpenSLM User Manual

**Author**: Ashwin Shirke

---

## 1. Codebase Layers

The project is split into four independent layers. Dependencies only flow downward — pipelines call core, core never calls pipelines.

```
main.py
  └── src/pipelines/          orchestration (training, tuning, eval, data prep, inference)
        ├── src/core/          framework: model primitives, registry, generation
        ├── src/infra/         I/O: YAML loading, checkpoints, device setup, logging
        └── src/utils/         stateless math helpers: optimizer, scheduler, scaler

src/models/                    SLM plugins (one folder per architecture) and config dataclasses
src/models/                    SLM plugins (one folder per architecture) and config dataclasses
configs/                       YAML files composed per experiment
```

**Rule**: `src/core/` contains zero model-specific code. Every SLM lives entirely inside `src/models/<name>/`.

---

## 2. Config System

### How it works

Experiment YAML files use `_includes_` to compose config fragments. Merging is deep (nested dicts merge key-by-key); later keys override earlier ones.

```yaml
# configs/miniGPT_config/experiments/exp_001_baseline.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/gpt_small.yaml"
  - "../training/default.yaml"

# Any key written here overrides its included value:
# model:
#   n_layer: 8
```

### Config dataclasses

`src/infra/config.load_config()` resolves all includes, then deserialises the merged dict into `AppConfig`. Each section maps to a dataclass:

| YAML section | Dataclass | Location |
|---|---|---|
| `project` | `ProjectConfig` | `src/models/config.py` |
| `logging` | `LoggingConfig` | `src/models/config.py` |
| `device` | `DeviceConfig` | `src/models/config.py` |
| `model_type` + `model` | model-specific (e.g. `GPTConfig`) | `src/models/<name>/config.py` |
| `data` | `DataConfig` | `src/models/config.py` |
| `training` | `TrainingConfig` | `src/models/config.py` |
| `inference` | `InferenceConfig` | `src/models/config.py` |

The `model` section is parsed using the config dataclass registered for `model_type` — so new architectures require no changes to `load_config`.

### Validation

`validate_config(config)` runs immediately after loading and raises `ValueError` on:
- `n_embd % n_head != 0`
- `batch_size <= 0`, `max_iters <= 0`, `gradient_accumulation_steps <= 0`
- `warmup_steps >= max_iters`

---

## 3. Plugin Model Registry

Every SLM is registered under a string key. Pipelines instantiate models by key — they never import a concrete class directly.

```python
# src/core/registry.py
MODEL_REGISTRY: dict[str, Type] = {}

register_model("gpt")(GPT)   # done once in src/models/gpt/__init__.py
create_model("gpt", config)  # used in pipelines
```

Auto-discovery: when `load_config` or `create_model` are called, `importlib.import_module("src.models")` runs, which executes every `src/models/<name>/__init__.py` and triggers their `register_model` calls. Nothing else needs to happen.

---

## 4. Adding a New SLM

Full walkthrough — no other files need to change.

### Step 1 — copy the src template

```bash
cp -r src/models/_template src/models/llama
```

### Step 2 — define the config dataclass

Edit `src/models/llama/config.py`:

```python
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    vocab_size: int = 32000
    block_size: int = 2048
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    dropout: float = 0.0
    bias: bool = False
```

### Step 3 — implement the model

Edit `src/models/llama/model.py`. The only contract:

```python
from src.core.base import BaseSLM
from .config import LlamaConfig

class Llama(BaseSLM):
    config_class = LlamaConfig          # required: registry uses this to parse YAML

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        # ... build layers ...

    def forward(self, idx, targets=None):
        # must return (logits, loss) or (logits, None)
        ...
```

You get `generate()` for free from `BaseSLM`. Override it only for custom sampling (beam search, speculative decoding, etc.).

### Step 4 — register

Edit `src/models/llama/__init__.py`:

```python
from src.core.registry import register_model
from .model import Llama

register_model("llama")(Llama)
```

### Step 5 — copy the config template

```bash
cp -r configs/config_template configs/llama_config
```

Then replace all `<<MODEL_NAME>>` placeholders with `llama` in:
- `configs/llama_config/base.yaml`
- `configs/llama_config/training/default.yaml`
- `configs/llama_config/training/fast_debug.yaml`

Do the same for `<<MODEL_CONFIG>>` in `scripts/template_scripts/` (copy to `scripts/llama_scripts/`).

### Step 6 — create a model preset

```yaml
# configs/llama_config/model/llama_small.yaml
model_type: llama
model:
  vocab_size: 32000
  block_size: 512
  n_layer: 8
  n_head: 8
  n_embd: 512
  dropout: 0.1
  bias: false
```

### Step 7 — create an experiment

```yaml
# configs/llama_config/experiments/exp_001_baseline.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/llama_small.yaml"
  - "../training/default.yaml"
```

### Step 8 — run

```bash
make train MODEL=llama_config
```

---

## 5. Changing the Dataset

### Use a different HuggingFace dataset

Create a new data config inside the model's config folder:

```yaml
# configs/miniGPT_config/data/openwebtext.yaml
data:
  dataset: "Skylion007/openwebtext"
  encoding: "gpt2"
  num_proc: 8
  total_shards: 1024
  output_dir: "data/openwebtext/"
  train_file: "train.bin"
  validation_file: "validation.bin"
```

Include it in your experiment instead of `tinystories.yaml`:

```yaml
_includes_:
  - "../base.yaml"
  - "../data/openwebtext.yaml"
  - "../model/gpt_small.yaml"
  - "../training/default.yaml"
```

Run data prep first, then train:

```bash
make prep  MODEL=miniGPT_config
make train MODEL=miniGPT_config
```

### Use pre-tokenised binary files you already have

Point `output_dir`, `train_file`, and `validation_file` directly at your files and skip `prep`:

```yaml
data:
  dataset: ""             # unused when files exist
  output_dir: "/data/my_corpus/"
  train_file: "train.bin"
  validation_file: "val.bin"
```

The binary format is `uint16` token IDs written with `np.memmap` — one flat array per split.

---

## 6. Running Experiments

### Full pipeline

```bash
make prep     MODEL=miniGPT_config                              # tokenise + write .bin files
make train    MODEL=miniGPT_config                              # train; checkpoints → outputs/miniGPT/checkpoints/
make evaluate MODEL=miniGPT_config                              # eval loss on validation set
make generate MODEL=miniGPT_config                              # generate text

# Or with explicit python commands:
python main.py prep     --config configs/miniGPT_config/experiments/exp_001_baseline.yaml
python main.py train    --config configs/miniGPT_config/experiments/exp_001_baseline.yaml
python main.py evaluate --config configs/miniGPT_config/experiments/exp_001_baseline.yaml
python main.py generate --config configs/miniGPT_config/experiments/exp_001_baseline.yaml \
                        --prompt "Once upon a time"
```

### Make shortcuts

```bash
make prep     MODEL=miniGPT_config
make train    MODEL=miniGPT_config
make train    MODEL=miniGPT_config  EXP=exp_002_bigger_model    # non-default experiment
make test
make lint
```

`MODEL` is the folder name under `configs/`. `EXP` is the experiment YAML name (without `.yaml`) inside that folder's `experiments/` directory.

### Override a single value without editing YAML

Add overrides directly in the experiment file — `_includes_` applies base configs first, then your overrides win:

```yaml
# configs/miniGPT_config/experiments/my_big_run.yaml
_includes_:
  - "../base.yaml"
  - "../data/tinystories.yaml"
  - "../model/gpt_small.yaml"
  - "../training/default.yaml"

training:
  max_iters: 50000
  batch_size: 64
```

Then run:

```bash
make train MODEL=miniGPT_config EXP=my_big_run
```

### Resume an interrupted run

Add `resume_from` under `training` in your experiment YAML (or a local override file):

```yaml
training:
  resume_from: "outputs/miniGPT/checkpoints/checkpoint_iter_10000.pt"
```

Checkpoints store model weights, optimizer state, scheduler state, scaler state, and the iteration number. Training continues from exactly where it stopped.

---

## 7. Key Design Decisions

**`generate()` is decoupled from model classes.** `src/core/generation.py` implements the autoregressive loop — it works for any model that satisfies the `(logits, None)` forward contract. Generation strategies (top-k, temperature) are not part of model code.

**Configs are pure data.** No business logic lives in a dataclass. Validation is centralised in `validate_config`. This makes it safe to construct `AppConfig` objects directly in tests.

**`BatchLoader` is a class, not a function.** Eliminates four implicit globals (`block_size`, `batch_size`, `device`, `device_type`). Testable and injectable.

**Device resolution is a single call.** `get_device_context(config.device)` returns `(device, device_type, dtype_str, pt_dtype, autocast_ctx)`. Called once per pipeline; nothing else detects devices.

**Metrics are written to disk.** `TrainingPipeline.save_results()` writes `outputs/metrics.json` after every run. Notebooks read from that file — they work during training and survive kernel restarts.

---

## 8. Notebooks

Notebooks live under `notebooks/<model_folder>/`. Each has a single responsibility and reads from disk rather than running training inline.

| Notebook | Purpose |
|---|---|
| `01_data_exploration` | Inspect dataset, token distributions, `.bin` sanity checks |
| `02_architecture_lab` | Prototype new layers from `src/core/` interactively |
| `03_training_monitor` | Plot loss curves from `outputs/<model>/metrics.json` |
| `04_generation_demo` | Load a checkpoint and generate text |

`notebooks/00_template/` contains blank versions of all four — copy the whole folder when starting a new model experiment.

---

*For GPT architecture details, parameter reference, and preset configs — see `reports/gpt.md`.*
