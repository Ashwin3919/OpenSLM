# ── Model selection ───────────────────────────────────────────────────────────
# Usage: make <task> MODEL=<config_folder>  [EXP=<experiment_name>]
#
# MODEL  : folder name under configs/  (default: miniGPT_config)
# EXP    : experiment yaml name without extension  (default: exp_001_baseline)
#
# Architecture Zoo — available MODEL values:
#   miniGPT_config      GPT-2 baseline (~29M)           reports/gpt.md
#   llama_config        LLaMA-style SLM (~31M)          reports/llama.md
#   deepseek_moe_config DeepSeek MoE SLM (~48M/~25M)   reports/deepseek_moe.md
#   mamba_config        Mamba SSM SLM (~32M)            reports/mamba.md
#   rwkv_config         RWKV SLM (~33M)                 reports/rwkv.md
#   jamba_config        Jamba Hybrid SLM (~35M)         reports/jamba.md
#   bitnet_config       BitNet 1.58b SLM (~30M)         reports/bitnet.md
#   retnet_config       RetNet SLM (~29M)               reports/retnet.md
#
# Comparison: reports/architecture_zoo.md

MODEL ?= miniGPT_config
EXP   ?= exp_001_baseline
CFG    = configs/$(MODEL)/experiments/$(EXP).yaml

.PHONY: prep train evaluate generate test test-core test-models lint format \
        clean clean-outputs clean-data \
        train-llama train-deepseek-moe train-mamba train-rwkv \
        train-jamba train-bitnet train-retnet help

help:
	@echo ""
	@echo "Usage: make <task> MODEL=<config_folder> [EXP=<experiment>]"
	@echo ""
	@echo "  MODEL  folder under configs/  (default: miniGPT_config)"
	@echo "  EXP    experiment yaml name   (default: exp_001_baseline)"
	@echo ""
	@echo "── Pipelines ────────────────────────────────────────────────"
	@echo "  prep            Tokenise dataset and write train/val .bin files"
	@echo "  train           Train the model; saves checkpoints to outputs/"
	@echo "  evaluate        Run evaluation loop on the validation set"
	@echo "  generate        Generate text from the latest checkpoint"
	@echo ""
	@echo "── Architecture Zoo (one-liners) ────────────────────────────"
	@echo "  train-llama         Train LLaMA-style SLM (~31M)"
	@echo "  train-deepseek-moe  Train DeepSeek MoE SLM (~48M total)"
	@echo "  train-mamba         Train Mamba SSM SLM (~32M)"
	@echo "  train-rwkv          Train RWKV SLM (~33M)"
	@echo "  train-jamba         Train Jamba Hybrid SLM (~35M)"
	@echo "  train-bitnet        Train BitNet 1.58b SLM (~30M)"
	@echo "  train-retnet        Train RetNet SLM (~29M)"
	@echo ""
	@echo "── Testing & Quality ────────────────────────────────────────"
	@echo "  test            Run the full pytest suite"
	@echo "  test-core       Run core model tests only — fast, CPU, no network"
	@echo "  test-models     Run architecture zoo model tests"
	@echo "  lint            Check code style with ruff"
	@echo "  format          Auto-fix formatting with ruff"
	@echo ""
	@echo "── Clean ────────────────────────────────────────────────────"
	@echo "  clean           Remove Python cache, pytest cache, build artefacts"
	@echo "  clean-outputs   Delete outputs/ (checkpoints + metrics)"
	@echo "  clean-data      Delete data/ (tokenised .bin files)"
	@echo ""
	@echo "── Examples ─────────────────────────────────────────────────"
	@echo "  make train    MODEL=miniGPT_config"
	@echo "  make train    MODEL=llama_config"
	@echo "  make train    MODEL=mamba_config  EXP=exp_002_bigger_model"
	@echo "  make generate MODEL=retnet_config"
	@echo ""

prep:
	python main.py prep --config $(CFG)

train:
	python main.py train --config $(CFG)

evaluate:
	python main.py evaluate --config $(CFG)

generate:
	python main.py generate --config $(CFG) --prompt "Once upon a time"

test:
	pytest tests/ -v

test-core:
	pytest tests/test_core/ -v

test-models:
	pytest tests/test_models/ -v

lint:
	ruff check src/ tests/ main.py

format:
	ruff format src/ tests/ main.py

# ── Clean targets ─────────────────────────────────────────────────────────────

clean:
	find . -not -path "./.venv*" -not -path "./.git*" \
		\( -type d -name "__pycache__" \
		-o -type d -name ".pytest_cache" \
		-o -type d -name ".ipynb_checkpoints" \
		-o -type d -name "*.egg-info" \
		-o -type d -name "htmlcov" \
		-o -type d -name "dist" \
		-o -type d -name "build" \
		-o -type f -name "*.pyc" \
		-o -type f -name "*.pyo" \
		-o -type f -name ".coverage" \) \
		-exec rm -rf {} + 2>/dev/null || true
	@echo "Clean."

clean-outputs:
	rm -rf outputs/
	@echo "Outputs removed."

clean-data:
	rm -rf data/
	@echo "Data removed."

# ── Architecture Zoo convenience targets ──────────────────────────────────────
# Each target trains the baseline experiment for that architecture.
# Override EXP to use a different experiment yaml.

train-llama:
	$(MAKE) train MODEL=llama_config EXP=$(EXP)

train-deepseek-moe:
	PYTORCH_ALLOC_CONF=expandable_segments:True $(MAKE) train MODEL=deepseek_moe_config EXP=$(EXP)

train-mamba:
	$(MAKE) train MODEL=mamba_config EXP=$(EXP)

train-rwkv:
	$(MAKE) train MODEL=rwkv_config EXP=$(EXP)

train-jamba:
	$(MAKE) train MODEL=jamba_config EXP=$(EXP)

train-bitnet:
	$(MAKE) train MODEL=bitnet_config EXP=$(EXP)

train-retnet:
	$(MAKE) train MODEL=retnet_config EXP=$(EXP)
