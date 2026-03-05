# ── Model selection ───────────────────────────────────────────────────────────
# Usage: make <task> MODEL=<config_folder>  [EXP=<experiment_name>]
#
# MODEL  : folder name under configs/  (default: miniGPT_config)
# EXP    : experiment yaml name without extension  (default: exp_001_baseline)
#
# Examples:
#   make prep     MODEL=miniGPT_config
#   make train    MODEL=miniGPT_config
#   make train    MODEL=miniGPT_config  EXP=exp_002_bigger_model
#   make evaluate MODEL=miniGPT_config
#   make generate MODEL=miniGPT_config

MODEL ?= miniGPT_config
EXP   ?= exp_001_baseline
CFG    = configs/$(MODEL)/experiments/$(EXP).yaml

.PHONY: prep train evaluate generate test test-core lint format clean clean-outputs clean-data help

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
	@echo "── Testing & Quality ────────────────────────────────────────"
	@echo "  test            Run the full pytest suite"
	@echo "  test-core       Run core model tests only — fast, CPU, no network"
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
	@echo "  make train    MODEL=miniGPT_config  EXP=exp_002_bigger_model"
	@echo "  make generate MODEL=miniGPT_config"
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
