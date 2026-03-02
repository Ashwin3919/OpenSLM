CFG ?= configs/experiments/exp_001_baseline.yaml

.PHONY: prep train evaluate generate test test-core lint format clean clean-outputs clean-data help

help:
	@echo ""
	@echo "Usage: make <target> [CFG=configs/experiments/your_exp.yaml]"
	@echo ""
	@echo "  CFG defaults to configs/experiments/exp_001_baseline.yaml"
	@echo ""
	@echo "── Pipelines ────────────────────────────────────────────────"
	@echo "  prep            Tokenise dataset and write train/val .bin files"
	@echo "  train           Train the model; saves checkpoints to outputs/"
	@echo "  evaluate        Run evaluation loop on the validation set"
	@echo "  generate        Generate text from the latest checkpoint"
	@echo ""
	@echo "── Testing & Quality ────────────────────────────────────────"
	@echo "  test            Run the full pytest suite (28 tests)"
	@echo "  test-core       Run core model tests only — fast, CPU, no network"
	@echo "  lint            Check code style with ruff"
	@echo "  format          Auto-fix formatting with ruff"
	@echo ""
	@echo "── Clean ────────────────────────────────────────────────────"
	@echo "  clean           Remove Python cache, pytest cache, build artefacts"
	@echo "                  Safe to run anytime — does not touch data/ or outputs/"
	@echo "  clean-outputs   Delete outputs/ (checkpoints + metrics)"
	@echo "                  Requires make train to regenerate"
	@echo "  clean-data      Delete data/ (tokenised .bin files)"
	@echo "                  Requires make prep (~10 min) to regenerate"
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

# Remove Python cache, pytest cache, notebook checkpoints, and build artifacts.
# Does NOT touch .venv/, data/, or outputs/.
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

# Remove training outputs (checkpoints, metrics). Requires re-running make train.
clean-outputs:
	rm -rf outputs/
	@echo "Outputs removed."

# Remove tokenised data files. Requires re-running make prep (~10 min).
clean-data:
	rm -rf data/
	@echo "Data removed."
