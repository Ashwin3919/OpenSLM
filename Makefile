CFG ?= configs/experiments/exp_001_baseline.yaml

.PHONY: prep train evaluate generate test test-core lint format

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
