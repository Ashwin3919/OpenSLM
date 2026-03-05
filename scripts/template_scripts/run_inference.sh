#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/<<MODEL_CONFIG>>/experiments/exp_001_baseline.yaml}"
PROMPT="${2:-Once upon a time}"

echo "Running inference with config: $CFG"
echo "Prompt: $PROMPT"
python main.py generate --config "$CFG" --prompt "$PROMPT"
