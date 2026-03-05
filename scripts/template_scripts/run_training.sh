#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/<<MODEL_CONFIG>>/experiments/exp_001_baseline.yaml}"
echo "Starting training with config: $CFG"
python main.py train --config "$CFG"
