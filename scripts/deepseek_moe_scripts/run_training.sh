#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/deepseek_moe_config/experiments/exp_001_baseline.yaml}"
echo "Starting training with config: $CFG"
python main.py train --config "$CFG"
