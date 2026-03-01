#!/usr/bin/env bash
set -euo pipefail

CFG="${1:-configs/experiments/exp_001_baseline.yaml}"
echo "Running data prep with config: $CFG"
python main.py prep --config "$CFG"
