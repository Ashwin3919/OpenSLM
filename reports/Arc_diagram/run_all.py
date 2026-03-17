#!/usr/bin/env python3
"""Generate all 8 OpenSLM architecture diagrams and save to plots/."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

SCRIPTS = [
    "gpt_arc.py",
    "llama_arc.py",
    "mamba_arc.py",
    "bitnet_arc.py",
    "retnet_arc.py",
    "rwkv_arc.py",
    "deepseek_arc.py",
    "jamba_arc.py",
]

print(f"Generating {len(SCRIPTS)} architecture diagrams ...\n")
for script in SCRIPTS:
    print(f"── {script}")
    subprocess.run(
        [sys.executable, str(HERE / script)],
        cwd=str(HERE),
        check=True,
    )

print(f"\nDone — diagrams saved to:\n  {HERE / 'plots'}/")
