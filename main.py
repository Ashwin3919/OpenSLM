import argparse
import sys

import models  # noqa: F401 — triggers auto-discovery of all model plugins

from src.infra.config import load_config
from src.pipelines.data_prep import DataPrepPipeline
from src.pipelines.evaluation import EvaluationPipeline
from src.pipelines.inference import InferencePipeline
from src.pipelines.training import TrainingPipeline

_PIPELINES = {
    "prep": DataPrepPipeline, "train": TrainingPipeline,
    "evaluate": EvaluationPipeline, "generate": InferencePipeline,
}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OpenSLM")
    p.add_argument("command", choices=_PIPELINES)
    p.add_argument("--config", required=True)
    p.add_argument("--prompt", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.prompt:
        cfg.inference.prompt = a.prompt
    pipeline = _PIPELINES[a.command](cfg)
    pipeline.execute()
    if a.command == "generate":
        sys.stdout.write(pipeline.output + "\n")
