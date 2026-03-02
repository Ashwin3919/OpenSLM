import os
from pathlib import Path
import yaml
import pytest

from src.models.config import AppConfig, TrainingConfig, ProjectConfig, OptimizerConfig
from src.pipelines.tuning import TuningPipeline

@pytest.fixture
def dummy_tuning_config(tmp_path):
    # Dummy config with tiny max_iters so training runs instantly
    cfg = AppConfig()
    cfg.project = ProjectConfig(name="test_tuning", output_dir=str(tmp_path / "outputs"))
    cfg.training = TrainingConfig(
        max_iters=2, 
        eval_interval=1, 
        eval_batches=1, 
        checkpoint_path=str(tmp_path / "checkpoints"),
        optimizer=OptimizerConfig(learning_rate=1e-3, weight_decay=0.1)
    )
    # Assume data prep already ran for tests generally; if not we'll mock it
    return cfg

def test_tuning_pipeline(dummy_tuning_config, mocker):
    mocker.patch("src.pipelines.training.TrainingPipeline.validate")
    # Mock execute so we don't actually run training and depend on data existing
    # Just set a dummy _best_val_loss so the objective function gets it
    def fake_execute(self):
        self._best_val_loss = 0.5
    
    mocker.patch("src.pipelines.training.TrainingPipeline.execute", fake_execute)
    
    pipeline = TuningPipeline(dummy_tuning_config)
    pipeline.configure()
    pipeline._n_trials = 2  # Keep it very short for the test
    pipeline.run()
    
    out_dir = Path(dummy_tuning_config.project.output_dir)
    best_config_path = out_dir / "best_config.yaml"
    
    assert best_config_path.exists()
    
    with open(best_config_path, "r") as f:
        best_cfg = yaml.safe_load(f)
        
    assert "learning_rate" in best_cfg["training"]["optimizer"]
    assert "weight_decay" in best_cfg["training"]["optimizer"]
