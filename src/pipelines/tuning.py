"""TuningPipeline: hyperparameter optimisation using Optuna.

Wraps the TrainingPipeline to search over a defined hyperparameter space.
"""

import copy
import logging
from pathlib import Path
import yaml
from typing import Any, Dict

import optuna

from src.models.config import AppConfig
from src.pipelines.base import BasePipeline
from src.pipelines.training import TrainingPipeline

logger = logging.getLogger(__name__)


class TuningPipeline(BasePipeline):
    """Orchestrates hyperparameter tuning using Optuna.
    
    This pipeline creates an Optuna study to optimise validation loss.
    For each trial, it samples hyperparameters, modifies a deep-copy of the
    base AppConfig, and runs a full TrainingPipeline.
    
    The best configuration found is saved to ``outputs/best_config.yaml``.
    """
    
    def configure(self) -> None:
        """Setup logging and output directories."""
        self._n_trials = 10  # default trials if not specified
        self._out_dir = Path(self.config.project.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._best_config_path = self._out_dir / "best_config.yaml"
        # Reduce logging chatter from trials
        optuna.logging.set_verbosity(optuna.logging.INFO)

    def validate(self) -> None:
        """Confirm data is ready (same validation as Trainer)."""
        # We can just instantiate a dummy training pipeline to reuse its validation
        dummy = TrainingPipeline(self.config)
        dummy.validate()

    def run(self) -> None:
        """Run the Optuna study."""
        study_name = f"{self.config.project.name}-tuning"
        
        # We use a memory storage here. For multi-node, this could be a DB.
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
        )
        
        logger.info(f"Starting tuning study: {study_name} with {self._n_trials} trials.")
        study.optimize(self._objective, n_trials=self._n_trials)
        
        logger.info("Tuning complete.")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best validation loss: {study.best_value:.4f}")
        logger.info("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            logger.info(f"  {key}: {value}")
            
        self._save_best_config(study.best_trial.params)

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for a single trial."""
        # 1. Sample hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1.0, log=True)
        
        # 2. Clone the base config and inject hyperparams
        trial_config = copy.deepcopy(self.config)
        trial_config.training.optimizer.learning_rate = lr
        trial_config.training.optimizer.weight_decay = weight_decay
        
        # Set trial-specific output paths to avoid collisions
        trial_out_dir = Path(trial_config.project.output_dir) / f"trial_{trial.number}"
        trial_config.project.output_dir = str(trial_out_dir)
        trial_config.training.checkpoint_path = str(trial_out_dir / "checkpoints")
        
        # 3. Initialise and run the training pipeline
        logger.info(f"--- Starting Trial {trial.number} ---")
        trainer = TrainingPipeline(trial_config)
        try:
            trainer.execute()
            # 4. Return the best validation loss achieved by this trial
            val_loss = trainer._best_val_loss
            logger.info(f"--- Finished Trial {trial.number} (val_loss: {val_loss:.4f}) ---")
            return val_loss
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()

    def _save_best_config(self, best_params: Dict[str, Any]) -> None:
        """Apply the best hyperparameters to the config and save to YAML."""
        best_config = copy.deepcopy(self.config)
        best_config.training.optimizer.learning_rate = best_params["learning_rate"]
        best_config.training.optimizer.weight_decay = best_params["weight_decay"]
        
        import dataclasses
        # Convert dataclass to dict
        config_dict = dataclasses.asdict(best_config)
        
        # PyYAML safe_load doesn't support tuples by default
        # The only tuple in our config is betas in OptimizerConfig
        if "training" in config_dict and "optimizer" in config_dict["training"]:
            if "betas" in config_dict["training"]["optimizer"]:
                config_dict["training"]["optimizer"]["betas"] = list(
                    config_dict["training"]["optimizer"]["betas"]
                )
        
        with open(self._best_config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
            
        logger.info(f"Saved best configuration to {self._best_config_path}")
        logger.info(f"You can now run full training with: python main.py train --config {self._best_config_path}")
