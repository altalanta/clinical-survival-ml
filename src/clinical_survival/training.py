from typing import Dict, Any
from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.pipeline.orchestrator import run_pipeline


def train_and_evaluate(
    params_config: ParamsConfig, features_config: FeaturesConfig, grid_config: Dict[str, Any]
) -> None:
    """
    Entry point for the training and evaluation pipeline.
    """
    run_pipeline(params_config, features_config, grid_config)
