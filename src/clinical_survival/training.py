from typing import Dict, Any
from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.pipeline.orchestrator import run_pipeline


def train_and_evaluate(
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    grid_config: Dict[str, Any],
    *,
    resume: bool = False,
    enable_checkpoints: bool = True,
    run_id: str | None = None,
    enable_performance_monitoring: bool = False,
    performance_report_path: str | None = None,
) -> None:
    """
    Entry point for the training and evaluation pipeline.
    """
    from pathlib import Path

    run_pipeline(
        params_config,
        features_config,
        grid_config,
        enable_checkpoints=enable_checkpoints,
        resume=resume,
        run_id=run_id,
        enable_performance_monitoring=enable_performance_monitoring,
        performance_report_path=Path(performance_report_path) if performance_report_path else None,
    )
