from __future__ import annotations

import mlflow
from pathlib import Path
from typing import Any, Dict, Optional

class MLflowTracker:
    """A wrapper for MLflow to handle experiment tracking."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}
        self.is_enabled = self.config.get("enabled", False)
        if self.is_enabled:
            mlflow.set_tracking_uri(self.config.get("tracking_uri", "file:./mlruns"))
            mlflow.set_experiment(self.config.get("experiment_name", "clinical-survival-ml"))

    def start_run(self, run_name: str) -> Optional[mlflow.ActiveRun]:
        if self.is_enabled:
            return mlflow.start_run(run_name=run_name)
        return None

    def log_params(self, params: Dict[str, Any]):
        if self.is_enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        if self.is_enabled:
            mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self.is_enabled and Path(local_path).exists():
            mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        if self.is_enabled and Path(local_dir).is_dir():
            mlflow.log_artifacts(local_dir, artifact_path)

    def register_model(self, model_uri: str, name: str) -> Optional[mlflow.pyfunc.PyFuncModel]:
        if self.is_enabled:
            return mlflow.register_model(model_uri, name)
        return None

    @staticmethod
    def end_run():
        if mlflow.active_run():
            mlflow.end_run()


