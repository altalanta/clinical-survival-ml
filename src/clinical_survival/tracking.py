import mlflow
import os
from typing import Dict, Any, Optional

class MLflowTracker:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_enabled = self.config.get("enabled", False)
        if self.is_enabled:
            mlflow.set_tracking_uri(self.config.get("tracking_uri", "file:./mlruns"))
            mlflow.set_experiment(self.config.get("experiment_name", "default"))

    def start_run(self, run_name: str, nested: bool = True):
        if self.is_enabled:
            return mlflow.start_run(run_name=run_name, nested=nested)
        return None

    def end_run(self):
        if self.is_enabled:
            mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        if self.is_enabled:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float]):
        if self.is_enabled:
            mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        if self.is_enabled and os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)

    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        if self.is_enabled and os.path.isdir(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path)

    def get_artifact_uri(self) -> Optional[str]:
        if self.is_enabled:
            return mlflow.get_artifact_uri()
        return None
