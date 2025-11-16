from pathlib import Path
from sklearn.pipeline import Pipeline
from joblib import dump
import mlflow

from clinical_survival.tracking import MLflowTracker


def save_and_register_model(
    pipeline: Pipeline,
    model_name: str,
    model_path: Path,
    tracker: MLflowTracker,
) -> None:
    """Saves the model artifact, logs it to MLflow, and registers it."""
    dump(pipeline, model_path)

    model_info = mlflow.sklearn.log_model(sk_model=pipeline, artifact_path=model_name)
    tracker.register_model(model_info.model_uri, model_name)

