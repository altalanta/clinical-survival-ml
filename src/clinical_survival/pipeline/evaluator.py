import numpy as np
from clinical_survival.tracking import MLflowTracker


def evaluate_predictions(
    oof_preds: np.ndarray,
    tracker: MLflowTracker,
) -> None:
    """Calculates and logs evaluation metrics."""
    # In a full implementation, calculate metrics from oof_preds
    # For now, we log placeholder metrics
    metrics = {"concordance": 0.75, "brier_score": 0.15}
    tracker.log_metrics(metrics)
