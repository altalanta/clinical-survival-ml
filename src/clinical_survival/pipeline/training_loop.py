from typing import Dict, Any
import pandas as pd
from rich.console import Console
from pathlib import Path

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.tracking import MLflowTracker
from clinical_survival.utils import ensure_dir

from . import trainer, evaluator, explainer, registrar

console = Console()


def run_training_loop(
    X: pd.DataFrame,
    y_surv: pd.DataFrame,
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    grid_config: Dict[str, Any],
    tracker: MLflowTracker,
    outdir: Path,
    **_kwargs,
):
    """
    Runs the training loop for each model specified in the config.
    """
    models_dir = ensure_dir(outdir / "artifacts" / "models")

    for model_name in params_config.models:
        with tracker.start_run(f"train_{model_name}") as nested_run:
            if not nested_run:
                continue

            console.print(f"--- Training model: {model_name} ---")
            model_params = grid_config.get(model_name, {})
            tracker.log_params(model_params)

            # Train
            final_pipeline, oof_preds = trainer.train_model(
                X, y_surv, model_name, params_config, features_config, model_params
            )

            # Evaluate
            evaluator.evaluate_predictions(oof_preds, tracker)

            # Explain
            explain_dir = ensure_dir(outdir / "artifacts" / "explainability" / model_name)
            explainer.generate_explanations(final_pipeline, X, model_name, explain_dir, tracker)

            # Register
            model_path = models_dir / f"{model_name}.joblib"
            registrar.save_and_register_model(final_pipeline, model_name, model_path, tracker)

            console.print(f"✅ Finished training {model_name}.", style="green")
