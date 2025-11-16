from typing import Dict, Any
import pandas as pd
from rich.console import Console

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.tuning import run_tuning

console = Console()


def tune_hyperparameters(
    X: pd.DataFrame,
    y_surv: pd.DataFrame,
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    grid_config: Dict[str, Any],
    **_kwargs,
) -> Dict[str, Any] | None:
    """
    Runs hyperparameter tuning for all models if enabled in the config.
    """
    if not params_config.tuning.enabled:
        console.print("Hyperparameter tuning is disabled. Skipping.", style="yellow")
        return {"best_params": grid_config}

    best_params_all_models = {}
    for model_name in params_config.models:
        if model_name not in grid_config:
            console.print(
                f"No search space defined for '{model_name}' in grid config. Skipping tuning.",
                style="yellow",
            )
            continue

        search_space = grid_config[model_name]
        best_params = run_tuning(
            X=X,
            y_surv=y_surv,
            model_name=model_name,
            search_space=search_space,
            params_config=params_config,
            features_config=features_config,
        )
        best_params_all_models[model_name] = best_params

    return {"best_params": best_params_all_models}
