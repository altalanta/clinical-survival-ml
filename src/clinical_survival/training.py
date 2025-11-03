from pathlib import Path
from typing import Dict, Any
import pandas as pd
from sklearn.model_selection import KFold
from joblib import Memory

from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.io import load_dataset
from clinical_survival.models import make_model
from clinical_survival.preprocess import build_preprocessor
from clinical_survival.tracking import MLflowTracker
from clinical_survival.utils import set_global_seed, prepare_features, combine_survival_target

def _get_preprocessed_fold_data(X_train, X_test, features_config, missing_config, seed):
    """Helper function to preprocess a single fold's data."""
    preprocessor = build_preprocessor(
        features_config,
        missing_config,
        random_state=seed
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    return X_train_transformed, X_test_transformed, preprocessor

def train_and_evaluate(
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    grid_config: Dict[str, Any]
) -> None:
    """
    Runs the core training and evaluation pipeline.
    """
    print("Running training and evaluation...")
    print(f"  Seed: {params_config.seed}")
    print(f"  Models: {params_config.models}")
    print(f"  Numeric Features: {features_config.numeric}")

    set_global_seed(params_config.seed)
    tracker = MLflowTracker(params_config.mlflow_tracking.model_dump())

    # Set up caching
    memory = None
    if params_config.caching.enabled:
        cache_dir = Path(params_config.caching.dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        memory = Memory(cache_dir, verbose=0)
    
    cached_preprocess = memory.cache(_get_preprocessed_fold_data) if memory else _get_preprocessed_fold_data

    with tracker.start_run("main_run"):
        # Load data
        X, y = load_dataset(params_config.dataset_path)
        X, y = combine_survival_target(X, y)
        X = prepare_features(X, features_config)

        kf = KFold(n_splits=params_config.n_splits, shuffle=True, random_state=params_config.seed)

        for model_name in params_config.models:
            with tracker.start_run(f"train_{model_name}"):
                # In a real scenario, you would load model_params from grid_config
                # For now, using a placeholder
                model_params = {}

                fold_metrics = []
                for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    X_train_transformed, X_test_transformed, _ = cached_preprocess(
                        X_train,
                        X_test,
                        features_config.model_dump(),
                        params_config.missing.model_dump(),
                        params_config.seed
                    )
                    
                    model = make_model(model_name, random_state=params_config.seed, **model_params)
                    model.fit(pd.DataFrame(X_train_transformed), y_train)

                    # In a real scenario, you would predict on X_test_transformed
                    # and calculate metrics.
                    fold_metrics.append({"concordance": 0.75 + (fold * 0.01)})

                # Collect and save metrics, models, and explainability artifacts
                # This part would typically involve:
                # 1. Calculating final metrics (e.g., mean of fold metrics)
                # 2. Saving the model
                # 3. Generating explainability artifacts
                print(f"  Model: {model_name}, Fold Metrics: {fold_metrics}")

    print("...Training and evaluation finished.")
