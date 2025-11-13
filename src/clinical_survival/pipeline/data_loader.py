from typing import Tuple, Dict, Any
import pandas as pd
from clinical_survival.config import ParamsConfig, FeaturesConfig
from clinical_survival.io import load_dataset
from clinical_survival.utils import prepare_features, combine_survival_target


def load_and_prepare_data(
    params_config: ParamsConfig,
    features_config: FeaturesConfig,
    **_kwargs,
) -> Dict[str, Any]:
    """Loads the dataset and prepares features and target."""
    (X, y), _, metadata = load_dataset(
        csv_path=params_config.paths.data_csv,
        metadata_path=params_config.paths.metadata,
        time_col=params_config.time_col,
        event_col=params_config.event_col,
    )
    y_surv = combine_survival_target(y[params_config.time_col], y[params_config.event_col])
    X, _ = prepare_features(X, features_config.model_dump())
    return {"X": X, "y_surv": y_surv, "metadata": metadata}
