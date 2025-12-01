"""
Declarative preprocessing pipeline builder with structured logging.
"""

from typing import List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from clinical_survival.config import FeaturesConfig
from clinical_survival.logging_config import get_logger
from clinical_survival.preprocess.components import get_component

# Get module logger
logger = get_logger(__name__)


def build_declarative_preprocessor(features_config: FeaturesConfig) -> Pipeline:
    """
    Builds a scikit-learn preprocessing pipeline from a declarative config.

    Args:
        features_config: The validated features configuration object.

    Returns:
        A scikit-learn Pipeline object.
    """
    logger.info("Building declarative preprocessing pipeline")

    numerical_transformers: List[Tuple] = []
    categorical_transformers: List[Tuple] = []

    for i, step in enumerate(features_config.preprocessing_pipeline):
        component_class = get_component(step.component)
        component = component_class(**step.params)
        step_name = f"{step.component.lower()}_{i}"

        # Heuristic to decide if this is a numerical or categorical transformer
        is_categorical_component = any(
            s in step.component.lower() for s in ["encoder", "imputer_cat"]
        )

        if is_categorical_component:
            columns = step.columns or features_config.categorical_cols
            categorical_transformers.append((step_name, component, columns))
            logger.debug(
                f"Added categorical transformer: {step.component}",
                extra={"step_name": step_name, "columns": columns, "params": step.params},
            )
        else:
            columns = step.columns or features_config.numerical_cols
            numerical_transformers.append((step_name, component, columns))
            logger.debug(
                f"Added numerical transformer: {step.component}",
                extra={"step_name": step_name, "columns": columns, "params": step.params},
            )

    # Create a ColumnTransformer to apply different steps to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", Pipeline(steps=numerical_transformers), features_config.numerical_cols),
            ("categorical", Pipeline(steps=categorical_transformers), features_config.categorical_cols),
        ],
        remainder="passthrough",  # Keep other columns (like binary)
    )

    logger.info(
        "Preprocessing pipeline built successfully",
        extra={
            "n_numerical_steps": len(numerical_transformers),
            "n_categorical_steps": len(categorical_transformers),
            "numerical_cols": features_config.numerical_cols,
            "categorical_cols": features_config.categorical_cols,
        },
    )
    
    return Pipeline(steps=[("preprocessor", preprocessor)])
