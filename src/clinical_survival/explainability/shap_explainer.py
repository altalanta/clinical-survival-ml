from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from clinical_survival.utils import ensure_dir

logger = logging.getLogger(__name__)


class ShapExplainer:
    """
    Handles the generation and plotting of SHAP explanations for a trained pipeline.
    """

    def __init__(self, pipeline: Pipeline, X_train: pd.DataFrame):
        self.pipeline = pipeline
        self.model = pipeline.named_steps["est"]
        self.preprocessor = pipeline.named_steps["pre"]
        self.X_train_transformed = self.preprocessor.transform(X_train)

        # Retrieve feature names after transformation
        try:
            self.feature_names = self.preprocessor.get_feature_names_out()
        except Exception:
            logger.warning(
                "Could not get feature names from preprocessor. Using original column names."
            )
            self.feature_names = X_train.columns.tolist()

        self.X_train_transformed_df = pd.DataFrame(
            self.X_train_transformed, columns=self.feature_names
        )

        self.explainer = self._get_explainer()
        self.shap_values = self.explainer(self.X_train_transformed_df)

    def _get_explainer(self) -> shap.Explainer:
        """
        Selects the most appropriate SHAP explainer based on the model type.
        """
        # Use TreeExplainer for tree-based models (much faster)
        if hasattr(self.model, "feature_importances_"):
            return shap.TreeExplainer(self.model, self.X_train_transformed_df)
        # Use PermutationExplainer as a robust fallback for black-box models
        else:
            return shap.PermutationExplainer(self.model.predict, self.X_train_transformed_df)

    def save_summary_plot(self, output_path: Path) -> None:
        """
        Generates and saves a SHAP summary plot (beeswarm style).
        """
        ensure_dir(output_path.parent)
        shap.summary_plot(
            self.shap_values,
            self.X_train_transformed_df,
            show=False,
            plot_type="dot",
        )
        plt.title("Feature Importance (SHAP Summary)")
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()
        logger.info(f"SHAP summary plot saved to {output_path}")

    def save_top_dependence_plots(
        self, output_dir: Path, n_features: int = 5
    ) -> list[Path]:
        """
        Generates and saves SHAP dependence plots for the top N most important features.
        """
        ensure_dir(output_dir)

        # Calculate mean absolute SHAP values to find top features
        mean_abs_shap = pd.Series(
            abs(self.shap_values.values).mean(axis=0), index=self.feature_names
        ).sort_values(ascending=False)

        top_features = mean_abs_shap.head(n_features).index.tolist()
        saved_paths = []

        for feature in top_features:
            output_path = output_dir / f"dependence_plot_{feature}.png"
            shap.dependence_plot(
                feature,
                self.shap_values.values,
                self.X_train_transformed_df,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches="tight", dpi=150)
            plt.close()
            saved_paths.append(output_path)
            logger.info(f"SHAP dependence plot for '{feature}' saved to {output_path}")
        return saved_paths
