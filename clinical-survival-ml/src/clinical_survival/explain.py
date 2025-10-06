"""Explainability helpers (permutation importance, SHAP, PDP)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay

try:  # pragma: no cover - optional dependency
    import shap
except ImportError:  # pragma: no cover
    shap = None

from sksurv.metrics import concordance_index_censored

from clinical_survival.utils import ensure_dir


def permutation_importance(
    model,
    X: pd.DataFrame,
    y,
    times: Iterable[float],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate permutation importance using drop in C-index."""

    rng = np.random.default_rng(random_state)
    baseline_risk = model.predict_risk(X)
    baseline_score = concordance_index_censored(y["event"], y["time"], -baseline_risk)[0]

    importances: List[Dict[str, float]] = []
    for column in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].values)
            risk = model.predict_risk(shuffled)
            score = concordance_index_censored(y["event"], y["time"], -risk)[0]
            drops.append(baseline_score - score)
        importances.append({"feature": column, "importance": float(np.mean(drops))})
    return pd.DataFrame(importances).sort_values("importance", ascending=False)


def shap_summary(
    model,
    X: pd.DataFrame,
    output_path: Path,
    max_samples: Optional[int] = None,
) -> Optional[Path]:
    """Compute and plot SHAP summary for tree models."""

    if shap is None:  # pragma: no cover
        return None
    if not hasattr(model, "predict_risk"):
        return None
    data = X.iloc[:max_samples] if max_samples else X
    explainer = shap.Explainer(lambda data_matrix: model.predict_risk(pd.DataFrame(data_matrix, columns=data.columns)), data.to_numpy())
    shap_values = explainer(data.to_numpy())
    ensure_dir(output_path.parent)
    shap.plots.beeswarm(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def partial_dependence_plots(
    model,
    X: pd.DataFrame,
    features: Iterable[str],
    output_dir: Path,
) -> List[Path]:
    """Generate PDP plots for selected features."""

    output_paths: List[Path] = []
    ensure_dir(output_dir)
    for feature in features:
        if feature not in X.columns:
            continue
        fig, ax = plt.subplots(figsize=(4, 3))
        PartialDependenceDisplay.from_estimator(
            model,
            X,
            [feature],
            ax=ax,
            kind="average",
        )
        ax.set_title(f"PDP - {feature}")
        path = output_dir / f"pdp_{feature}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        output_paths.append(path)
    return output_paths


def explain_model(
    model,
    X: pd.DataFrame,
    y,
    times: Iterable[float],
    shap_samples: Optional[int],
    pdp_features: Iterable[str],
    out_dir: Path,
) -> Dict[str, List[Path] | Optional[Path]]:
    """Produce explainability artifacts and return file paths."""

    ensure_dir(out_dir)
    permutation_df = permutation_importance(model, X, y, times)
    perm_path = out_dir / "permutation_importance.csv"
    permutation_df.to_csv(perm_path, index=False)

    shap_path = None
    if shap_samples and shap is not None:
        shap_path = shap_summary(model, X, out_dir / "shap_summary.png", max_samples=shap_samples)

    pdp_paths = partial_dependence_plots(model, X, pdp_features, out_dir)
    return {
        "permutation": [perm_path],
        "shap": [shap_path] if shap_path else [],
        "pdp": pdp_paths,
    }
