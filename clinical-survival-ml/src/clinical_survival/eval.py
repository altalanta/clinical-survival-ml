"""Model evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sksurv.metrics import (
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from clinical_survival.utils import ensure_dir


@dataclass
class EvaluationResult:
    model_name: str
    metrics: Dict[str, float]
    reliability: pd.DataFrame
    net_benefit: pd.DataFrame
    calibration_path: Path
    decision_path: Path


def _brier_scores(y_train, y_test, surv_probs: np.ndarray, times: np.ndarray) -> np.ndarray:
    from sksurv.metrics import brier_score

    scores = []
    for t_idx, t in enumerate(times):
        score = brier_score(y_train, y_test, surv_probs[:, t_idx], t)[0]
        scores.append(score)
    return np.asarray(scores)


def evaluate(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train,
    X_test: pd.DataFrame,
    y_test,
    times: Iterable[float],
    output_dir: Path,
    *,
    label: str = "cv",
) -> EvaluationResult:
    """Evaluate a fitted survival model and persist plots."""

    times = np.asarray(list(times))
    ensure_dir(output_dir)
    calibration_dir = ensure_dir(output_dir / "calibration")
    decision_dir = ensure_dir(output_dir / "decision_curves")
    risk = model.predict_risk(X_test)
    cindex = concordance_index_censored(y_test["event"], y_test["time"], -risk)[0]

    surv_probs = model.predict_survival_function(X_test, times)
    ibs = integrated_brier_score(y_train, y_test, surv_probs, times)
    brier_scores = _brier_scores(y_train, y_test, surv_probs, times)

    auc, _ = cumulative_dynamic_auc(y_train, y_test, risk, times)
    reliability_df = reliability_curve(y_test, surv_probs, times)
    net_benefit_df = decision_curve(y_test, surv_probs, times, thresholds=[0.05, 0.1, 0.2, 0.3])

    metrics = {
        "concordance": float(cindex),
        "ibs": float(ibs),
        "label": label,
    }
    for idx, t in enumerate(times):
        metrics[f"brier@{int(t)}"] = float(brier_scores[idx])

    calibration_path = plot_reliability(
        reliability_df,
        calibration_dir / f"reliability_{model_name}_{label}.png",
    )
    decision_path = plot_net_benefit(
        net_benefit_df,
        decision_dir / f"net_benefit_{model_name}_{label}.png",
    )
    reliability_df = reliability_df.assign(label=label)
    net_benefit_df = net_benefit_df.assign(label=label)
    return EvaluationResult(model_name, metrics, reliability_df, net_benefit_df, calibration_path, decision_path)


def reliability_curve(y, surv_probs: np.ndarray, times: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Compute calibration reliability curve via binning predicted survival."""

    records: List[Dict[str, float]] = []
    events = y["event"].astype(int)
    survival_times = y["time"].astype(float)

    for idx, t in enumerate(times):
        preds = surv_probs[:, idx]
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_ids = np.digitize(preds, bin_edges) - 1
        for b in range(bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            observed = np.mean((survival_times[mask] > t) | (events[mask] == 0))
            expected = np.mean(preds[mask])
            records.append(
                {
                    "time": float(t),
                    "bin": b,
                    "expected": float(expected),
                    "observed": float(observed),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(records)


def decision_curve(y, surv_probs: np.ndarray, times: np.ndarray, thresholds: Iterable[float]) -> pd.DataFrame:
    """Calculate net benefit for risk thresholds."""

    times = np.asarray(times)
    events = y["event"].astype(int)
    survival_times = y["time"].astype(float)
    df_rows = []

    for idx, t in enumerate(times):
        risk = 1 - surv_probs[:, idx]
        for thr in thresholds:
            treat = risk >= thr
            treated = treat.mean()
            true_events = ((survival_times <= t) & (events == 1)).astype(int)
            nb = (true_events[treat].sum() / len(y)) - treated * thr
            df_rows.append({"time": float(t), "threshold": float(thr), "net_benefit": float(nb)})
    return pd.DataFrame(df_rows)


def plot_reliability(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=df, x="expected", y="observed", hue="time", marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    plt.xlabel("Predicted survival")
    plt.ylabel("Observed survival")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_net_benefit(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(5, 4))
    sns.lineplot(data=df, x="threshold", y="net_benefit", hue="time", marker="o")
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Threshold")
    plt.ylabel("Net benefit")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
