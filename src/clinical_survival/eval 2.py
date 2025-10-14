"""Model evaluation utilities with censoring-aware metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

from clinical_survival.utils import ensure_dir


@dataclass
class MetricInterval:
    estimate: float
    ci_lower: float
    ci_upper: float

    def as_dict(self) -> dict[str, float]:
        return {
            "estimate": float(self.estimate),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
        }


@dataclass
class EvaluationResult:
    model_name: str
    metrics: dict[str, MetricInterval]
    reliability: pd.DataFrame
    decision: pd.DataFrame
    calibration_path: Path
    decision_path: Path


def _km_censoring(
    times: NDArray[np.float64], events: NDArray[np.int_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Kaplan-Meier estimate of the censoring distribution G(t)."""

    order = np.argsort(times)
    times_ord = times[order]
    censored = 1 - events[order]
    unique_times = np.unique(times_ord)

    survivors = []
    n_at_risk = len(times_ord)
    prod = 1.0
    for t in unique_times:
        mask = times_ord == t
        censored_at_t = censored[mask].sum()
        prod *= 1.0 - censored_at_t / max(n_at_risk, 1)
        survivors.append(prod)
        n_at_risk -= mask.sum()
    return unique_times, np.asarray(survivors)


def _km_lookup(
    km_times: NDArray[np.float64], km_surv: NDArray[np.float64], query: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.clip(np.interp(query, km_times, km_surv, left=1.0, right=km_surv[-1]), 1e-8, 1.0)


def _ipcw_weights(
    times: NDArray[np.float64],
    events: NDArray[np.int_],
    horizon: float,
    km_times: NDArray[np.float64],
    km_surv: NDArray[np.float64],
) -> NDArray[np.float64]:
    weights = np.zeros_like(times, dtype=float)
    mask_event = (times <= horizon) & (events == 1)
    mask_survive = times > horizon
    if np.any(mask_event):
        weights[mask_event] = 1.0 / _km_lookup(km_times, km_surv, times[mask_event])
    if np.any(mask_survive):
        weights[mask_survive] = 1.0 / _km_lookup(
            km_times, km_surv, np.full(mask_survive.sum(), horizon)
        )
    return weights


def ipcw_reliability_curve(
    y: pd.DataFrame,
    surv_probs: NDArray[np.float64],
    times: Iterable[float],
    *,
    bins: int,
) -> pd.DataFrame:
    """Compute censoring-aware calibration curves."""

    times_arr = np.asarray(times, dtype=float)
    events = y["event"].astype(int).to_numpy()
    obs_times = y["time"].astype(float).to_numpy()
    km_times, km_surv = _km_censoring(obs_times, events)

    records: list[dict[str, Any]] = []
    bin_edges = np.linspace(0, 1, bins + 1)

    for t_idx, horizon in enumerate(times_arr):
        preds = np.clip(surv_probs[:, t_idx], 0.0, 1.0)
        bin_ids = np.digitize(preds, bin_edges, right=False) - 1
        weights = _ipcw_weights(obs_times, events, horizon, km_times, km_surv)
        indicator = (obs_times > horizon).astype(float)
        for b in range(bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue
            w = weights[mask]
            w_sum = w.sum()
            if w_sum <= 0:
                continue
            observed = float(np.dot(w, indicator[mask]) / w_sum)
            expected = float(preds[mask].mean())
            se = np.sqrt(max(observed * (1 - observed), 1e-8) / w_sum)
            records.append(
                {
                    "time": float(horizon),
                    "bin": b,
                    "expected": expected,
                    "observed": observed,
                    "ci_lower": float(max(observed - 1.96 * se, 0.0)),
                    "ci_upper": float(min(observed + 1.96 * se, 1.0)),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def decision_curve_ipcw(
    y: pd.DataFrame,
    surv_probs: NDArray[np.float64],
    times: Iterable[float],
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Compute decision curves using IPCW-adjusted event probabilities."""

    times_arr = np.asarray(times, dtype=float)
    thresholds_arr = np.asarray(list(thresholds), dtype=float)
    events = y["event"].astype(int).to_numpy()
    obs_times = y["time"].astype(float).to_numpy()
    km_times, km_surv = _km_censoring(obs_times, events)
    n = len(obs_times)

    records: list[dict[str, float]] = []
    for t_idx, horizon in enumerate(times_arr):
        risk = 1.0 - np.clip(surv_probs[:, t_idx], 0.0, 1.0)
        weights = _ipcw_weights(obs_times, events, horizon, km_times, km_surv)
        event_weight = ((obs_times <= horizon) & (events == 1)).astype(float) * weights
        non_event_weight = (obs_times > horizon).astype(float) * weights
        prevalence = event_weight.sum() / max(n, 1)

        for thr in thresholds_arr:
            treat_mask = risk >= thr
            tp = event_weight[treat_mask].sum() / max(n, 1)
            fp = non_event_weight[treat_mask].sum() / max(n, 1)
            odds = thr / max(1.0 - thr, 1e-8)
            net_benefit = tp - fp * odds
            treat_all = prevalence - (1.0 - prevalence) * odds
            records.append(
                {
                    "time": float(horizon),
                    "threshold": float(thr),
                    "net_benefit": float(net_benefit),
                    "treat_all": float(treat_all),
                    "treat_none": 0.0,
                }
            )
    return pd.DataFrame.from_records(records)


def plot_reliability(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    plt.figure(figsize=(5.5, 4.0))
    sns.lineplot(data=df, x="expected", y="observed", hue="time", marker="o")
    plt.fill_between([0, 1], [0, 1], alpha=0.1, color="grey")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    plt.xlabel("Predicted survival")
    plt.ylabel("Observed survival")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_decision(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    plt.figure(figsize=(5.5, 4.0))
    sns.lineplot(data=df, x="threshold", y="net_benefit", hue="time", marker="o")
    for time_value, group in df.groupby("time"):
        treat_all = group[["threshold", "treat_all"]].drop_duplicates()
        sns.lineplot(
            data=treat_all,
            x="threshold",
            y="treat_all",
            label=f"treat-all @ {int(time_value)}",
            linestyle=":",
        )
    plt.axhline(0, linestyle="--", color="black", label="treat-none")
    plt.xlabel("Risk threshold")
    plt.ylabel("Net benefit")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def compute_metrics(
    y: pd.DataFrame,
    risk: NDArray[np.float64],
    survival: NDArray[np.float64],
    times: Iterable[float],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, MetricInterval]:
    """Compute point estimates and bootstrap CIs for key metrics."""

    times_arr = np.asarray(times, dtype=float)
    surv_struct = Surv.from_dataframe("event", "time", y)
    metrics: dict[str, MetricInterval] = {}

    cindex = concordance_index_censored(y["event"].astype(bool), y["time"].to_numpy(), -risk)[0]
    metrics["concordance"] = MetricInterval(cindex, cindex, cindex)

    ibs = integrated_brier_score(surv_struct, surv_struct, survival, times_arr)
    metrics["ibs"] = MetricInterval(ibs, ibs, ibs)

    brier_vals = []
    auc_vals = []
    for idx, t in enumerate(times_arr):
        brier_t = brier_score(surv_struct, surv_struct, survival[:, idx], t)[0]
        brier_vals.append(brier_t)
    auc_curve, _ = cumulative_dynamic_auc(surv_struct, surv_struct, risk, times_arr)
    auc_vals = list(auc_curve)

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(y))
    boot_metrics: dict[str, list[float]] = {"concordance": []}
    for t in times_arr:
        boot_metrics[f"brier@{int(t)}"] = []
        boot_metrics[f"auc@{int(t)}"] = []
    boot_metrics["ibs"] = []

    for _ in range(max(bootstrap, 0)):
        sample = rng.choice(idxs, size=len(idxs), replace=True)
        y_boot = y.iloc[sample]
        risk_boot = risk[sample]
        surv_boot = survival[sample]
        surv_struct_boot = Surv.from_dataframe("event", "time", y_boot)
        boot_cindex = concordance_index_censored(
            y_boot["event"].astype(bool),
            y_boot["time"].to_numpy(),
            -risk_boot,
        )[0]
        boot_metrics["concordance"].append(float(boot_cindex))
        boot_metrics["ibs"].append(
            float(integrated_brier_score(surv_struct_boot, surv_struct_boot, surv_boot, times_arr))
        )
        auc_boot, _ = cumulative_dynamic_auc(
            surv_struct_boot, surv_struct_boot, risk_boot, times_arr
        )
        for idx, t in enumerate(times_arr):
            boot_metrics[f"auc@{int(t)}"].append(float(auc_boot[idx]))
            brier_boot = brier_score(surv_struct_boot, surv_struct_boot, surv_boot[:, idx], t)[0]
            boot_metrics[f"brier@{int(t)}"].append(float(brier_boot))

    def _interval(values: list[float], estimate: float) -> MetricInterval:
        if not values:
            return MetricInterval(estimate, estimate, estimate)
        lower, upper = np.percentile(values, [2.5, 97.5])
        return MetricInterval(estimate, float(lower), float(upper))

    metrics["concordance"] = _interval(boot_metrics["concordance"], float(cindex))
    metrics["ibs"] = _interval(boot_metrics["ibs"], float(ibs))
    for idx, t in enumerate(times_arr):
        metrics[f"brier@{int(t)}"] = _interval(
            boot_metrics[f"brier@{int(t)}"],
            float(brier_vals[idx]),
        )
        metrics[f"auc@{int(t)}"] = _interval(
            boot_metrics[f"auc@{int(t)}"],
            float(auc_vals[idx]),
        )

    return metrics


def evaluate_model(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    times: Iterable[float],
    output_dir: Path,
    *,
    label: str,
    thresholds: Iterable[float],
    bins: int,
) -> EvaluationResult:
    """Evaluate a fitted estimator on provided data and persist plots."""

    ensure_dir(output_dir)
    calibration_dir = ensure_dir(output_dir / "calibration")
    decision_dir = ensure_dir(output_dir / "decision_curves")

    risk = estimator.predict_risk(X_test)
    survival = estimator.predict_survival_function(X_test, times)

    reliability = ipcw_reliability_curve(y_test, survival, times, bins=bins)
    decision = decision_curve_ipcw(y_test, survival, times, thresholds)

    calibration_path = plot_reliability(
        reliability.assign(label=label),
        calibration_dir / f"reliability_{label}.png",
    )
    decision_path = plot_decision(
        decision.assign(label=label),
        decision_dir / f"net_benefit_{label}.png",
    )

    metrics = compute_metrics(
        y_test,
        np.asarray(risk, dtype=float),
        np.asarray(survival, dtype=float),
        times,
        bootstrap=0,
        seed=0,
    )

    return EvaluationResult(
        model_name=getattr(estimator, "name", "model"),
        metrics=metrics,
        reliability=reliability.assign(label=label),
        decision=decision.assign(label=label),
        calibration_path=calibration_path,
        decision_path=decision_path,
    )
