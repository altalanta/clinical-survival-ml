"""Time-dependent calibration utilities using IPCW adjustments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from clinical_survival.utils import ensure_dir


@dataclass(slots=True)
class KaplanMeierEstimate:
    """Container for the censoring distribution estimate."""

    times: np.ndarray
    survival: np.ndarray


def _kaplan_meier(times: np.ndarray, events: np.ndarray) -> KaplanMeierEstimate:
    order = np.argsort(times)
    times = times[order]
    censored = 1 - events[order]
    unique_times = np.unique(times)

    n_at_risk = len(times)
    prod = 1.0
    surv_values = []
    for t in unique_times:
        mask = times == t
        censored_at_t = censored[mask].sum()
        denom = max(n_at_risk, 1)
        prod *= 1.0 - censored_at_t / denom
        surv_values.append(prod)
        n_at_risk -= mask.sum()
    return KaplanMeierEstimate(unique_times, np.asarray(surv_values, dtype=float))


def _km_lookup(km: KaplanMeierEstimate, query: np.ndarray) -> np.ndarray:
    return np.clip(
        np.interp(query, km.times, km.survival, left=1.0, right=km.survival[-1]),
        1e-8,
        1.0,
    )


def _ipcw_weights(
    times: np.ndarray,
    events: np.ndarray,
    horizon: float,
    km: KaplanMeierEstimate,
) -> np.ndarray:
    weights = np.zeros_like(times, dtype=float)
    mask_event = (times <= horizon) & (events == 1)
    mask_survive = times > horizon
    if np.any(mask_event):
        weights[mask_event] = 1.0 / _km_lookup(km, times[mask_event])
    if np.any(mask_survive):
        weights[mask_survive] = 1.0 / _km_lookup(km, np.full(mask_survive.sum(), horizon))
    return weights


def ipcw_reliability_curve(
    y: pd.DataFrame,
    survival: np.ndarray,
    times: Iterable[float],
    *,
    bins: int,
) -> pd.DataFrame:
    """Compute calibration data across evaluation horizons."""

    obs_times = y["time"].to_numpy(dtype=float)
    events = y["event"].astype(int).to_numpy()
    km = _kaplan_meier(obs_times, events)
    records: list[dict[str, float]] = []
    bin_edges = np.linspace(0, 1, bins + 1)

    for idx, horizon in enumerate(np.asarray(list(times), dtype=float)):
        preds = np.clip(survival[:, idx], 0.0, 1.0)
        bin_ids = np.digitize(preds, bin_edges, right=False) - 1
        weights = _ipcw_weights(obs_times, events, float(horizon), km)
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
                    "bin": int(b),
                    "expected": expected,
                    "observed": observed,
                    "ci_lower": float(max(observed - 1.96 * se, 0.0)),
                    "ci_upper": float(min(observed + 1.96 * se, 1.0)),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame.from_records(records)


def calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate calibration bins into horizon-level summaries."""

    summaries = []
    for horizon, group in df.groupby("time"):
        mean_abs_error = float(np.abs(group["expected"] - group["observed"]).mean())
        summaries.append({"time": float(horizon), "mae": mean_abs_error})
    return pd.DataFrame(summaries)


def plot_calibration_curve(df: pd.DataFrame, path: Path) -> Path:
    """Persist calibration plot to disk."""

    ensure_dir(path.parent)
    plt.figure(figsize=(6.0, 4.5))
    sns.lineplot(data=df, x="expected", y="observed", hue="time", marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Ideal")
    plt.xlabel("Predicted survival")
    plt.ylabel("Observed survival")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path
