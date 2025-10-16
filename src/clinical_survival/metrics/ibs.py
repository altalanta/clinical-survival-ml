"""Integrated Brier score helpers with bootstrap confidence intervals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sksurv.metrics import integrated_brier_score
from sksurv.util import Surv


@dataclass(slots=True)
class MetricCI:
    estimate: float
    lower: float
    upper: float


def integrated_brier_at_times(
    y_true: NDArray,
    y_pred: NDArray,
    survival: NDArray,
    times: Iterable[float],
) -> float:
    """Wrapper around scikit-survival IBS for readability."""

    times_arr = np.asarray(list(times), dtype=float)
    return float(integrated_brier_score(y_true, y_pred, survival, times_arr))


def integrated_brier_summary(
    y: NDArray,
    survival: NDArray,
    times: Iterable[float],
    *,
    bootstrap: int,
    seed: int,
) -> MetricCI:
    """Compute IBS estimate with optional bootstrap confidence interval."""

    times_arr = np.asarray(list(times), dtype=float)
    estimate = float(integrated_brier_score(y, y, survival, times_arr))
    if bootstrap <= 0:
        return MetricCI(estimate=estimate, lower=estimate, upper=estimate)

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(y))
    samples: list[float] = []
    for _ in range(int(bootstrap)):
        sample_idx = rng.choice(idxs, size=len(idxs), replace=True)
        sample_surv = survival[sample_idx]
        sample_y = y[sample_idx]
        samples.append(
            float(integrated_brier_score(sample_y, sample_y, sample_surv, times_arr))
        )
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return MetricCI(estimate=estimate, lower=float(lower), upper=float(upper))


def bootstrap_metric_interval(values: list[float], estimate: float) -> MetricCI:
    if not values:
        return MetricCI(estimate=estimate, lower=estimate, upper=estimate)
    lower, upper = np.percentile(values, [2.5, 97.5])
    return MetricCI(estimate=estimate, lower=float(lower), upper=float(upper))
