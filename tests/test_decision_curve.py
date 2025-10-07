# ruff: noqa: S101
from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_survival.eval import decision_curve_ipcw


def _manual_net_benefit(y: pd.DataFrame, surv: np.ndarray, t: float, threshold: float) -> float:
    times = y["time"].to_numpy()
    events = y["event"].to_numpy()
    risk = 1.0 - surv[:, 0]
    treat = risk >= threshold
    n = len(y)

    tp = np.sum(((times <= t) & (events == 1) & treat).astype(float)) / n
    fp = np.sum(((times > t) & treat).astype(float)) / n
    odds = threshold / (1.0 - threshold)
    return tp - fp * odds


def test_net_benefit_ipcw_vs_reference():
    y = pd.DataFrame({"time": [5.0, 7.0, 9.0, 12.0], "event": [1, 0, 1, 1]})
    survival = np.array([[0.6], [0.5], [0.4], [0.3]])
    thresholds = [0.2, 0.4]

    curves = decision_curve_ipcw(y, survival, [8.0], thresholds)
    for thr in thresholds:
        nb = curves.loc[(curves["time"] == 8.0) & (curves["threshold"] == thr), "net_benefit"].iloc[
            0
        ]
        manual = _manual_net_benefit(y, survival, 8.0, thr)
        assert np.isclose(nb, manual, atol=0.05)
