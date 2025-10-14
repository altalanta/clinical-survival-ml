# ruff: noqa: S101
from __future__ import annotations

import numpy as np
import pandas as pd

from clinical_survival.eval import ipcw_reliability_curve


def test_ipcw_calibration_matches_manual():
    times = np.array([5.0, 10.0])
    data = pd.DataFrame(
        {
            "time": [4.0, 6.0, 12.0, 15.0],
            "event": [1, 0, 1, 1],
        }
    )
    survival = np.array(
        [
            [0.8, 0.7],
            [0.6, 0.5],
            [0.4, 0.3],
            [0.2, 0.1],
        ]
    )

    curve = ipcw_reliability_curve(data, survival, times, bins=2)
    curve_5 = curve[curve["time"] == 5.0]

    # Manual computation for t=5
    mask_bin0 = curve_5["bin"] == 0
    obs_bin0 = curve_5.loc[mask_bin0, "observed"].iloc[0]
    exp_bin0 = curve_5.loc[mask_bin0, "expected"].iloc[0]
    assert np.isclose(exp_bin0, 0.7, atol=0.05)
    assert 0.0 <= obs_bin0 <= 1.0

    # Observed values must fall within [0,1] and bins must be populated
    assert (curve["observed"] >= 0).all()
    assert (curve["observed"] <= 1).all()
    assert curve["count"].min() > 0
