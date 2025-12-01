from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd


def simulate_survival(n: int, seed: int = 42) -> Tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(seed)

    age = rng.normal(65, 10, size=n)
    sofa = rng.normal(6, 2, size=n)
    lactate = rng.lognormal(mean=0.3, sigma=0.4, size=n)
    creatinine = rng.lognormal(mean=0.1, sigma=0.5, size=n)
    heart_rate = rng.normal(90, 12, size=n)

    sex = rng.choice(["Male", "Female"], size=n)
    stage = rng.choice(["I", "II", "III"], size=n)
    icu_type = rng.choice(["MICU", "SICU"], size=n)
    treatment = rng.choice(["Standard", "Experimental"], size=n, p=[0.7, 0.3])

    note = np.array(["synthetic"] * n)

    beta = np.array([0.03, 0.12, 0.15, 0.2, 0.01])
    x_numeric = np.vstack([age, sofa, lactate, creatinine, heart_rate]).T
    lin_pred = x_numeric @ beta
    lin_pred += (sex == "Male") * 0.1
    lin_pred += (stage == "III") * 0.25
    lin_pred += (treatment == "Experimental") * -0.15

    baseline_scale = 365 / np.exp(lin_pred)
    survival_time = rng.weibull(a=1.5, size=n) * baseline_scale
    censor_time = rng.uniform(180, 540, size=n)
    observed_time = np.minimum(survival_time, censor_time)
    event = (survival_time <= censor_time).astype(int)

    df = pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "time": observed_time,
            "event": event,
            "age": age,
            "sofa": sofa,
            "lactate": lactate,
            "creatinine": creatinine,
            "heart_rate": heart_rate,
            "sex": sex,
            "stage": stage,
            "icu_type": icu_type,
            "treatment": treatment,
            "note": note,
        }
    )
    df["time"] = df["time"].clip(lower=1.0)
    return df, {"seed": seed, "n": n}








