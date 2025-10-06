from __future__ import annotations

import pandas as pd

from clinical_survival.io import load_dataset


def test_load_dataset(toy_csv, metadata_path):
    (X_train, y_train), external, meta = load_dataset(toy_csv, metadata_path)
    assert y_train.shape[1] == 2
    assert meta["n_samples"] == len(X_train)
    assert external is None
