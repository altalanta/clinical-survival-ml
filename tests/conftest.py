from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def toy_csv(project_root: Path) -> Path:
    return project_root / "data" / "toy" / "toy_survival.csv"


@pytest.fixture(scope="session")
def metadata_path(project_root: Path) -> Path:
    return project_root / "data" / "toy" / "metadata.yaml"


@pytest.fixture()
def toy_frame(toy_csv: Path) -> pd.DataFrame:
    return pd.read_csv(toy_csv)
