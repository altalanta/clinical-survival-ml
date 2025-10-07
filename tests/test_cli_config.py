# ruff: noqa: S101
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from clinical_survival.cli import app


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _write_config(base_config: Path, tmp_path: Path) -> Path:
    with base_config.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    params["models"] = ["coxph"]
    params["n_splits"] = 2
    params["inner_splits"] = 2
    params.setdefault("evaluation", {})["bootstrap"] = 0
    params["paths"]["outdir"] = str(tmp_path / "results")
    params["paths"]["external_csv"] = None

    config_path = tmp_path / "params.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(params, handle)
    return config_path


def test_feature_spec_override_is_respected(runner: CliRunner, tmp_path: Path):
    base_config = Path("configs/params.yaml")
    config_path = _write_config(base_config, tmp_path)

    features_path = tmp_path / "features.yaml"
    features_path.write_text(
        "numeric:\n  - age\n  - sofa\ncategorical: []\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--grid",
            "configs/model_grid.yaml",
            "--features-yaml",
            str(features_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    pipeline_path = tmp_path / "results" / "artifacts" / "models" / "coxph" / "pipeline.joblib"
    assert pipeline_path.exists()
    joblib = pytest.importorskip("joblib")
    pipeline = joblib.load(pipeline_path)
    numeric_features = list(pipeline.named_steps["pre"].transformers_[0][2])
    assert numeric_features == ["age", "sofa"]


def test_thresholds_from_config_applied(runner: CliRunner, tmp_path: Path):
    base_config = Path("configs/params.yaml")
    config_path = _write_config(base_config, tmp_path)

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--grid",
            "configs/model_grid.yaml",
            "--thresholds",
            "0.2",
            "0.4",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    decision_path = tmp_path / "results" / "artifacts" / "metrics" / "decision_coxph_holdout.csv"
    assert decision_path.exists()
    df = pd.read_csv(decision_path)
    assert set(df["threshold"].unique()) == {0.2, 0.4}
