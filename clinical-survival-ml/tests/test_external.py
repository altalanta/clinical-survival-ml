from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from clinical_survival.cli import app


def test_external_validation_split(tmp_path: Path, project_root: Path) -> None:
    df = pd.read_csv(project_root / "data" / "toy" / "toy_survival.csv")
    midpoint = len(df) // 2
    df["group"] = ["train"] * midpoint + ["external"] * (len(df) - midpoint)
    external_csv = tmp_path / "toy_external.csv"
    df.to_csv(external_csv, index=False)

    config_path = tmp_path / "params.yaml"
    with open(project_root / "configs" / "params.yaml", "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    params["paths"]["data_csv"] = str(external_csv)
    params["paths"]["outdir"] = str(tmp_path / "results")
    params["models"] = ["coxph"]
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(params, handle)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--grid",
            str(project_root / "configs" / "model_grid.yaml"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    metrics_dir = tmp_path / "results" / "artifacts" / "metrics"
    external_summary = metrics_dir / "external_summary.csv"
    assert external_summary.exists()
    external_df = pd.read_csv(external_summary)
    assert set(["model", "concordance", "ibs"]).issubset(external_df.columns)

    cv_df = pd.read_csv(metrics_dir / "cv_coxph.csv")
    expected_train_rows = (df["group"] == "train").sum()
    assert (cv_df["train_size"] + cv_df["test_size"]).iloc[0] == expected_train_rows

    external_label = params.get("external", {}).get("label", "holdout")
    model_dir = tmp_path / "results" / "artifacts" / "models" / external_label
    assert (model_dir / "transformer.joblib").exists()
    assert any(model_dir.glob("*.joblib"))
