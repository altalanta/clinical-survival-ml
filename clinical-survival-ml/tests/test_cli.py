from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from clinical_survival.cli import app


def test_cli_run(tmp_path: Path):
    runner = CliRunner()
    config_path = tmp_path / "params.yaml"
    with open("configs/params.yaml", "r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    params["paths"]["outdir"] = str(tmp_path / "results")
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(params, handle)

    result = runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config_path),
            "--grid",
            "configs/model_grid.yaml",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "results" / "report.html").exists()
