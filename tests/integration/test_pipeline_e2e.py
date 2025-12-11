import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clinical_survival.cli.main import app
from clinical_survival.config import load_config

runner = CliRunner()

# Use the toy dataset config for the integration test
PARAMS_PATH = "configs/params.yaml"
FEATURES_PATH = "configs/features.yaml"
MODEL_GRID_PATH = "configs/model_grid.yaml"


@pytest.fixture(scope="module")
def pipeline_output_dir() -> Path:
    """A fixture to create a temporary output directory for the pipeline run."""
    temp_dir = Path("tests/temp_output")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    return temp_dir


def test_pipeline_end_to_end(pipeline_output_dir: Path):
    """
    Runs the full 'training run' command and asserts that all expected
    artifacts are generated.
    """
    # Arrange
    outdir = str(pipeline_output_dir)
    result = runner.invoke(
        app,
        [
            "training",
            "run",
            "--params-path",
            PARAMS_PATH,
            "--features-path",
            FEATURES_PATH,
            "--model-grid-path",
            MODEL_GRID_PATH,
            "--outdir",
            outdir,
        ],
        catch_exceptions=False,
    )

    # Assert CLI command was successful
    assert result.exit_code == 0, f"CLI command failed: {result.stdout}"
    assert "✅ Pipeline finished successfully." in result.stdout

    # Assert that key artifact directories were created
    artifacts_dir = pipeline_output_dir / "artifacts"
    assert artifacts_dir.exists()

    models_dir = artifacts_dir / "models"
    explain_dir = artifacts_dir / "explainability"
    cf_dir = artifacts_dir / "counterfactuals"
    metrics_dir = pipeline_output_dir / "metrics"

    assert models_dir.exists()
    assert explain_dir.exists()
    assert cf_dir.exists()
    assert metrics_dir.exists()
    assert (metrics_dir / "leaderboard.csv").exists()
    assert (metrics_dir / "model_comparison.json").exists()

    # Assert that artifacts were created for each model
    params_config, _, _ = load_config(PARAMS_PATH, FEATURES_PATH, MODEL_GRID_PATH)
    for model_name in params_config.models:
        assert (models_dir / f"{model_name}.joblib").exists()
        assert (explain_dir / model_name).exists()
        assert (cf_dir / f"{model_name}_counterfactuals.json").exists()

    # Assert that metrics were created
    assert (metrics_dir / "summary_metrics.csv").exists()
    assert (metrics_dir / "calibration_plot.png").exists()
    assert (metrics_dir / "decision_curve.png").exists()

    # Assert that GE validation results were created
    ge_validations_dir = Path("great_expectations/uncommitted/validations")
    assert ge_validations_dir.exists()
    assert len(list(ge_validations_dir.rglob("*.html"))) > 0, "No GE validation docs found."

    # Check checkpoints directory created by default
    checkpoints_dir = pipeline_output_dir / "checkpoints"
    assert checkpoints_dir.exists()
    assert any(checkpoints_dir.rglob("state.json")), "No checkpoint state found."


