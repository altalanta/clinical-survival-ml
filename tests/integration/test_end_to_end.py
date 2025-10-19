"""End-to-end integration tests for the clinical survival ML pipeline."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from clinical_survival.cli.main import app
from clinical_survival.config_validation import validate_config_files


class TestEndToEndPipeline:
    """Test the complete pipeline from configuration to model deployment."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create a temporary directory for test results."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config(self, temp_results_dir):
        """Create a minimal test configuration."""
        config = {
            "seed": 42,
            "n_splits": 3,
            "inner_splits": 2,
            "test_split": 0.2,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "scoring": {
                "primary": "concordance",
                "secondary": ["ibs", "brier@365"]
            },
            "calibration": {
                "times_days": [90, 180, 365],
                "bins": 10
            },
            "decision_curve": {
                "times_days": [365],
                "thresholds": [0.05, 0.1, 0.2, 0.3]
            },
            "missing": {
                "strategy": "iterative",
                "max_iter": 10,
                "initial_strategy": "median"
            },
            "models": ["coxph", "rsf"],
            "explain": {
                "shap_samples": 50,
                "pdp_features": ["age", "sofa"]
            },
            "monitoring": {
                "alert_thresholds": {
                    "concordance_drop": 0.05,
                    "brier_increase": 0.02,
                    "feature_drift": 0.1,
                    "concept_drift": 0.15
                },
                "baseline_window_days": 7,
                "max_history_records": 1000
            },
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(temp_results_dir),
                "features": "configs/features.yaml"
            },
            "evaluation": {
                "bootstrap": 50
            },
            "external": {
                "label": "test",
                "group_column": None,
                "train_value": "train",
                "external_value": "external"
            }
        }

        config_path = temp_results_dir / "test_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def test_config_validation(self, test_config):
        """Test that configuration validation works correctly."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test valid configuration
        result = runner.invoke(app, ["validate-config",
                                   "--config", str(test_config),
                                   "--grid", "configs/model_grid.yaml",
                                   "--features", "configs/features.yaml"])
        assert result.exit_code == 0

    def test_data_loading_integration(self, test_config):
        """Test data loading and preprocessing integration."""
        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(app, ["load",
                                   "--data", "data/toy/toy_survival.csv",
                                   "--meta", "data/toy/metadata.yaml"])
        assert result.exit_code == 0

    def test_training_pipeline(self, test_config):
        """Test the complete training pipeline."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run training
        result = runner.invoke(app, ["train",
                                   "--config", str(test_config),
                                   "--grid", "configs/model_grid.yaml"])
        assert result.exit_code == 0

        # Check that artifacts were created
        artifacts_dir = Path(test_config.parent) / "artifacts"
        assert artifacts_dir.exists()

        # Check that models were saved
        models_dir = artifacts_dir / "models"
        assert models_dir.exists()

        # Check that metrics were computed
        metrics_dir = artifacts_dir / "metrics"
        assert metrics_dir.exists()

    def test_evaluation_pipeline(self, test_config):
        """Test the evaluation and reporting pipeline."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First run training to generate artifacts
        train_result = runner.invoke(app, ["train",
                                         "--config", str(test_config),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Run evaluation
        result = runner.invoke(app, ["evaluate", "--config", str(test_config)])
        assert result.exit_code == 0

        # Check that evaluation artifacts exist
        artifacts_dir = Path(test_config.parent) / "artifacts"
        metrics_dir = artifacts_dir / "metrics"
        assert metrics_dir.exists()

        # Check for calibration plots
        calibration_files = list(metrics_dir.glob("*calibration*"))
        assert len(calibration_files) > 0

        # Check for decision curve analysis
        decision_files = list(metrics_dir.glob("*decision*"))
        assert len(decision_files) > 0

    def test_explanation_generation(self, test_config):
        """Test model explanation generation."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First run training to generate models
        train_result = runner.invoke(app, ["train",
                                         "--config", str(test_config),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Run explanation generation
        result = runner.invoke(app, ["explain", "--config", str(test_config)])
        assert result.exit_code == 0

        # Check that explanation artifacts exist
        artifacts_dir = Path(test_config.parent) / "artifacts"
        explain_dir = artifacts_dir / "explain"
        assert explain_dir.exists()

        # Check for SHAP values
        shap_files = list(explain_dir.glob("*shap*"))
        assert len(shap_files) > 0

    def test_report_generation(self, test_config):
        """Test HTML report generation."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First run training and evaluation
        train_result = runner.invoke(app, ["train",
                                         "--config", str(test_config),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        eval_result = runner.invoke(app, ["evaluate", "--config", str(test_config)])
        assert eval_result.exit_code == 0

        # Generate report
        report_path = test_config.parent / "test_report.html"
        result = runner.invoke(app, ["report",
                                   "--config", str(test_config),
                                   "--out", str(report_path)])
        assert result.exit_code == 0
        assert report_path.exists()

    def test_complete_run_command(self, test_config):
        """Test the complete run command (train + report)."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run complete pipeline
        result = runner.invoke(app, ["run",
                                   "--config", str(test_config),
                                   "--grid", "configs/model_grid.yaml"])
        assert result.exit_code == 0

        # Check that all expected outputs exist
        artifacts_dir = Path(test_config.parent) / "artifacts"
        assert artifacts_dir.exists()

        models_dir = artifacts_dir / "models"
        assert models_dir.exists()

        metrics_dir = artifacts_dir / "metrics"
        assert metrics_dir.exists()

        explain_dir = artifacts_dir / "explain"
        assert explain_dir.exists()

        # Check for report
        report_path = test_config.parent / "report.html"
        assert report_path.exists()

    def test_monitoring_integration(self, test_config):
        """Test monitoring functionality with trained models."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train models
        train_result = runner.invoke(app, ["train",
                                         "--config", str(test_config),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Run monitoring
        result = runner.invoke(app, ["monitor",
                                   "--config", str(test_config),
                                   "--data", "data/toy/toy_survival.csv"])
        assert result.exit_code == 0

        # Check that monitoring data was saved
        monitoring_dir = Path(test_config.parent) / "monitoring"
        assert monitoring_dir.exists()

    def test_automl_integration(self, test_config):
        """Test AutoML functionality."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run AutoML with short time limit for testing
        result = runner.invoke(app, ["automl",
                                   "--config", str(test_config),
                                   "--data", "data/toy/toy_survival.csv",
                                   "--meta", "data/toy/metadata.yaml",
                                   "--time-limit", "10",  # Short time for testing
                                   "--model-types", "coxph", "rsf",
                                   "--output-dir", str(test_config.parent / "automl")])
        assert result.exit_code == 0

        # Check that AutoML outputs exist
        automl_dir = test_config.parent / "automl"
        assert automl_dir.exists()

        results_file = automl_dir / "automl_results.json"
        assert results_file.exists()

    def test_counterfactual_integration(self, test_config):
        """Test counterfactual explanation generation."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train a model
        train_result = runner.invoke(app, ["train",
                                         "--config", str(test_config),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Generate counterfactuals
        result = runner.invoke(app, ["counterfactual",
                                   "--config", str(test_config),
                                   "--data", "data/toy/toy_survival.csv",
                                   "--meta", "data/toy/metadata.yaml",
                                   "--model-name", "coxph",
                                   "--target-risk", "0.3",
                                   "--n-counterfactuals", "2",
                                   "--output-dir", str(test_config.parent / "counterfactuals")])
        assert result.exit_code == 0

        # Check that counterfactual outputs exist
        counterfactuals_dir = test_config.parent / "counterfactuals"
        assert counterfactuals_dir.exists()

        results_file = counterfactuals_dir / "counterfactual_results.json"
        assert results_file.exists()

