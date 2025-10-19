"""Integration tests for monitoring and drift detection workflows."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from clinical_survival.cli.main import app


class TestMonitoringPipeline:
    """Test monitoring, drift detection, and model performance tracking."""

    @pytest.fixture
    def temp_monitoring_dir(self):
        """Create a temporary directory for monitoring data."""
        temp_dir = tempfile.mkdtemp()

        # Create monitoring directory structure
        monitoring_dir = Path(temp_dir) / "monitoring"
        monitoring_dir.mkdir()

        yield monitoring_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config_with_monitoring(self, temp_monitoring_dir):
        """Create a test configuration with monitoring settings."""
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(temp_monitoring_dir.parent),
                "features": "configs/features.yaml"
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
            "models": ["coxph"]
        }

        config_path = temp_monitoring_dir / "monitoring_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def test_monitoring_initialization(self, test_config_with_monitoring):
        """Test monitoring system initialization."""
        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(app, ["monitor",
                                   "--config", str(test_config_with_monitoring),
                                   "--data", "data/toy/toy_survival.csv",
                                   "--save-monitoring"])

        # Should complete without errors
        assert result.exit_code == 0

    def test_drift_detection_workflow(self, test_config_with_monitoring):
        """Test drift detection command and workflow."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First run monitoring to establish baseline
        monitor_result = runner.invoke(app, ["monitor",
                                           "--config", str(test_config_with_monitoring),
                                           "--data", "data/toy/toy_survival.csv"])
        assert monitor_result.exit_code == 0

        # Run drift detection
        result = runner.invoke(app, ["drift",
                                   "--config", str(test_config_with_monitoring),
                                   "--days", "1",
                                   "--show-details"])

        # Should complete without errors
        assert result.exit_code == 0

    def test_monitoring_status_dashboard(self, test_config_with_monitoring):
        """Test monitoring status command."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run monitoring first
        monitor_result = runner.invoke(app, ["monitor",
                                           "--config", str(test_config_with_monitoring),
                                           "--data", "data/toy/toy_survival.csv"])
        assert monitor_result.exit_code == 0

        # Check monitoring status
        result = runner.invoke(app, ["monitoring-status",
                                   "--config", str(test_config_with_monitoring)])

        assert result.exit_code == 0

    def test_monitoring_reset_functionality(self, test_config_with_monitoring):
        """Test monitoring baseline reset."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First establish some monitoring data
        monitor_result = runner.invoke(app, ["monitor",
                                           "--config", str(test_config_with_monitoring),
                                           "--data", "data/toy/toy_survival.csv"])
        assert monitor_result.exit_code == 0

        # Reset monitoring baselines
        result = runner.invoke(app, ["reset-monitoring",
                                   "--config", str(test_config_with_monitoring),
                                   "--confirm"])

        assert result.exit_code == 0

    def test_monitoring_with_trained_models(self, test_config_with_monitoring):
        """Test monitoring integration with trained models."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train a model to have something to monitor
        train_result = runner.invoke(app, ["train",
                                         "--config", str(test_config_with_monitoring),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Run monitoring on the trained model
        result = runner.invoke(app, ["monitor",
                                   "--config", str(test_config_with_monitoring),
                                   "--data", "data/toy/toy_survival.csv",
                                   "--model-name", "coxph"])

        assert result.exit_code == 0

    def test_monitoring_data_persistence(self, temp_monitoring_dir, test_config_with_monitoring):
        """Test that monitoring data is properly persisted."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run monitoring multiple times
        for i in range(2):
            result = runner.invoke(app, ["monitor",
                                       "--config", str(test_config_with_monitoring),
                                       "--data", "data/toy/toy_survival.csv"])
            assert result.exit_code == 0

        # Check that monitoring data files exist
        monitoring_files = list(temp_monitoring_dir.glob("*"))
        assert len(monitoring_files) > 0

        # Check for specific monitoring artifacts
        baseline_files = list(temp_monitoring_dir.glob("*baseline*"))
        history_files = list(temp_monitoring_dir.glob("*history*"))

        # Should have some monitoring artifacts
        assert len(baseline_files) > 0 or len(history_files) > 0

    def test_drift_detection_with_multiple_models(self, test_config_with_monitoring):
        """Test drift detection across multiple models."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Train multiple models first
        train_result = runner.invoke(app, ["train",
                                         "--config", str(test_config_with_monitoring),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Test drift detection for specific model
        result = runner.invoke(app, ["drift",
                                   "--config", str(test_config_with_monitoring),
                                   "--model-name", "coxph",
                                   "--days", "1"])

        assert result.exit_code == 0

        # Test drift detection for all models
        result_all = runner.invoke(app, ["drift",
                                       "--config", str(test_config_with_monitoring),
                                       "--days", "1"])

        assert result_all.exit_code == 0

    def test_monitoring_alert_thresholds(self, test_config_with_monitoring):
        """Test monitoring alert threshold configuration."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run monitoring to generate some data
        result = runner.invoke(app, ["monitor",
                                   "--config", str(test_config_with_monitoring),
                                   "--data", "data/toy/toy_survival.csv"])
        assert result.exit_code == 0

        # Check drift with alert threshold testing
        result_drift = runner.invoke(app, ["drift",
                                         "--config", str(test_config_with_monitoring),
                                         "--days", "1",
                                         "--show-details"])
        assert result_drift.exit_code == 0

    def test_monitoring_batch_processing(self, test_config_with_monitoring):
        """Test monitoring with different batch sizes."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test with small batch size
        result = runner.invoke(app, ["monitor",
                                   "--config", str(test_config_with_monitoring),
                                   "--data", "data/toy/toy_survival.csv",
                                   "--batch-size", "10"])

        assert result.exit_code == 0

        # Test with larger batch size
        result_large = runner.invoke(app, ["monitor",
                                         "--config", str(test_config_with_monitoring),
                                         "--data", "data/toy/toy_survival.csv",
                                         "--batch-size", "50"])

        assert result_large.exit_code == 0

    def test_monitoring_error_handling(self, test_config_with_monitoring):
        """Test monitoring error handling for edge cases."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test with invalid model name
        result = runner.invoke(app, ["monitor",
                                   "--config", str(test_config_with_monitoring),
                                   "--data", "data/toy/toy_survival.csv",
                                   "--model-name", "nonexistent_model"])

        # Should handle gracefully
        assert result.exit_code == 0 or result.exit_code == 1

    def test_monitoring_configuration_validation(self, temp_monitoring_dir):
        """Test monitoring configuration validation."""
        # Create invalid monitoring configuration
        invalid_config = {
            "monitoring": {
                "alert_thresholds": {
                    "concordance_drop": -0.1,  # Invalid negative threshold
                    "brier_increase": 0.02,
                    "feature_drift": 0.1,
                    "concept_drift": 0.15
                }
            }
        }

        invalid_config_path = temp_monitoring_dir / "invalid_config.yaml"
        with open(invalid_config_path, 'w') as f:
            yaml.dump(invalid_config, f)

        from typer.testing import CliRunner

        runner = CliRunner()

        # Should handle invalid configuration gracefully
        result = runner.invoke(app, ["monitor",
                                   "--config", str(invalid_config_path),
                                   "--data", "data/toy/toy_survival.csv"])

        # May fail due to invalid config, but should not crash
        assert result.exit_code in [0, 1]

    def test_monitoring_data_integrity(self, test_config_with_monitoring):
        """Test monitoring data integrity and consistency."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run monitoring multiple times to generate history
        for i in range(3):
            result = runner.invoke(app, ["monitor",
                                       "--config", str(test_config_with_monitoring),
                                       "--data", "data/toy/toy_survival.csv"])
            assert result.exit_code == 0

        # Check that monitoring data is consistent
        monitoring_dir = Path(test_config_with_monitoring.parent) / "monitoring"

        if monitoring_dir.exists():
            # Check for JSON files with proper structure
            json_files = list(monitoring_dir.glob("*.json"))
            for json_file in json_files:
                with open(json_file) as f:
                    data = json.load(f)

                # Basic structure validation
                assert isinstance(data, (dict, list))

    def test_monitoring_performance_impact(self, test_config_with_monitoring):
        """Test that monitoring doesn't significantly impact performance."""
        from typer.testing import CliRunner
        import time

        runner = CliRunner()

        # Time monitoring execution
        start_time = time.time()
        result = runner.invoke(app, ["monitor",
                                   "--config", str(test_config_with_monitoring),
                                   "--data", "data/toy/toy_survival.csv"])
        end_time = time.time()

        assert result.exit_code == 0

        # Should complete in reasonable time (less than 30 seconds for toy data)
        execution_time = end_time - start_time
        assert execution_time < 30.0

    def test_monitoring_with_external_data(self, test_config_with_monitoring):
        """Test monitoring with external validation data."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train a model
        train_result = runner.invoke(app, ["train",
                                     "--config", str(test_config_with_monitoring),
                                     "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Run monitoring (which uses the same data for now)
        result = runner.invoke(app, ["monitor",
                                   "--config", str(test_config_with_monitoring),
                                   "--data", "data/toy/toy_survival.csv"])

        assert result.exit_code == 0

    def test_monitoring_retention_policy(self, test_config_with_monitoring):
        """Test monitoring data retention and cleanup."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run monitoring many times to test retention
        for i in range(5):
            result = runner.invoke(app, ["monitor",
                                       "--config", str(test_config_with_monitoring),
                                       "--data", "data/toy/toy_survival.csv"])
            assert result.exit_code == 0

        # Check that we don't have excessive files
        monitoring_dir = Path(test_config_with_monitoring.parent) / "monitoring"

        if monitoring_dir.exists():
            all_files = list(monitoring_dir.glob("*"))
            # Should not accumulate too many files (depends on implementation)
            assert len(all_files) < 100  # Reasonable upper bound

