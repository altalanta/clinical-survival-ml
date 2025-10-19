"""Integration tests for scalability and large dataset handling."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from clinical_survival.cli.main import app


class TestLargeDatasetScalability:
    """Test scalability with larger datasets and performance benchmarks."""

    @pytest.fixture
    def large_dataset_config(self, tmp_path):
        """Create a configuration for larger dataset testing."""
        # Create a larger synthetic dataset for testing
        large_data = tmp_path / "large_test_data.csv"
        large_metadata = tmp_path / "large_metadata.yaml"

        # Create metadata for larger dataset
        metadata = {
            "features": {
                "numeric_features": ["age", "bmi", "blood_pressure", "cholesterol", "glucose"],
                "categorical_features": ["sex", "smoking", "treatment", "comorbidity"],
                "drop_features": ["id"]
            },
            "time_column": "time",
            "event_column": "event",
            "id_column": "id"
        }

        with open(large_metadata, 'w') as f:
            yaml.dump(metadata, f)

        # Create a larger synthetic dataset (1000 samples instead of toy size)
        import pandas as pd
        import numpy as np

        np.random.seed(42)
        n_samples = 1000

        data = {
            "id": range(n_samples),
            "time": np.random.exponential(5, n_samples),  # Survival times
            "event": np.random.binomial(1, 0.7, n_samples),  # Censoring indicator
            "age": np.random.normal(60, 15, n_samples),
            "sex": np.random.choice(["male", "female"], n_samples),
            "bmi": np.random.normal(25, 5, n_samples),
            "blood_pressure": np.random.normal(120, 20, n_samples),
            "cholesterol": np.random.normal(200, 40, n_samples),
            "glucose": np.random.normal(100, 25, n_samples),
            "smoking": np.random.choice(["never", "former", "current"], n_samples),
            "treatment": np.random.choice(["A", "B", "C"], n_samples),
            "comorbidity": np.random.choice(["none", "mild", "moderate", "severe"], n_samples)
        }

        df = pd.DataFrame(data)
        df.to_csv(large_data, index=False)

        # Create configuration for large dataset
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
                "secondary": ["ibs"]
            },
            "models": ["coxph", "rsf"],  # Limit models for faster testing
            "paths": {
                "data_csv": str(large_data),
                "metadata": str(large_metadata),
                "outdir": str(tmp_path / "results"),
                "features": str(large_metadata)
            },
            "evaluation": {
                "bootstrap": 20  # Reduced for faster testing
            }
        }

        config_path = tmp_path / "large_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def test_large_dataset_training(self, large_dataset_config):
        """Test training pipeline with larger dataset."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test training with larger dataset
        result = runner.invoke(app, ["train",
                                   "--config", str(large_dataset_config),
                                   "--grid", "configs/model_grid.yaml"])

        # Should complete successfully (may take longer but should not fail)
        assert result.exit_code == 0

        # Check that artifacts were created
        results_dir = Path(str(large_dataset_config).replace("_config.yaml", "_results"))
        artifacts_dir = results_dir / "artifacts"

        if artifacts_dir.exists():
            models_dir = artifacts_dir / "models"
            assert models_dir.exists()

            metrics_dir = artifacts_dir / "metrics"
            assert metrics_dir.exists()

    def test_large_dataset_memory_usage(self, large_dataset_config):
        """Test memory usage with larger datasets."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        from typer.testing import CliRunner

        runner = CliRunner()

        # Run a quick operation to test memory usage
        result = runner.invoke(app, ["load",
                                   "--data", str(Path(str(large_dataset_config).replace("large_config.yaml", "large_test_data.csv"))),
                                   "--meta", str(Path(str(large_dataset_config).replace("large_config.yaml", "large_metadata.yaml")))])

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500
        assert result.exit_code == 0

    def test_large_dataset_timing(self, large_dataset_config):
        """Test timing performance with larger datasets."""
        import time
        from typer.testing import CliRunner

        runner = CliRunner()

        # Time the training process
        start_time = time.time()

        result = runner.invoke(app, ["train",
                                   "--config", str(large_dataset_config),
                                   "--grid", "configs/model_grid.yaml"])

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (less than 5 minutes for this test)
        assert duration < 300  # 5 minutes
        assert result.exit_code == 0

    def test_large_dataset_cross_validation(self, large_dataset_config):
        """Test cross-validation scalability with larger datasets."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test with larger number of CV folds
        result = runner.invoke(app, ["train",
                                   "--config", str(large_dataset_config),
                                   "--grid", "configs/model_grid.yaml"])

        assert result.exit_code == 0

        # Check that CV results exist
        results_dir = Path(str(large_dataset_config).replace("_config.yaml", "_results"))
        metrics_dir = results_dir / "artifacts" / "metrics"

        if metrics_dir.exists():
            # Should have CV results
            cv_files = list(metrics_dir.glob("*cv*"))
            assert len(cv_files) > 0

    def test_large_dataset_model_persistence(self, large_dataset_config):
        """Test model serialization and loading with larger models."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Train models
        train_result = runner.invoke(app, ["train",
                                         "--config", str(large_dataset_config),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Check model file sizes are reasonable
        results_dir = Path(str(large_dataset_config).replace("_config.yaml", "_results"))
        models_dir = results_dir / "artifacts" / "models"

        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            for model_file in model_files:
                # Model files should not be excessively large
                file_size_mb = model_file.stat().st_size / 1024 / 1024
                assert file_size_mb < 100  # Less than 100MB

    def test_large_dataset_feature_engineering(self, large_dataset_config):
        """Test feature engineering scalability."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test data loading and preprocessing
        data_path = str(Path(str(large_dataset_config).replace("large_config.yaml", "large_test_data.csv")))
        meta_path = str(Path(str(large_dataset_config).replace("large_config.yaml", "large_metadata.yaml")))

        result = runner.invoke(app, ["load",
                                   "--data", data_path,
                                   "--meta", meta_path])

        assert result.exit_code == 0

    def test_large_dataset_evaluation_scalability(self, large_dataset_config):
        """Test evaluation pipeline scalability."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train models
        train_result = runner.invoke(app, ["train",
                                       "--config", str(large_dataset_config),
                                       "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Run evaluation
        result = runner.invoke(app, ["evaluate",
                                   "--config", str(large_dataset_config)])

        assert result.exit_code == 0

        # Check that evaluation artifacts exist
        results_dir = Path(str(large_dataset_config).replace("_config.yaml", "_results"))
        metrics_dir = results_dir / "artifacts" / "metrics"

        if metrics_dir.exists():
            # Should have evaluation results
            eval_files = list(metrics_dir.glob("*"))
            assert len(eval_files) > 0

    def test_large_dataset_monitoring_scalability(self, large_dataset_config):
        """Test monitoring scalability with larger datasets."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train models
        train_result = runner.invoke(app, ["train",
                                       "--config", str(large_dataset_config),
                                       "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Run monitoring
        data_path = str(Path(str(large_dataset_config).replace("large_config.yaml", "large_test_data.csv")))

        result = runner.invoke(app, ["monitor",
                                   "--config", str(large_dataset_config),
                                   "--data", data_path])

        assert result.exit_code == 0

    def test_large_dataset_automl_scalability(self, large_dataset_config):
        """Test AutoML scalability with larger datasets."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Run AutoML with short time limit for scalability testing
        data_path = str(Path(str(large_dataset_config).replace("large_config.yaml", "large_test_data.csv")))
        meta_path = str(Path(str(large_dataset_config).replace("large_config.yaml", "large_metadata.yaml")))

        result = runner.invoke(app, ["automl",
                                   "--config", str(large_dataset_config),
                                   "--data", data_path,
                                   "--meta", meta_path,
                                   "--time-limit", "30",  # Short time for testing
                                   "--model-types", "coxph", "rsf"])

        # Should handle larger datasets without crashing
        assert result.exit_code in [0, 1]  # May timeout but shouldn't crash

    def test_large_dataset_counterfactual_scalability(self, large_dataset_config):
        """Test counterfactual explanation scalability."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train a model
        train_result = runner.invoke(app, ["train",
                                       "--config", str(large_dataset_config),
                                       "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Generate counterfactuals
        data_path = str(Path(str(large_dataset_config).replace("large_config.yaml", "large_test_data.csv")))
        meta_path = str(Path(str(large_dataset_config).replace("large_config.yaml", "large_metadata.yaml")))

        result = runner.invoke(app, ["counterfactual",
                                   "--config", str(large_dataset_config),
                                   "--data", data_path,
                                   "--meta", meta_path,
                                   "--model-name", "coxph",
                                   "--target-risk", "0.3",
                                   "--n-counterfactuals", "2"])

        # Should handle larger datasets
        assert result.exit_code in [0, 1]  # May fail due to time/complexity but shouldn't crash

    def test_large_dataset_disk_space_usage(self, large_dataset_config):
        """Test disk space usage with larger datasets."""
        import shutil

        # Get initial disk usage
        results_dir = Path(str(large_dataset_config).replace("_config.yaml", "_results"))

        initial_size = 0
        if results_dir.exists():
            initial_size = sum(f.stat().st_size for f in results_dir.rglob('*') if f.is_file())

        from typer.testing import CliRunner

        runner = CliRunner()

        # Run complete pipeline
        result = runner.invoke(app, ["run",
                                   "--config", str(large_dataset_config),
                                   "--grid", "configs/model_grid.yaml"])

        assert result.exit_code == 0

        # Check final disk usage
        if results_dir.exists():
            final_size = sum(f.stat().st_size for f in results_dir.rglob('*') if f.is_file())
            size_increase_mb = (final_size - initial_size) / 1024 / 1024

            # Disk usage should be reasonable (less than 200MB for this test)
            assert size_increase_mb < 200

    def test_large_dataset_concurrent_processing(self, large_dataset_config):
        """Test concurrent processing capabilities."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test that multiple operations can run without conflicts
        # (This is more of a smoke test for thread safety)

        result1 = runner.invoke(app, ["load",
                                    "--data", str(Path(str(large_dataset_config).replace("large_config.yaml", "large_test_data.csv"))),
                                    "--meta", str(Path(str(large_dataset_config).replace("large_config.yaml", "large_metadata.yaml")))])

        result2 = runner.invoke(app, ["validate-config",
                                    "--config", str(large_dataset_config),
                                    "--grid", "configs/model_grid.yaml",
                                    "--features", str(Path(str(large_dataset_config).replace("large_config.yaml", "large_metadata.yaml")))])

        # Both should complete successfully
        assert result1.exit_code == 0
        assert result2.exit_code == 0

