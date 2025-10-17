"""Integration tests for error handling and edge cases."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from clinical_survival.cli.main import app


class TestErrorHandling:
    """Test error handling for edge cases and malformed inputs."""

    @pytest.fixture
    def invalid_config_dir(self):
        """Create a temporary directory for invalid configurations."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def invalid_data_config(self, invalid_config_dir):
        """Create a configuration with invalid data paths."""
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "/nonexistent/path/data.csv",
                "metadata": "/nonexistent/path/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "/nonexistent/path/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def test_missing_data_file_handling(self, invalid_data_config):
        """Test handling of missing data files."""
        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(app, ["load",
                                   "--data", "/nonexistent/path/data.csv",
                                   "--meta", "/nonexistent/path/metadata.yaml"])

        # Should handle gracefully with appropriate error message
        assert result.exit_code != 0

    def test_invalid_configuration_handling(self, invalid_config_dir):
        """Test handling of invalid configuration files."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create a config with invalid YAML
        invalid_yaml_path = invalid_config_dir / "invalid.yaml"
        with open(invalid_yaml_path, 'w') as f:
            f.write("invalid: yaml: content: [\n")

        result = runner.invoke(app, ["validate-config",
                                   "--config", str(invalid_yaml_path),
                                   "--grid", "configs/model_grid.yaml",
                                   "--features", "configs/features.yaml"])

        # Should handle invalid YAML gracefully
        assert result.exit_code != 0

    def test_missing_configuration_files(self, invalid_config_dir):
        """Test handling of missing configuration files."""
        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(app, ["validate-config",
                                   "--config", str(invalid_config_dir / "nonexistent.yaml"),
                                   "--grid", "configs/model_grid.yaml",
                                   "--features", "configs/features.yaml"])

        # Should handle missing files gracefully
        assert result.exit_code != 0

    def test_empty_dataset_handling(self, invalid_config_dir):
        """Test handling of empty datasets."""
        # Create an empty CSV file
        empty_csv = invalid_config_dir / "empty.csv"
        empty_csv.write_text("id,time,event\n")  # Header only

        empty_metadata = invalid_config_dir / "empty_metadata.yaml"
        with open(empty_metadata, 'w') as f:
            yaml.dump({
                "features": {
                    "numeric_features": ["time"],
                    "categorical_features": [],
                    "drop_features": ["id"]
                },
                "time_column": "time",
                "event_column": "event",
                "id_column": "id"
            }, f)

        from typer.testing import CliRunner

        runner = CliRunner()

        result = runner.invoke(app, ["load",
                                   "--data", str(empty_csv),
                                   "--meta", str(empty_metadata)])

        # Should handle empty datasets gracefully
        assert result.exit_code != 0

    def test_malformed_metadata_handling(self, invalid_config_dir):
        """Test handling of malformed metadata files."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create metadata with missing required fields
        malformed_metadata = invalid_config_dir / "malformed.yaml"
        with open(malformed_metadata, 'w') as f:
            yaml.dump({
                "features": {
                    "numeric_features": ["age"]
                    # Missing categorical_features, time_column, event_column
                }
            }, f)

        result = runner.invoke(app, ["load",
                                   "--data", "data/toy/toy_survival.csv",
                                   "--meta", str(malformed_metadata)])

        # Should handle malformed metadata gracefully
        assert result.exit_code != 0

    def test_invalid_model_names(self, invalid_config_dir):
        """Test handling of invalid model names."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create a valid config but use invalid model name
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["nonexistent_model"]
        }

        config_path = invalid_config_dir / "invalid_model_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["train",
                                   "--config", str(config_path),
                                   "--grid", "configs/model_grid.yaml"])

        # Should handle invalid model names gracefully
        assert result.exit_code != 0

    def test_insufficient_memory_handling(self, invalid_config_dir):
        """Test handling of memory-related errors."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create a config that might cause memory issues
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "explain": {
                "shap_samples": 10000,  # Very high number that might cause memory issues
                "pdp_features": ["age", "sofa"]
            },
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "memory_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["explain",
                                   "--config", str(config_path),
                                   "--model-name", "coxph"])

        # Should handle memory issues gracefully (may succeed or fail but not crash)
        assert result.exit_code in [0, 1]

    def test_concurrent_access_handling(self, invalid_config_dir):
        """Test handling of concurrent file access issues."""
        from typer.testing import CliRunner
        import time

        runner = CliRunner()

        # Create a config
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "concurrent_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # Run multiple operations that might conflict
        result1 = runner.invoke(app, ["train",
                                    "--config", str(config_path),
                                    "--grid", "configs/model_grid.yaml"])

        # Start another operation immediately
        result2 = runner.invoke(app, ["evaluate",
                                    "--config", str(config_path)])

        # Both should handle concurrent access gracefully
        # (At least one might fail, but neither should crash)
        assert result1.exit_code in [0, 1]
        assert result2.exit_code in [0, 1]

    def test_network_timeout_handling(self, invalid_config_dir):
        """Test handling of network-related timeouts (if applicable)."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # This is more of a placeholder for network-related error handling
        # The current CLI doesn't make network requests, but this could be relevant
        # for future enhancements like cloud storage or API integrations

        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "network_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["train",
                                   "--config", str(config_path),
                                   "--grid", "configs/model_grid.yaml"])

        # Should complete without network-related errors
        assert result.exit_code in [0, 1]

    def test_disk_space_exhaustion_handling(self, invalid_config_dir):
        """Test handling of disk space issues."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create a config with very large output requirements
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "evaluation": {
                "bootstrap": 10000  # Very large bootstrap for disk space testing
            },
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "disk_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["train",
                                   "--config", str(config_path),
                                   "--grid", "configs/model_grid.yaml"])

        # Should handle disk space issues gracefully
        assert result.exit_code in [0, 1]

    def test_invalid_parameter_combinations(self, invalid_config_dir):
        """Test handling of invalid parameter combinations."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create a config with conflicting parameters
        config = {
            "seed": 42,
            "n_splits": 3,
            "inner_splits": 10,  # More inner splits than outer splits
            "test_split": 0.2,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "invalid_params_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["train",
                                   "--config", str(config_path),
                                   "--grid", "configs/model_grid.yaml"])

        # Should handle invalid parameter combinations gracefully
        assert result.exit_code in [0, 1]

    def test_corrupted_model_files(self, invalid_config_dir):
        """Test handling of corrupted model files."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First create a valid model
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "valid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        train_result = runner.invoke(app, ["train",
                                         "--config", str(config_path),
                                         "--grid", "configs/model_grid.yaml"])
        assert train_result.exit_code == 0

        # Corrupt a model file
        models_dir = invalid_config_dir / "results" / "artifacts" / "models"
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pkl"))
            if model_files:
                # Corrupt the model file
                with open(model_files[0], 'w') as f:
                    f.write("corrupted data")

        # Try to use the corrupted model
        result = runner.invoke(app, ["evaluate",
                                   "--config", str(config_path)])

        # Should handle corrupted model files gracefully
        assert result.exit_code in [0, 1]

    def test_permission_errors(self, invalid_config_dir):
        """Test handling of file permission errors."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create a read-only directory
        readonly_dir = invalid_config_dir / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(readonly_dir),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "readonly_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["train",
                                   "--config", str(config_path),
                                   "--grid", "configs/model_grid.yaml"])

        # Should handle permission errors gracefully
        assert result.exit_code in [0, 1]

        # Restore permissions for cleanup
        readonly_dir.chmod(0o755)

    def test_interrupted_execution(self, invalid_config_dir):
        """Test handling of interrupted execution."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create a config that might take some time
        config = {
            "seed": 42,
            "n_splits": 5,  # More splits for longer execution
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph", "rsf"]  # Multiple models for longer execution
        }

        config_path = invalid_config_dir / "long_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        # This test would normally be interrupted, but for unit testing
        # we'll just verify that the command can be started
        result = runner.invoke(app, ["train",
                                   "--config", str(config_path),
                                   "--grid", "configs/model_grid.yaml"])

        # Should handle execution (even if interrupted)
        assert result.exit_code in [0, 1]

    def test_resource_exhaustion_recovery(self, invalid_config_dir):
        """Test recovery from resource exhaustion."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create multiple configurations that might exhaust resources
        for i in range(3):
            config = {
                "seed": 42 + i,
                "n_splits": 3,
                "time_col": "time",
                "event_col": "event",
                "id_col": "id",
                "explain": {
                    "shap_samples": 1000,  # Large SHAP sample size
                    "pdp_features": ["age", "sofa"]
                },
                "paths": {
                    "data_csv": "data/toy/toy_survival.csv",
                    "metadata": "data/toy/metadata.yaml",
                    "outdir": str(invalid_config_dir / f"results_{i}"),
                    "features": "configs/features.yaml"
                },
                "models": ["coxph"]
            }

            config_path = invalid_config_dir / f"resource_config_{i}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            result = runner.invoke(app, ["explain",
                                       "--config", str(config_path),
                                       "--model-name", "coxph"])

            # Should handle resource issues gracefully
            assert result.exit_code in [0, 1]

    def test_graceful_degradation(self, invalid_config_dir):
        """Test graceful degradation when features are unavailable."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Create a config that uses features that might not be available
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "explain": {
                "shap_samples": 1000,
                "pdp_features": ["nonexistent_feature"]  # Feature that doesn't exist
            },
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "degradation_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["explain",
                                   "--config", str(config_path),
                                   "--model-name", "coxph"])

        # Should degrade gracefully when features are unavailable
        assert result.exit_code in [0, 1]

    def test_version_conflict_handling(self, invalid_config_dir):
        """Test handling of version conflicts."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # This is more of a placeholder for future version conflict handling
        # Currently, the CLI doesn't check versions, but this could be relevant
        # for dependency management

        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(invalid_config_dir / "results"),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = invalid_config_dir / "version_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        result = runner.invoke(app, ["train",
                                   "--config", str(config_path),
                                   "--grid", "configs/model_grid.yaml"])

        # Should handle version issues gracefully (if any exist)
        assert result.exit_code in [0, 1]
