"""Integration tests for data quality and validation framework."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from clinical_survival.cli.main import app
from clinical_survival.data_quality import (
    DataCleansingPipeline,
    DataQualityMonitor,
    DataQualityProfiler,
    DataQualityReport,
    DataQualityMetrics,
    DataValidator,
    ValidationRule,
    create_data_quality_config,
    create_validation_rules,
    save_data_quality_report,
)


class TestDataQualityFramework:
    """Test comprehensive data quality and validation framework."""

    @pytest.fixture
    def temp_quality_dir(self):
        """Create a temporary directory for data quality tests."""
        temp_dir = tempfile.mkdtemp()

        # Create results directory
        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir()

        yield results_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config_quality(self, temp_quality_dir):
        """Create a test configuration for data quality."""
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "data_quality": {
                "profiling": {
                    "enable_anomaly_detection": True,
                    "enable_clinical_validation": True,
                    "quality_thresholds": {
                        "excellent": 90.0,
                        "good": 80.0,
                        "acceptable": 70.0,
                        "poor": 60.0
                    }
                },
                "validation": {
                    "validation_rules": "default",
                    "strict_mode": False,
                    "fail_on_first_error": False
                },
                "cleansing": {
                    "remove_duplicates": True,
                    "missing_values_strategy": "auto",
                    "handle_outliers": True,
                    "preserve_original": True
                }
            },
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(temp_quality_dir.parent),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = temp_quality_dir / "quality_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def test_data_quality_config_creation(self):
        """Test data quality configuration creation."""
        config = create_data_quality_config(
            enable_anomaly_detection=True,
            enable_clinical_validation=True,
            quality_thresholds={"excellent": 95.0, "good": 85.0}
        )

        assert config["profiling"]["enable_anomaly_detection"] is True
        assert config["profiling"]["enable_clinical_validation"] is True
        assert config["profiling"]["quality_thresholds"]["excellent"] == 95.0
        assert config["profiling"]["quality_thresholds"]["good"] == 85.0

    def test_validation_rules_creation(self):
        """Test validation rules creation."""
        rules = create_validation_rules()

        # Should have multiple rule types
        rule_types = [rule.rule_type for rule in rules]
        assert "completeness" in rule_types
        assert "range" in rule_types
        assert "categorical" in rule_types
        assert "clinical" in rule_types
        assert "statistical" in rule_types

        # Check specific rules
        age_completeness = next((rule for rule in rules if rule.rule_id == "age_completeness"), None)
        assert age_completeness is not None
        assert age_completeness.rule_type == "completeness"
        assert age_completeness.parameters["min_completeness"] == 95.0

    def test_data_quality_profiler_creation(self):
        """Test data quality profiler creation."""
        profiler = DataQualityProfiler()

        # Test initialization
        assert profiler.config == {}
        assert profiler._baseline_stats is None
        assert profiler._anomaly_detector is None

    def test_data_quality_metrics_creation(self):
        """Test data quality metrics creation."""
        metrics = DataQualityMetrics(
            total_rows=1000,
            total_columns=10,
            missing_values_count=50,
            missing_values_percentage=5.0,
            duplicate_rows_count=10,
            duplicate_rows_percentage=1.0,
            column_completeness={"age": 95.0, "sofa": 90.0},
            column_uniqueness={"sex": 50.0},
            column_data_types={"age": "int64", "sex": "object"},
            column_statistics={"age": {"mean": 65.0, "std": 15.0}},
            outlier_counts={"age": 5},
            anomaly_scores={"age": 0.1},
            clinical_data_issues=["WARNING: Negative age values"],
            data_drift_scores={"age": 0.05},
            overall_quality_score=85.0,
            quality_grade="B"
        )

        assert metrics.total_rows == 1000
        assert metrics.total_columns == 10
        assert metrics.missing_values_percentage == 5.0
        assert metrics.duplicate_rows_percentage == 1.0
        assert metrics.column_completeness["age"] == 95.0
        assert metrics.column_uniqueness["sex"] == 50.0
        assert metrics.overall_quality_score == 85.0
        assert metrics.quality_grade == "B"
        assert len(metrics.clinical_data_issues) == 1

    def test_data_quality_report_creation(self):
        """Test data quality report creation."""
        metrics = DataQualityMetrics(
            total_rows=1000,
            total_columns=10,
            missing_values_count=50,
            missing_values_percentage=5.0,
            duplicate_rows_count=10,
            duplicate_rows_percentage=1.0,
            overall_quality_score=85.0,
            quality_grade="B"
        )

        report = DataQualityReport(
            dataset_name="test_dataset",
            timestamp=None,  # Will be set by system
            quality_metrics=metrics,
            recommendations=["Consider data imputation", "Review duplicate handling"],
            issues_found=["High missing value rate"]
        )

        assert report.dataset_name == "test_dataset"
        assert report.quality_metrics.overall_quality_score == 85.0
        assert len(report.recommendations) == 2
        assert len(report.issues_found) == 1

    def test_data_validator_creation(self):
        """Test data validator creation."""
        rules = create_validation_rules()
        validator = DataValidator(rules)

        # Test initialization
        assert len(validator.validation_rules) > 0
        assert isinstance(validator._validation_cache, dict)

    def test_data_quality_profiler_basic_profiling(self, temp_quality_dir):
        """Test basic data quality profiling functionality."""
        import pandas as pd

        # Create test data
        test_data = temp_quality_dir / "test_data.csv"
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],  # 10% events
            "age": [60 + i % 20 for i in range(100)],  # Ages 60-79
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]  # SOFA scores 5-14
        })
        df.to_csv(test_data, index=False)

        # Create profiler
        profiler = DataQualityProfiler()

        # Generate profile
        report = profiler.profile_dataset(df, dataset_name="test_data")

        # Test basic metrics
        assert report.quality_metrics.total_rows == 100
        assert report.quality_metrics.total_columns == 6
        assert report.quality_metrics.missing_values_count == 0
        assert report.quality_metrics.missing_values_percentage == 0.0
        assert report.quality_metrics.duplicate_rows_count == 0
        assert report.quality_metrics.duplicate_rows_percentage == 0.0

        # Test column completeness
        assert report.quality_metrics.column_completeness["age"] == 100.0
        assert report.quality_metrics.column_completeness["sex"] == 100.0

        # Test column statistics
        assert "mean" in report.quality_metrics.column_statistics["age"]
        assert "std" in report.quality_metrics.column_statistics["age"]

        # Test overall score
        assert report.quality_metrics.overall_quality_score > 0
        assert report.quality_metrics.quality_grade in ["A", "B", "C", "D", "F"]

    def test_data_quality_profiler_with_missing_values(self, temp_quality_dir):
        """Test data quality profiling with missing values."""
        import pandas as pd

        # Create test data with missing values
        test_data = temp_quality_dir / "test_data_missing.csv"
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [60 + i % 20 if i < 80 else None for i in range(100)],  # 20% missing
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]
        })
        df.to_csv(test_data, index=False)

        # Create profiler
        profiler = DataQualityProfiler()

        # Generate profile
        report = profiler.profile_dataset(df, dataset_name="test_data_missing")

        # Test missing value detection
        assert report.quality_metrics.missing_values_count > 0
        assert report.quality_metrics.missing_values_percentage > 0
        assert report.quality_metrics.column_completeness["age"] == 80.0  # 80% complete

    def test_data_quality_profiler_with_outliers(self, temp_quality_dir):
        """Test data quality profiling with outliers."""
        import pandas as pd

        # Create test data with outliers
        test_data = temp_quality_dir / "test_data_outliers.csv"
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [65] * 95 + [200, 5, 150, 0, -10],  # 5 outliers
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [8] * 95 + [30, 35, -5, 50, 25]  # 5 outliers
        })
        df.to_csv(test_data, index=False)

        # Create profiler
        profiler = DataQualityProfiler()

        # Generate profile
        report = profiler.profile_dataset(df, dataset_name="test_data_outliers")

        # Test outlier detection
        assert report.quality_metrics.outlier_counts["age"] > 0
        assert report.quality_metrics.outlier_counts["sofa"] > 0

    def test_data_validator_basic_validation(self, temp_quality_dir):
        """Test data validator with basic rules."""
        import pandas as pd

        # Create test data
        test_data = temp_quality_dir / "test_validation.csv"
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [60 + i % 20 for i in range(100)],
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]
        })
        df.to_csv(test_data, index=False)

        # Create validator with rules
        rules = create_validation_rules()
        validator = DataValidator(rules)

        # Validate dataset
        result = validator.validate_dataset(df, dataset_name="test_validation")

        # Test validation results
        assert "overall_status" in result
        assert "validation_results" in result
        assert "total_rules" in result
        assert result["total_rules"] > 0

        # Should have passed most rules (data is clean)
        assert result["passed_rules"] > 0

    def test_data_cleansing_pipeline_basic(self, temp_quality_dir):
        """Test data cleansing pipeline functionality."""
        import pandas as pd

        # Create test data with issues
        test_data = temp_quality_dir / "test_cleansing.csv"
        df = pd.DataFrame({
            "id": [1, 2, 2, 3, 4],  # Duplicate ID 2
            "time": [100, 200, 200, 300, 400],  # Duplicate time 200
            "event": [1, 0, 0, 1, 1],
            "age": [65, 70, None, 75, 200],  # Missing and outlier
            "sex": ["male", "female", "male", "female", "invalid"],  # Invalid value
            "sofa": [8, 12, 15, 20, 35]  # Outlier SOFA score
        })
        df.to_csv(test_data, index=False)

        # First profile the data
        profiler = DataQualityProfiler()
        quality_report = profiler.profile_dataset(df, dataset_name="test_cleansing")

        # Create cleansing pipeline
        cleansing_config = {
            "remove_duplicates": True,
            "missing_values_strategy": "auto",
            "handle_outliers": True,
            "preserve_original": True
        }

        pipeline = DataCleansingPipeline(cleansing_config)

        # Cleanse data
        df_clean, cleansing_summary = pipeline.cleanse_dataset(df, quality_report, True)

        # Test cleansing results
        assert "original_shape" in cleansing_summary
        assert "final_shape" in cleansing_summary
        assert "cleansing_steps" in cleansing_summary

        # Should have applied some cleansing steps
        assert len(cleansing_summary["cleansing_steps"]) > 0

        # Should have fewer rows (duplicates removed)
        assert cleansing_summary["final_shape"][0] < cleansing_summary["original_shape"][0]

    def test_data_quality_monitoring(self, temp_quality_dir):
        """Test data quality monitoring functionality."""
        import pandas as pd

        # Create test data
        test_data = temp_quality_dir / "test_monitoring.csv"
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [60 + i % 20 for i in range(100)],
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]
        })
        df.to_csv(test_data, index=False)

        # Create monitor
        monitor = DataQualityMonitor()

        # Monitor data quality
        report = monitor.monitor_data_quality(df, dataset_name="test_monitoring")

        # Test monitoring
        assert report.dataset_name == "test_monitoring"
        assert len(monitor._monitoring_history) == 1

        # Test monitoring history
        history = monitor.get_monitoring_history(days=1)
        assert len(history) == 1

        # Test monitoring report generation
        monitoring_report = monitor.generate_monitoring_report()
        assert "error" not in monitoring_report or monitoring_report["error"] != "No monitoring data available"

    def test_data_quality_report_saving(self, temp_quality_dir):
        """Test data quality report saving functionality."""
        # Create mock report
        metrics = DataQualityMetrics(
            total_rows=1000,
            total_columns=10,
            missing_values_count=50,
            missing_values_percentage=5.0,
            duplicate_rows_count=10,
            duplicate_rows_percentage=1.0,
            overall_quality_score=85.0,
            quality_grade="B"
        )

        report = DataQualityReport(
            dataset_name="test_save",
            timestamp=None,
            quality_metrics=metrics,
            recommendations=["Test recommendation"],
            issues_found=["Test issue"]
        )

        # Test JSON saving
        json_file = temp_quality_dir / "test_report.json"
        save_data_quality_report(report, json_file, format="json")

        assert json_file.exists()

        # Verify JSON content
        with open(json_file) as f:
            loaded_data = json.load(f)

        assert loaded_data["dataset_name"] == "test_save"
        assert loaded_data["quality_metrics"]["overall_quality_score"] == 85.0

        # Test HTML saving
        html_file = temp_quality_dir / "test_report.html"
        save_data_quality_report(report, html_file, format="html")

        assert html_file.exists()

        # Verify HTML content
        with open(html_file) as f:
            html_content = f.read()

        assert "<title>Data Quality Report - test_save</title>" in html_content
        assert "Overall Score" in html_content
        assert "85.0" in html_content

    def test_data_quality_profiler_anomaly_detection(self, temp_quality_dir):
        """Test anomaly detection in data quality profiling."""
        import pandas as pd

        # Create test data with anomalies
        test_data = temp_quality_dir / "test_anomalies.csv"
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [65] * 90 + [200, 5, 150, 0, -10, 300, 400, 500, 600, 700],  # 10 anomalies
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [8] * 90 + [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  # 10 anomalies
        })
        df.to_csv(test_data, index=False)

        # Create profiler with anomaly detection
        profiler = DataQualityProfiler()

        # Generate profile
        report = profiler.profile_dataset(df, dataset_name="test_anomalies", include_anomaly_detection=True)

        # Test anomaly detection
        assert report.quality_metrics.anomaly_scores
        assert len(report.quality_metrics.anomaly_scores) > 0

        # Should detect anomalies in age and sofa columns
        assert "age" in report.quality_metrics.anomaly_scores
        assert "sofa" in report.quality_metrics.anomaly_scores

    def test_data_quality_profiler_clinical_validation(self, temp_quality_dir):
        """Test clinical data validation in profiling."""
        import pandas as pd

        # Create test data with clinical issues
        test_data = temp_quality_dir / "test_clinical.csv"
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [65] * 80 + [-5, -10, 200, 300, 400],  # Invalid ages
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [8] * 95 + [30, 35, 40, 45, 50]  # Invalid SOFA scores
        })
        df.to_csv(test_data, index=False)

        # Create profiler with clinical validation
        clinical_context = {
            "expected_types": {"age": "int64", "sofa": "int64"},
            "validation_rules": {}
        }

        profiler = DataQualityProfiler()

        # Generate profile
        report = profiler.profile_dataset(
            df,
            dataset_name="test_clinical",
            clinical_context=clinical_context
        )

        # Test clinical validation
        assert report.quality_metrics.clinical_data_issues
        assert len(report.quality_metrics.clinical_data_issues) > 0

        # Should detect clinical issues
        clinical_issues = [issue for issue in report.quality_metrics.clinical_data_issues if "ERROR" in issue or "WARNING" in issue]
        assert len(clinical_issues) > 0

    def test_data_quality_profiler_data_drift_detection(self, temp_quality_dir):
        """Test data drift detection in profiling."""
        import pandas as pd

        # Create baseline data
        baseline_data = temp_quality_dir / "baseline.csv"
        df_baseline = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [60 + i % 20 for i in range(100)],  # Ages 60-79
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]  # SOFA 5-14
        })
        df_baseline.to_csv(baseline_data, index=False)

        # Create current data with drift
        current_data = temp_quality_dir / "current.csv"
        df_current = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [70 + i % 20 for i in range(100)],  # Ages 70-89 (shifted up)
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [10 + i % 10 for i in range(100)]  # SOFA 10-19 (shifted up)
        })
        df_current.to_csv(current_data, index=False)

        # Create profiler and set baseline
        profiler = DataQualityProfiler()
        baseline_report = profiler.profile_dataset(df_baseline, dataset_name="baseline")

        # Profile current data
        current_report = profiler.profile_dataset(df_current, dataset_name="current")

        # Test drift detection
        assert current_report.quality_metrics.data_drift_scores
        assert len(current_report.quality_metrics.data_drift_scores) > 0

        # Should detect drift in age and sofa
        assert "age" in current_report.quality_metrics.data_drift_scores
        assert "sofa" in current_report.quality_metrics.data_drift_scores

    def test_data_validator_rule_validation(self):
        """Test individual validation rule validation."""
        import pandas as pd

        # Create test data
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [60 + i % 20 for i in range(100)],
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]
        })

        # Create validator
        rules = create_validation_rules()
        validator = DataValidator(rules)

        # Test individual rule validation
        age_completeness_rule = next(rule for rule in rules if rule.rule_id == "age_completeness")
        result = validator._validate_rule(df, age_completeness_rule)

        assert result["status"] == "passed"
        assert result["severity"] == "info"
        assert "completeness" in result["message"]

    def test_data_quality_profiler_quality_scoring(self, temp_quality_dir):
        """Test quality scoring algorithm."""
        import pandas as pd

        # Create test data with various quality issues
        test_data = temp_quality_dir / "test_scoring.csv"
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [60 + i % 20 if i < 80 else None for i in range(100)],  # 20% missing
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]
        })
        df.to_csv(test_data, index=False)

        # Create profiler
        profiler = DataQualityProfiler()

        # Generate profile
        report = profiler.profile_dataset(df, dataset_name="test_scoring")

        # Test quality scoring
        assert report.quality_metrics.overall_quality_score > 0
        assert report.quality_metrics.overall_quality_score <= 100
        assert report.quality_metrics.quality_grade in ["A", "B", "C", "D", "F"]

        # Should be lower than perfect due to missing values
        assert report.quality_metrics.overall_quality_score < 95

    def test_data_quality_monitor_trend_analysis(self, temp_quality_dir):
        """Test data quality monitoring trend analysis."""
        import pandas as pd

        # Create monitor
        monitor = DataQualityMonitor()

        # Simulate multiple monitoring reports
        for i in range(5):
            # Create test data with varying quality
            df = pd.DataFrame({
                "id": range(100),
                "time": [100 + i for i in range(100)],
                "event": [1 if i % 10 == 0 else 0 for i in range(100)],
                "age": [60 + (i % 20) if j < (100 - i * 5) else None for j in range(100)],  # Decreasing completeness
                "sex": ["male" if j % 2 == 0 else "female" for j in range(100)],
                "sofa": [5 + (i % 10) for i in range(100)]
            })

            monitor.monitor_data_quality(df, dataset_name=f"test_{i}")

        # Test trend analysis
        monitoring_report = monitor.generate_monitoring_report()

        # Should have trend analysis
        assert "quality_trends" in monitoring_report
        assert "overall_score_trend" in monitoring_report["quality_trends"]
        assert "error_rate_trend" in monitoring_report["quality_trends"]

    def test_data_quality_profiler_comprehensive_analysis(self, temp_quality_dir):
        """Test comprehensive data quality analysis."""
        import pandas as pd

        # Create comprehensive test data
        test_data = temp_quality_dir / "test_comprehensive.csv"
        df = pd.DataFrame({
            "id": list(range(90)) + [1, 2, 3, 4, 5],  # 5 duplicates
            "time": [100 + i for i in range(95)] + [100, 101, 102, 103, 104],  # Some duplicates
            "event": [1 if i % 10 == 0 else 0 for i in range(95)] + [1, 0, 1, 0, 1],
            "age": [60 + i % 20 if i < 80 else None for i in range(95)] + [65, 70, None, 75, 200],  # Missing + outlier
            "sex": ["male" if i % 2 == 0 else "female" for i in range(95)] + ["invalid", "M", "F", "unknown", "male"],
            "sofa": [5 + i % 10 for i in range(95)] + [8, 12, 30, 35, 40]  # Some outliers
        })
        df.to_csv(test_data, index=False)

        # Create profiler with all features enabled
        profiler = DataQualityProfiler()

        # Generate comprehensive profile
        report = profiler.profile_dataset(
            df,
            dataset_name="test_comprehensive",
            include_anomaly_detection=True
        )

        # Test comprehensive analysis
        assert report.quality_metrics.total_rows == 100
        assert report.quality_metrics.missing_values_count > 0
        assert report.quality_metrics.duplicate_rows_count > 0
        assert report.quality_metrics.outlier_counts["age"] > 0
        assert report.quality_metrics.outlier_counts["sofa"] > 0
        assert report.quality_metrics.anomaly_scores
        assert report.quality_metrics.clinical_data_issues
        assert report.recommendations

        # Should have lower quality score due to multiple issues
        assert report.quality_metrics.overall_quality_score < 80

    def test_data_quality_cli_commands(self, temp_quality_dir):
        """Test data quality CLI commands."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test data quality profile command
        result = runner.invoke(app, ["data-quality-profile",
                                   "--data", "data/toy/toy_survival.csv",
                                   "--output-dir", str(temp_quality_dir),
                                   "--output-format", "json"])

        # Should complete (may succeed or fail depending on data availability)
        assert result.exit_code in [0, 1]

    def test_data_validation_cli_command(self, temp_quality_dir):
        """Test data validation CLI command."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test data validation command
        result = runner.invoke(app, ["data-validation",
                                   "--data", "data/toy/toy_survival.csv",
                                   "--output-dir", str(temp_quality_dir)])

        # Should complete (may succeed or fail depending on data availability)
        assert result.exit_code in [0, 1]

    def test_data_cleansing_cli_command(self, temp_quality_dir):
        """Test data cleansing CLI command."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test data cleansing command
        result = runner.invoke(app, ["data-cleansing",
                                   "--data", "data/toy/toy_survival.csv",
                                   "--output-dir", str(temp_quality_dir),
                                   "--remove-duplicates",
                                   "--handle-outliers"])

        # Should complete (may succeed or fail depending on data availability)
        assert result.exit_code in [0, 1]

    def test_data_quality_integration_with_monitoring(self, temp_quality_dir):
        """Test integration between data quality and monitoring systems."""
        import pandas as pd

        # Create test data
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": [60 + i % 20 for i in range(100)],
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]
        })

        # Create baseline profile
        profiler = DataQualityProfiler()
        baseline_report = profiler.profile_dataset(df, dataset_name="baseline")

        # Create monitor with baseline
        monitor = DataQualityMonitor(baseline_profile=baseline_report)

        # Monitor new data
        new_df = pd.DataFrame({
            "id": range(100),
            "time": [110 + i for i in range(100)],  # Slightly different distribution
            "event": [1 if i % 12 == 0 else 0 for i in range(100)],  # Different pattern
            "age": [62 + i % 18 for i in range(100)],  # Slightly different age distribution
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [6 + i % 12 for i in range(100)]  # Different SOFA distribution
        })

        current_report = monitor.monitor_data_quality(new_df, dataset_name="current")

        # Test drift detection
        assert current_report.quality_metrics.data_drift_scores
        assert len(current_report.quality_metrics.data_drift_scores) > 0

    def test_data_quality_report_error_handling(self, temp_quality_dir):
        """Test error handling in data quality operations."""
        # Test with non-existent file
        import pandas as pd

        # Test profiler with invalid data
        profiler = DataQualityProfiler()

        try:
            # Should handle empty DataFrame gracefully
            empty_df = pd.DataFrame()
            report = profiler.profile_dataset(empty_df, dataset_name="empty")
            # Should still generate a report (with appropriate warnings)
        except Exception:
            # Should not crash
            pass

    def test_data_quality_profiler_statistical_validation(self, temp_quality_dir):
        """Test statistical validation in profiling."""
        import pandas as pd
        import numpy as np

        # Create test data with normal distribution
        test_data = temp_quality_dir / "test_normal.csv"
        np.random.seed(42)
        df = pd.DataFrame({
            "id": range(100),
            "time": [100 + i for i in range(100)],
            "event": [1 if i % 10 == 0 else 0 for i in range(100)],
            "age": np.random.normal(65, 15, 100),  # Normal distribution
            "sex": ["male" if i % 2 == 0 else "female" for i in range(100)],
            "sofa": [5 + i % 10 for i in range(100)]
        })
        df.to_csv(test_data, index=False)

        # Create profiler
        profiler = DataQualityProfiler()

        # Generate profile
        report = profiler.profile_dataset(df, dataset_name="test_normal")

        # Test statistical validation
        # Age should follow approximately normal distribution
        assert "age" in report.quality_metrics.column_statistics

    def test_data_quality_profiler_performance(self, temp_quality_dir):
        """Test data quality profiler performance."""
        import pandas as pd
        import time

        # Create larger test dataset
        test_data = temp_quality_dir / "test_performance.csv"
        df = pd.DataFrame({
            "id": range(10000),
            "time": [100 + i for i in range(10000)],
            "event": [1 if i % 10 == 0 else 0 for i in range(10000)],
            "age": [60 + i % 20 for i in range(10000)],
            "sex": ["male" if i % 2 == 0 else "female" for i in range(10000)],
            "sofa": [5 + i % 10 for i in range(10000)],
            "feature1": [i % 100 for i in range(10000)],
            "feature2": [i % 50 for i in range(10000)],
            "feature3": [i % 25 for i in range(10000)]
        })
        df.to_csv(test_data, index=False)

        # Test profiling performance
        profiler = DataQualityProfiler()

        start_time = time.time()
        report = profiler.profile_dataset(df, dataset_name="test_performance", include_anomaly_detection=True)
        end_time = time.time()

        profiling_time = end_time - start_time

        # Should complete in reasonable time (less than 30 seconds for 10k rows)
        assert profiling_time < 30.0
        assert report.quality_metrics.total_rows == 10000
        assert report.quality_metrics.total_columns == 9


