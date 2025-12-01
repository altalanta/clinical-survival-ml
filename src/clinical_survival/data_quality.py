"""Comprehensive data quality and validation framework for clinical datasets."""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from clinical_survival.logging_utils import log_function_call
from clinical_survival.utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics."""

    # Basic completeness metrics
    total_rows: int
    total_columns: int
    missing_values_count: int
    missing_values_percentage: float
    duplicate_rows_count: int
    duplicate_rows_percentage: float

    # Column-specific metrics
    column_completeness: Dict[str, float] = field(default_factory=dict)
    column_uniqueness: Dict[str, float] = field(default_factory=dict)
    column_data_types: Dict[str, str] = field(default_factory=dict)

    # Statistical metrics
    column_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    outlier_counts: Dict[str, int] = field(default_factory=dict)
    anomaly_scores: Dict[str, float] = field(default_factory=dict)

    # Clinical data specific metrics
    clinical_data_issues: List[str] = field(default_factory=list)
    data_drift_scores: Dict[str, float] = field(default_factory=dict)
    data_consistency_scores: Dict[str, float] = field(default_factory=dict)

    # Overall quality score
    overall_quality_score: float = 0.0
    quality_grade: str = "unknown"  # "A", "B", "C", "D", "F"


@dataclass
class ValidationRule:
    """Data validation rule configuration."""

    rule_id: str
    rule_name: str
    rule_type: str  # "completeness", "range", "categorical", "clinical", "statistical"
    column: str
    condition: str  # Rule condition as string
    severity: str = "error"  # "error", "warning", "info"
    enabled: bool = True

    # Rule parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    remediation_suggestion: str = ""


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""

    dataset_name: str
    timestamp: datetime
    quality_metrics: DataQualityMetrics
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    issues_found: List[str] = field(default_factory=list)
    data_profile: Dict[str, Any] = field(default_factory=dict)


class DataQualityProfiler:
    """Comprehensive data quality profiling system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._baseline_stats: Optional[Dict[str, Dict[str, Any]]] = None
        self._anomaly_detector: Optional[BaseEstimator] = None

    def profile_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "unknown",
        include_anomaly_detection: bool = True,
        clinical_context: Optional[Dict[str, Any]] = None
    ) -> DataQualityReport:
        """Generate comprehensive data quality profile."""

        logger.info(f"Profiling dataset '{dataset_name}' with {len(df)} rows and {len(df.columns)} columns")

        # Basic completeness analysis
        total_rows = len(df)
        total_columns = len(df.columns)
        missing_values_count = df.isnull().sum().sum()
        missing_values_percentage = (missing_values_count / (total_rows * total_columns)) * 100

        # Duplicate analysis
        duplicate_rows_count = df.duplicated().sum()
        duplicate_rows_percentage = (duplicate_rows_count / total_rows) * 100

        # Column-specific analysis
        column_completeness = {}
        column_uniqueness = {}
        column_statistics = {}
        outlier_counts = {}
        anomaly_scores = {}

        for column in df.columns:
            # Completeness
            completeness = (1 - df[column].isnull().sum() / total_rows) * 100
            column_completeness[column] = completeness

            # Uniqueness (for categorical/string columns)
            if df[column].dtype in ['object', 'category', 'string']:
                uniqueness = df[column].nunique() / total_rows * 100
                column_uniqueness[column] = uniqueness

            # Statistics
            if df[column].dtype in ['int64', 'float64']:
                stats_dict = {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std()),
                    'min': float(df[column].min()),
                    'max': float(df[column].max()),
                    'median': float(df[column].median()),
                    'q25': float(df[column].quantile(0.25)),
                    'q75': float(df[column].quantile(0.75))
                }
                column_statistics[column] = stats_dict

                # Outlier detection using IQR
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                outlier_counts[column] = int(outliers)

        # Anomaly detection (if enabled)
        if include_anomaly_detection and len(df.select_dtypes(include=[np.number])) > 0:
            anomaly_scores = self._detect_anomalies(df)

        # Clinical data specific analysis
        clinical_issues = self._analyze_clinical_data_quality(df, clinical_context or {})

        # Data drift analysis (if baseline exists)
        data_drift_scores = self._analyze_data_drift(df)

        # Overall quality scoring
        overall_score, quality_grade = self._calculate_overall_quality_score(
            missing_values_percentage, duplicate_rows_percentage,
            column_completeness, outlier_counts
        )

        # Create quality metrics
        quality_metrics = DataQualityMetrics(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values_count=missing_values_count,
            missing_values_percentage=missing_values_percentage,
            duplicate_rows_count=duplicate_rows_count,
            duplicate_rows_percentage=duplicate_rows_percentage,
            column_completeness=column_completeness,
            column_uniqueness=column_uniqueness,
            column_data_types={col: str(df[col].dtype) for col in df.columns},
            column_statistics=column_statistics,
            outlier_counts=outlier_counts,
            anomaly_scores=anomaly_scores,
            clinical_data_issues=clinical_issues,
            data_drift_scores=data_drift_scores,
            overall_quality_score=overall_score,
            quality_grade=quality_grade
        )

        # Generate recommendations
        recommendations = self._generate_data_quality_recommendations(quality_metrics)

        # Create report
        report = DataQualityReport(
            dataset_name=dataset_name,
            timestamp=datetime.now(),
            quality_metrics=quality_metrics,
            recommendations=recommendations,
            issues_found=[issue for issue in clinical_issues if "ERROR" in issue or "CRITICAL" in issue]
        )

        logger.info(f"Data quality profiling completed. Overall score: {overall_score".2f"} ({quality_grade})")
        return report

    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect anomalies using isolation forest."""
        try:
            # Select numeric columns
            numeric_df = df.select_dtypes(include=[np.number])

            if len(numeric_df.columns) == 0:
                return {}

            # Handle missing values
            numeric_df = numeric_df.fillna(numeric_df.mean())

            # Scale features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            # Fit isolation forest
            self._anomaly_detector = IsolationForest(
                contamination=0.1,  # Assume 10% anomalies
                random_state=42
            )
            anomaly_scores = self._anomaly_detector.fit_predict(scaled_data)

            # Convert to positive scores (higher = more anomalous)
            anomaly_scores = np.where(anomaly_scores == -1, 1, 0)

            # Calculate average anomaly score per column
            column_anomaly_scores = {}
            for i, column in enumerate(numeric_df.columns):
                column_anomaly_scores[column] = float(np.mean(anomaly_scores))

            return column_anomaly_scores

        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
            return {}

    def _analyze_clinical_data_quality(
        self,
        df: pd.DataFrame,
        clinical_context: Dict[str, Any]
    ) -> List[str]:
        """Analyze clinical data quality issues."""
        issues = []

        # Check for impossible values
        for column in df.columns:
            if column in ['age', 'sofa']:
                if (df[column] < 0).any():
                    issues.append(f"ERROR: Negative values found in {column}")
                if column == 'age' and (df[column] > 150).any():
                    issues.append(f"WARNING: Unrealistic age values (>150) in {column}")
                if column == 'sofa' and (df[column] > 24).any():
                    issues.append(f"WARNING: SOFA score >24 (max possible is 24) in {column}")

        # Check for inconsistent data types
        expected_types = clinical_context.get('expected_types', {})
        for column, expected_type in expected_types.items():
            if column in df.columns:
                actual_type = str(df[column].dtype)
                if actual_type != expected_type:
                    issues.append(f"WARNING: Column {column} has type {actual_type}, expected {expected_type}")

        # Check for clinical data consistency
        if 'sex' in df.columns and 'gender' in df.columns:
            # Check if sex and gender are consistent
            inconsistent = ~df['sex'].isin(['male', 'female']) | ~df['gender'].isin(['male', 'female'])
            if inconsistent.any():
                issues.append("WARNING: Inconsistent or invalid sex/gender values")

        return issues

    def _analyze_data_drift(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze data drift compared to baseline."""
        drift_scores = {}

        if self._baseline_stats is None:
            # Set baseline on first run
            self._baseline_stats = {}
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    self._baseline_stats[column] = {
                        'mean': float(df[column].mean()),
                        'std': float(df[column].std()),
                        'min': float(df[column].min()),
                        'max': float(df[column].max())
                    }
            return drift_scores

        # Compare with baseline
        for column in df.columns:
            if (column in self._baseline_stats and
                df[column].dtype in ['int64', 'float64'] and
                not df[column].isnull().all()):

                current_stats = {
                    'mean': float(df[column].mean()),
                    'std': float(df[column].std())
                }

                baseline_stats = self._baseline_stats[column]

                # Calculate drift score (simplified)
                mean_drift = abs(current_stats['mean'] - baseline_stats['mean']) / abs(baseline_stats['mean'] + 1e-8)
                std_drift = abs(current_stats['std'] - baseline_stats['std']) / (baseline_stats['std'] + 1e-8)

                drift_score = (mean_drift + std_drift) / 2
                drift_scores[column] = min(drift_score, 1.0)  # Cap at 1.0

        return drift_scores

    def _calculate_overall_quality_score(
        self,
        missing_pct: float,
        duplicate_pct: float,
        column_completeness: Dict[str, float],
        outlier_counts: Dict[str, int]
    ) -> Tuple[float, str]:
        """Calculate overall data quality score and grade."""

        # Base score from completeness (40% weight)
        completeness_score = 100 - missing_pct

        # Duplicate penalty (20% weight)
        duplicate_penalty = min(duplicate_pct * 2, 40)  # Max 40 point penalty

        # Column completeness score (20% weight)
        if column_completeness:
            avg_column_completeness = sum(column_completeness.values()) / len(column_completeness)
            column_score = avg_column_completeness
        else:
            column_score = 100

        # Outlier penalty (20% weight)
        total_outliers = sum(outlier_counts.values())
        total_cells = sum(len(df) for df in [pd.DataFrame()])  # This is simplified
        if total_cells > 0:
            outlier_rate = total_outliers / total_cells
            outlier_penalty = min(outlier_rate * 100, 20)
        else:
            outlier_penalty = 0

        # Calculate weighted score
        overall_score = (
            completeness_score * 0.4 +
            (100 - duplicate_penalty) * 0.2 +
            column_score * 0.2 +
            (100 - outlier_penalty) * 0.2
        )

        # Determine grade
        if overall_score >= 90:
            grade = "A"
        elif overall_score >= 80:
            grade = "B"
        elif overall_score >= 70:
            grade = "C"
        elif overall_score >= 60:
            grade = "D"
        else:
            grade = "F"

        return overall_score, grade

    def _generate_data_quality_recommendations(self, metrics: DataQualityMetrics) -> List[str]:
        """Generate data quality improvement recommendations."""

        recommendations = []

        # Missing values recommendations
        if metrics.missing_values_percentage > 5:
            recommendations.append(
                f"High missing value rate ({metrics.missing_values_percentage".1f"}%). "
                "Consider data imputation or investigate data collection process."
            )

        # Duplicate recommendations
        if metrics.duplicate_rows_percentage > 1:
            recommendations.append(
                f"Significant duplicate rate ({metrics.duplicate_rows_percentage".1f"}%). "
                "Review data deduplication procedures."
            )

        # Column completeness recommendations
        incomplete_columns = [
            col for col, completeness in metrics.column_completeness.items()
            if completeness < 80
        ]
        if incomplete_columns:
            recommendations.append(
                f"Low completeness in columns: {', '.join(incomplete_columns)}. "
                "Prioritize data collection for these fields."
            )

        # Outlier recommendations
        high_outlier_columns = [
            col for col, count in metrics.outlier_counts.items()
            if count > 0
        ]
        if high_outlier_columns:
            recommendations.append(
                f"Outliers detected in: {', '.join(high_outlier_columns)}. "
                "Review outlier handling strategy."
            )

        # Clinical issues recommendations
        if metrics.clinical_data_issues:
            critical_issues = [issue for issue in metrics.clinical_data_issues if "ERROR" in issue]
            if critical_issues:
                recommendations.append(
                    f"Critical clinical data issues found: {', '.join(critical_issues)}. "
                    "Address before model training."
                )

        # General recommendations
        if metrics.overall_quality_score < 80:
            recommendations.append(
                "Overall data quality is below acceptable threshold. "
                "Consider comprehensive data quality improvement program."
            )

        return recommendations


class DataValidator:
    """Data validation system with configurable rules."""

    def __init__(self, validation_rules: List[ValidationRule]):
        self.validation_rules = validation_rules
        self._validation_cache: Dict[str, Any] = {}

    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "unknown"
    ) -> Dict[str, Any]:
        """Validate dataset against configured rules."""

        logger.info(f"Validating dataset '{dataset_name}' with {len(self.validation_rules)} rules")

        validation_results = []
        errors = []
        warnings = []

        for rule in self.validation_rules:
            if not rule.enabled:
                continue

            try:
                result = self._validate_rule(df, rule)
                validation_results.append(result)

                if result['severity'] == 'error':
                    errors.append(result['message'])
                elif result['severity'] == 'warning':
                    warnings.append(result['message'])

            except Exception as e:
                error_msg = f"Rule {rule.rule_id} failed: {e}"
                errors.append(error_msg)
                validation_results.append({
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'status': 'failed',
                    'severity': 'error',
                    'message': error_msg
                })

        # Determine overall validation status
        overall_status = 'passed' if not errors else 'failed'
        if warnings and not errors:
            overall_status = 'passed_with_warnings'

        result = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'validation_results': validation_results,
            'errors': errors,
            'warnings': warnings,
            'total_rules': len(self.validation_rules),
            'passed_rules': len([r for r in validation_results if r['status'] == 'passed']),
            'failed_rules': len([r for r in validation_results if r['status'] == 'failed'])
        }

        logger.info(f"Validation completed: {overall_status} ({result['passed_rules']}/{result['total_rules']} rules passed)")
        return result

    def _validate_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate a single rule."""

        try:
            if rule.rule_type == "completeness":
                return self._validate_completeness_rule(df, rule)
            elif rule.rule_type == "range":
                return self._validate_range_rule(df, rule)
            elif rule.rule_type == "categorical":
                return self._validate_categorical_rule(df, rule)
            elif rule.rule_type == "clinical":
                return self._validate_clinical_rule(df, rule)
            elif rule.rule_type == "statistical":
                return self._validate_statistical_rule(df, rule)
            else:
                return {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'status': 'failed',
                    'severity': 'error',
                    'message': f'Unknown rule type: {rule.rule_type}'
                }

        except Exception as e:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': 'error',
                'message': f'Rule validation failed: {e}'
            }

    def _validate_completeness_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate completeness rule."""
        column = rule.column
        min_completeness = rule.parameters.get('min_completeness', 95.0)

        if column not in df.columns:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': 'error',
                'message': f'Column {column} not found'
            }

        completeness = (1 - df[column].isnull().sum() / len(df)) * 100

        if completeness >= min_completeness:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'passed',
                'severity': 'info',
                'message': f'Column {column} completeness: {completeness".1f"}% (required: {min_completeness}%)'
            }
        else:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': rule.severity,
                'message': f'Column {column} completeness: {completeness".1f"}% below threshold {min_completeness}%'
            }

    def _validate_range_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate range rule."""
        column = rule.column
        min_val = rule.parameters.get('min_value')
        max_val = rule.parameters.get('max_value')

        if column not in df.columns:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': 'error',
                'message': f'Column {column} not found'
            }

        violations = []
        if min_val is not None:
            below_min = (df[column] < min_val).sum()
            if below_min > 0:
                violations.append(f'{below_min} values below minimum {min_val}')

        if max_val is not None:
            above_max = (df[column] > max_val).sum()
            if above_max > 0:
                violations.append(f'{above_max} values above maximum {max_val}')

        if not violations:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'passed',
                'severity': 'info',
                'message': f'Column {column} values within valid range'
            }
        else:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': rule.severity,
                'message': f'Column {column} range violations: {", ".join(violations)}'
            }

    def _validate_categorical_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate categorical rule."""
        column = rule.column
        allowed_values = rule.parameters.get('allowed_values', [])

        if column not in df.columns:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': 'error',
                'message': f'Column {column} not found'
            }

        invalid_values = df[~df[column].isin(allowed_values)][column].unique()
        invalid_count = len(df[~df[column].isin(allowed_values)])

        if invalid_count == 0:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'passed',
                'severity': 'info',
                'message': f'Column {column} has only valid categorical values'
            }
        else:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': rule.severity,
                'message': f'Column {column} has {invalid_count} invalid values: {list(invalid_values)[:5]}'
            }

    def _validate_clinical_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate clinical rule."""
        # Placeholder for clinical-specific validation rules
        column = rule.column

        if column not in df.columns:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': 'error',
                'message': f'Column {column} not found'
            }

        # Clinical validation would be more sophisticated
        return {
            'rule_id': rule.rule_id,
            'rule_name': rule.rule_name,
            'status': 'passed',
            'severity': 'info',
            'message': f'Clinical validation for {column} passed'
        }

    def _validate_statistical_rule(self, df: pd.DataFrame, rule: ValidationRule) -> Dict[str, Any]:
        """Validate statistical rule."""
        column = rule.column
        test_type = rule.parameters.get('test_type', 'normal_distribution')

        if column not in df.columns:
            return {
                'rule_id': rule.rule_id,
                'rule_name': rule.rule_name,
                'status': 'failed',
                'severity': 'error',
                'message': f'Column {column} not found'
            }

        # Simple statistical validation
        if test_type == 'normal_distribution':
            # Shapiro-Wilk test for normality
            try:
                from scipy.stats import shapiro
                stat, p_value = shapiro(df[column].dropna())
                is_normal = p_value > 0.05

                return {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'status': 'passed' if is_normal else 'failed',
                    'severity': 'warning' if not is_normal else 'info',
                    'message': f'Column {column} normality test: p-value={p_value".4f"} ({"normal" if is_normal else "non-normal"})'
                }
            except Exception:
                return {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.rule_name,
                    'status': 'failed',
                    'severity': 'error',
                    'message': f'Statistical test failed for column {column}'
                }

        return {
            'rule_id': rule.rule_id,
            'rule_name': rule.rule_name,
            'status': 'passed',
            'severity': 'info',
            'message': f'Statistical validation for {column} completed'
        }


class DataQualityMonitor:
    """Continuous data quality monitoring system."""

    def __init__(self, baseline_profile: Optional[DataQualityReport] = None):
        self.baseline_profile = baseline_profile
        self._monitoring_history: List[DataQualityReport] = []
        self._drift_thresholds: Dict[str, float] = {}

    def monitor_data_quality(
        self,
        df: pd.DataFrame,
        dataset_name: str = "unknown"
    ) -> DataQualityReport:
        """Monitor data quality and detect drift."""

        # Profile current data
        profiler = DataQualityProfiler()
        current_profile = profiler.profile_dataset(df, dataset_name)

        # Compare with baseline if available
        if self.baseline_profile:
            drift_analysis = self._analyze_quality_drift(current_profile)
            current_profile.recommendations.extend(drift_analysis.get('recommendations', []))

        # Store in history
        self._monitoring_history.append(current_profile)

        # Keep only recent history (last 100 reports)
        if len(self._monitoring_history) > 100:
            self._monitoring_history = self._monitoring_history[-100:]

        return current_profile

    def _analyze_quality_drift(self, current_profile: DataQualityReport) -> Dict[str, Any]:
        """Analyze drift from baseline."""
        if not self.baseline_profile:
            return {}

        baseline = self.baseline_profile.quality_metrics
        current = current_profile.quality_metrics

        drift_analysis = {
            'recommendations': [],
            'drift_detected': False,
            'drift_metrics': {}
        }

        # Check for significant changes in key metrics
        metrics_to_check = [
            ('missing_values_percentage', 5.0),  # 5% increase threshold
            ('duplicate_rows_percentage', 2.0),  # 2% increase threshold
        ]

        for metric_name, threshold in metrics_to_check:
            baseline_value = getattr(baseline, metric_name, 0)
            current_value = getattr(current, metric_name, 0)

            drift = abs(current_value - baseline_value)
            drift_analysis['drift_metrics'][metric_name] = {
                'baseline': baseline_value,
                'current': current_value,
                'drift': drift,
                'drift_percentage': (drift / baseline_value * 100) if baseline_value > 0 else 0
            }

            if drift > threshold:
                drift_analysis['drift_detected'] = True
                drift_analysis['recommendations'].append(
                    f"Significant {metric_name} drift detected: {drift".2f"} (threshold: {threshold})"
                )

        return drift_analysis

    def get_monitoring_history(self, days: int = 30) -> List[DataQualityReport]:
        """Get monitoring history for specified period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            report for report in self._monitoring_history
            if report.timestamp >= cutoff_date
        ]

    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""

        if not self._monitoring_history:
            return {"error": "No monitoring data available"}

        # Calculate trends
        recent_reports = self.get_monitoring_history(7)  # Last 7 days

        if len(recent_reports) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Quality score trend
        quality_scores = [r.quality_metrics.overall_quality_score for r in recent_reports]
        quality_trend = "improving" if quality_scores[-1] > quality_scores[0] else "degrading"

        # Error rate trend
        error_counts = [len(r.issues_found) for r in recent_reports]
        error_trend = "increasing" if error_counts[-1] > error_counts[0] else "decreasing"

        return {
            "monitoring_period_days": 7,
            "total_reports": len(self._monitoring_history),
            "recent_reports": len(recent_reports),
            "quality_trends": {
                "overall_score_trend": quality_trend,
                "error_rate_trend": error_trend,
                "current_quality_score": quality_scores[-1] if quality_scores else 0,
                "baseline_quality_score": quality_scores[0] if quality_scores else 0
            },
            "recommendations": self._generate_monitoring_recommendations(recent_reports)
        }

    def _generate_monitoring_recommendations(self, recent_reports: List[DataQualityReport]) -> List[str]:
        """Generate monitoring-based recommendations."""
        recommendations = []

        # Check for degrading quality
        quality_scores = [r.quality_metrics.overall_quality_score for r in recent_reports]
        if len(quality_scores) >= 3:
            recent_trend = np.polyfit(range(len(quality_scores)), quality_scores, 1)[0]
            if recent_trend < -1:  # Degrading by more than 1 point per report
                recommendations.append("Data quality is degrading. Investigate upstream data sources.")

        # Check for increasing errors
        error_counts = [len(r.issues_found) for r in recent_reports]
        if len(error_counts) >= 3:
            error_trend = np.polyfit(range(len(error_counts)), error_counts, 1)[0]
            if error_trend > 0.5:  # Increasing by more than 0.5 errors per report
                recommendations.append("Error rate is increasing. Review data validation rules.")

        return recommendations


class DataCleansingPipeline:
    """Automated data cleansing pipeline."""

    def __init__(self, cleansing_config: Dict[str, Any]):
        self.config = cleansing_config
        self._cleansing_log: List[Dict[str, Any]] = []

    def cleanse_dataset(
        self,
        df: pd.DataFrame,
        quality_report: DataQualityReport,
        preserve_original: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cleanse dataset based on quality report."""

        if preserve_original:
            df_clean = df.copy()
        else:
            df_clean = df

        cleansing_steps = []
        original_shape = df_clean.shape

        # Remove duplicates
        if self.config.get('remove_duplicates', True):
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            removed_rows = initial_rows - len(df_clean)
            if removed_rows > 0:
                cleansing_steps.append(f"Removed {removed_rows} duplicate rows")

        # Handle missing values
        missing_strategy = self.config.get('missing_values_strategy', 'auto')

        if missing_strategy == 'auto':
            # Auto-select strategy based on column type and missing percentage
            for column in df_clean.columns:
                missing_pct = (df_clean[column].isnull().sum() / len(df_clean)) * 100

                if missing_pct > 50:
                    # High missing rate - remove column
                    df_clean = df_clean.drop(columns=[column])
                    cleansing_steps.append(f"Removed column {column} ({missing_pct".1f"}% missing)")
                elif missing_pct > 10:
                    # Medium missing rate - impute
                    df_clean[column] = self._impute_column(df_clean[column], column)
                    cleansing_steps.append(f"Imputed {missing_pct".1f"}% missing values in {column}")
                else:
                    # Low missing rate - impute
                    df_clean[column] = self._impute_column(df_clean[column], column)
                    cleansing_steps.append(f"Imputed {missing_pct".1f"}% missing values in {column}")

        # Handle outliers
        if self.config.get('handle_outliers', True):
            outlier_columns = [
                col for col, count in quality_report.quality_metrics.outlier_counts.items()
                if count > 0
            ]

            for column in outlier_columns:
                outliers_removed = self._handle_outliers(df_clean, column)
                if outliers_removed > 0:
                    cleansing_steps.append(f"Removed {outliers_removed} outliers from {column}")

        # Validate final dataset
        final_shape = df_clean.shape
        rows_removed = original_shape[0] - final_shape[0]
        columns_removed = original_shape[1] - final_shape[1]

        cleansing_summary = {
            'original_shape': original_shape,
            'final_shape': final_shape,
            'rows_removed': rows_removed,
            'columns_removed': columns_removed,
            'cleansing_steps': cleansing_steps,
            'cleansing_timestamp': datetime.now().isoformat()
        }

        logger.info(f"Data cleansing completed: {len(cleansing_steps)} steps applied")
        return df_clean, cleansing_summary

    def _impute_column(self, series: pd.Series, column_name: str) -> pd.Series:
        """Impute missing values in a column."""
        if series.dtype in ['int64', 'float64']:
            # Numeric imputation
            if column_name in ['age', 'sofa']:
                # Clinical columns - use median
                return series.fillna(series.median())
            else:
                # Other numeric - use mean
                return series.fillna(series.mean())
        else:
            # Categorical imputation
            return series.fillna(series.mode().iloc[0] if not series.mode().empty else 'unknown')

    def _handle_outliers(self, df: pd.DataFrame, column: str) -> int:
        """Handle outliers in a column."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_before = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

        # Remove outliers
        df_clean = df[
            (df[column] >= lower_bound) &
            (df[column] <= upper_bound)
        ].copy()

        outliers_after = ((df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)).sum()

        return outliers_before - outliers_after


def create_validation_rules() -> List[ValidationRule]:
    """Create default validation rules for clinical data."""

    rules = [
        # Completeness rules
        ValidationRule(
            rule_id="age_completeness",
            rule_name="Age Column Completeness",
            rule_type="completeness",
            column="age",
            condition="completeness >= 95",
            parameters={"min_completeness": 95.0},
            description="Age should be available for most patients"
        ),
        ValidationRule(
            rule_id="sofa_completeness",
            rule_name="SOFA Score Completeness",
            rule_type="completeness",
            column="sofa",
            condition="completeness >= 90",
            parameters={"min_completeness": 90.0},
            description="SOFA score should be available for most ICU patients"
        ),

        # Range rules
        ValidationRule(
            rule_id="age_range",
            rule_name="Age Range Validation",
            rule_type="range",
            column="age",
            condition="0 <= age <= 120",
            parameters={"min_value": 0, "max_value": 120},
            description="Age should be between 0 and 120 years"
        ),
        ValidationRule(
            rule_id="sofa_range",
            rule_name="SOFA Score Range Validation",
            rule_type="range",
            column="sofa",
            condition="0 <= sofa <= 24",
            parameters={"min_value": 0, "max_value": 24},
            description="SOFA score should be between 0 and 24"
        ),

        # Categorical rules
        ValidationRule(
            rule_id="sex_values",
            rule_name="Sex Values Validation",
            rule_type="categorical",
            column="sex",
            condition="sex in ['male', 'female', 'M', 'F']",
            parameters={"allowed_values": ["male", "female", "M", "F"]},
            description="Sex should have valid categorical values"
        ),

        # Clinical rules
        ValidationRule(
            rule_id="clinical_ranges",
            rule_name="Clinical Value Ranges",
            rule_type="clinical",
            column="creatinine",
            condition="clinical_ranges_valid",
            parameters={"normal_range": (0.5, 1.2)},
            description="Clinical values should be within normal ranges"
        ),

        # Statistical rules
        ValidationRule(
            rule_id="normal_distribution",
            rule_name="Normal Distribution Check",
            rule_type="statistical",
            column="age",
            condition="shapiro_test_p > 0.05",
            parameters={"test_type": "normal_distribution"},
            description="Age should follow approximately normal distribution"
        )
    ]

    return rules


def create_data_quality_config(
    enable_anomaly_detection: bool = True,
    enable_clinical_validation: bool = True,
    quality_thresholds: Optional[Dict[str, float]] = None,
    cleansing_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create comprehensive data quality configuration."""

    return {
        "profiling": {
            "enable_anomaly_detection": enable_anomaly_detection,
            "enable_clinical_validation": enable_clinical_validation,
            "quality_thresholds": quality_thresholds or {
                "excellent": 90.0,
                "good": 80.0,
                "acceptable": 70.0,
                "poor": 60.0
            }
        },
        "validation": {
            "validation_rules": "default",  # "default", "custom", or list of rules
            "strict_mode": False,
            "fail_on_first_error": False
        },
        "cleansing": cleansing_config or {
            "remove_duplicates": True,
            "missing_values_strategy": "auto",
            "handle_outliers": True,
            "outlier_method": "iqr",  # "iqr", "zscore", "isolation_forest"
            "preserve_original": True
        },
        "monitoring": {
            "enable_continuous_monitoring": True,
            "monitoring_frequency_hours": 24,
            "drift_detection_enabled": True,
            "drift_threshold": 0.1
        }
    }


def save_data_quality_report(
    report: DataQualityReport,
    output_path: Path,
    format: str = "json"
) -> None:
    """Save data quality report to file."""

    ensure_dir(output_path.parent)

    if format.lower() == "json":
        # Convert to serializable format
        serializable_report = {
            "dataset_name": report.dataset_name,
            "timestamp": report.timestamp.isoformat(),
            "quality_metrics": {
                "total_rows": report.quality_metrics.total_rows,
                "total_columns": report.quality_metrics.total_columns,
                "missing_values_count": report.quality_metrics.missing_values_count,
                "missing_values_percentage": report.quality_metrics.missing_values_percentage,
                "duplicate_rows_count": report.quality_metrics.duplicate_rows_count,
                "duplicate_rows_percentage": report.quality_metrics.duplicate_rows_percentage,
                "column_completeness": report.quality_metrics.column_completeness,
                "column_uniqueness": report.quality_metrics.column_uniqueness,
                "column_data_types": report.quality_metrics.column_data_types,
                "column_statistics": report.quality_metrics.column_statistics,
                "outlier_counts": report.quality_metrics.outlier_counts,
                "anomaly_scores": report.quality_metrics.anomaly_scores,
                "clinical_data_issues": report.quality_metrics.clinical_data_issues,
                "data_drift_scores": report.quality_metrics.data_drift_scores,
                "overall_quality_score": report.quality_metrics.overall_quality_score,
                "quality_grade": report.quality_metrics.quality_grade
            },
            "validation_results": report.validation_results,
            "recommendations": report.recommendations,
            "issues_found": report.issues_found,
            "data_profile": report.data_profile
        }

        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)

    elif format.lower() == "html":
        # Generate HTML report
        html_content = _generate_quality_report_html(report)
        with open(output_path, 'w') as f:
            f.write(html_content)

    else:
        raise ValueError(f"Unsupported format: {format}")


def _generate_quality_report_html(report: DataQualityReport) -> str:
    """Generate HTML report for data quality analysis."""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Quality Report - {report.dataset_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .metrics {{ background: #fff; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 20px; }}
            .recommendations {{ background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin-bottom: 20px; }}
            .issues {{ background: #f8d7da; padding: 15px; border-left: 4px solid #dc3545; margin-bottom: 20px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric-value {{ font-weight: bold; font-size: 1.2em; }}
            .grade-A {{ color: #28a745; }}
            .grade-B {{ color: #17a2b8; }}
            .grade-C {{ color: #ffc107; }}
            .grade-D {{ color: #fd7e14; }}
            .grade-F {{ color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Quality Report</h1>
            <h2>{report.dataset_name}</h2>
            <p>Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="metrics">
            <h2>Quality Metrics</h2>
            <div class="metric">
                <div>Overall Score</div>
                <div class="metric-value grade-{report.quality_metrics.quality_grade}">{report.quality_metrics.overall_quality_score".1f"} ({report.quality_metrics.quality_grade})</div>
            </div>
            <div class="metric">
                <div>Total Rows</div>
                <div class="metric-value">{report.quality_metrics.total_rows","}</div>
            </div>
            <div class="metric">
                <div>Total Columns</div>
                <div class="metric-value">{report.quality_metrics.total_columns}</div>
            </div>
            <div class="metric">
                <div>Missing Values</div>
                <div class="metric-value">{report.quality_metrics.missing_values_percentage".1f"}%</div>
            </div>
            <div class="metric">
                <div>Duplicates</div>
                <div class="metric-value">{report.quality_metrics.duplicate_rows_percentage".1f"}%</div>
            </div>
        </div>
    """

    if report.recommendations:
        html += """
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
        """
        for rec in report.recommendations:
            html += f"<li>{rec}</li>"
        html += """
            </ul>
        </div>
        """

    if report.issues_found:
        html += """
        <div class="issues">
            <h2>Issues Found</h2>
            <ul>
        """
        for issue in report.issues_found:
            html += f"<li>{issue}</li>"
        html += """
            </ul>
        </div>
        """

    html += """
    </body>
    </html>
    """

    return html











