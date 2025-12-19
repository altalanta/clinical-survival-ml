"""
Continuous data quality monitoring service for production clinical pipelines.

This module provides:
- Continuous monitoring of data quality metrics
- Integration with Great Expectations for validation
- Alerting when data quality issues are detected
- Historical tracking of data quality over time
- Automated remediation suggestions
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum

import pandas as pd
import numpy as np

from clinical_survival.data_quality import (
    DataQualityValidator,
    DataQualityMetrics,
    DataQualityReport,
)
from clinical_survival.logging_config import get_logger
from clinical_survival.utils import ensure_dir

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class DataQualityAlert:
    """Represents a data quality alert."""

    alert_id: str
    timestamp: datetime
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    affected_columns: List[str] = field(default_factory=list)
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    remediation_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "affected_columns": self.affected_columns,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold_value": self.threshold_value,
            "remediation_suggestions": self.remediation_suggestions,
        }


@dataclass
class MonitoringRule:
    """A rule for monitoring data quality."""

    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., ">", "<", ">=", "<=", "==", "!="
    threshold: float
    severity: AlertSeverity
    enabled: bool = True
    affected_columns: Optional[List[str]] = None

    def evaluate(self, metrics: DataQualityMetrics) -> Optional[DataQualityAlert]:
        """
        Evaluate the rule against quality metrics.

        Returns an alert if the condition is met, None otherwise.
        """
        if not self.enabled:
            return None

        # Get the metric value
        metric_value = self._get_metric_value(metrics)
        if metric_value is None:
            return None

        # Check condition
        condition_met = self._check_condition(metric_value, self.condition, self.threshold)

        if condition_met:
            return DataQualityAlert(
                alert_id=f"{self.rule_id}_{int(time.time())}",
                timestamp=datetime.now(),
                severity=self.severity,
                status=AlertStatus.ACTIVE,
                title=f"Data Quality Alert: {self.name}",
                description=f"{self.description} - {self.metric_name} {self.condition} {self.threshold} (current: {metric_value:.3f})",
                affected_columns=self.affected_columns or [],
                metric_name=self.metric_name,
                metric_value=metric_value,
                threshold_value=self.threshold,
                remediation_suggestions=self._get_remediation_suggestions(),
            )

        return None

    def _get_metric_value(self, metrics: DataQualityMetrics) -> Optional[float]:
        """Extract the metric value from the metrics object."""
        if self.metric_name == "missing_values_percentage":
            return metrics.missing_values_percentage
        elif self.metric_name == "duplicate_rows_percentage":
            return metrics.duplicate_rows_percentage
        elif self.metric_name.startswith("column_completeness_"):
            col_name = self.metric_name.replace("column_completeness_", "")
            return metrics.column_completeness.get(col_name)
        elif self.metric_name.startswith("outlier_count_"):
            col_name = self.metric_name.replace("outlier_count_", "")
            return metrics.outlier_counts.get(col_name, 0)
        elif self.metric_name.startswith("anomaly_score_"):
            col_name = self.metric_name.replace("anomaly_score_", "")
            return metrics.anomaly_scores.get(col_name)

        return None

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if the condition is met."""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 1e-6
        elif condition == "!=":
            return abs(value - threshold) >= 1e-6
        return False

    def _get_remediation_suggestions(self) -> List[str]:
        """Get remediation suggestions based on the metric."""
        suggestions = []

        if "missing" in self.metric_name:
            suggestions.extend([
                "Consider imputation strategies (mean, median, or model-based)",
                "Review data collection process for missing data",
                "Implement data validation at source",
            ])
        elif "duplicate" in self.metric_name:
            suggestions.extend([
                "Implement deduplication logic in data pipeline",
                "Add unique constraints to data collection",
                "Review data merging processes for duplicates",
            ])
        elif "outlier" in self.metric_name:
            suggestions.extend([
                "Review outlier detection thresholds",
                "Consider robust statistical methods",
                "Validate outliers with domain experts",
            ])
        elif "anomaly" in self.metric_name:
            suggestions.extend([
                "Monitor for data drift or concept drift",
                "Review recent changes to data sources",
                "Consider retraining models if data distribution changed",
            ])

        return suggestions


@dataclass
class MonitoringConfiguration:
    """Configuration for data quality monitoring."""

    # Monitoring schedule
    check_interval_seconds: int = 3600  # 1 hour
    enable_continuous_monitoring: bool = True

    # Data sources
    data_source_path: Optional[Path] = None
    data_source_callback: Optional[Callable[[], pd.DataFrame]] = None

    # Storage
    monitoring_history_path: Path = Path("results/monitoring/data_quality_history.json")
    alerts_path: Path = Path("results/monitoring/alerts.json")

    # Alerting
    enable_alerts: bool = True
    alert_callbacks: List[Callable[[DataQualityAlert], None]] = field(default_factory=list)

    # Rules
    monitoring_rules: List[MonitoringRule] = field(default_factory=lambda: [
        # Default rules
        MonitoringRule(
            rule_id="missing_values_high",
            name="High Missing Values",
            description="Overall missing values percentage is too high",
            metric_name="missing_values_percentage",
            condition=">",
            threshold=5.0,
            severity=AlertSeverity.HIGH,
        ),
        MonitoringRule(
            rule_id="duplicates_high",
            name="High Duplicate Rows",
            description="Duplicate rows percentage is too high",
            metric_name="duplicate_rows_percentage",
            condition=">",
            threshold=1.0,
            severity=AlertSeverity.MEDIUM,
        ),
        MonitoringRule(
            rule_id="completeness_low",
            name="Low Column Completeness",
            description="Column completeness is below threshold",
            metric_name="column_completeness_time",  # Will be templated
            condition="<",
            threshold=95.0,
            severity=AlertSeverity.MEDIUM,
            affected_columns=["time"],
        ),
    ])

    # Great Expectations integration
    enable_ge: bool = False
    ge_context_path: Optional[Path] = None


class DataQualityMonitor:
    """
    Continuous data quality monitoring service.

    Monitors data quality metrics over time and alerts on issues.
    """

    def __init__(self, config: MonitoringConfiguration):
        self.config = config
        self.logger = get_logger(f"{__name__}.DataQualityMonitor")

        # Initialize storage
        ensure_dir(self.config.monitoring_history_path.parent)
        ensure_dir(self.config.alerts_path.parent)

        # Load existing data
        self.monitoring_history: List[Dict[str, Any]] = self._load_history()
        self.active_alerts: List[DataQualityAlert] = self._load_alerts()

        # Initialize validator
        self.validator = DataQualityValidator()

        # Monitoring state
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        self.last_check_time: Optional[datetime] = None

        # Great Expectations setup
        self.ge_context = None
        if self.config.enable_ge and self.config.ge_context_path:
            self._setup_ge()

        self.logger.info("Data quality monitor initialized")

    def _setup_ge(self) -> None:
        """Setup Great Expectations context."""
        try:
            import great_expectations as ge
            from great_expectations.data_context import FileDataContext

            if self.config.ge_context_path.exists():
                self.ge_context = FileDataContext.load(str(self.config.ge_context_path))
                self.logger.info("Great Expectations context loaded")
            else:
                self.logger.warning(f"GE context path does not exist: {self.config.ge_context_path}")
        except ImportError:
            self.logger.warning("Great Expectations not available")
        except Exception as e:
            self.logger.error(f"Failed to setup Great Expectations: {e}")

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load monitoring history from disk."""
        if self.config.monitoring_history_path.exists():
            try:
                with open(self.config.monitoring_history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load monitoring history: {e}")
        return []

    def _save_history(self) -> None:
        """Save monitoring history to disk."""
        try:
            with open(self.config.monitoring_history_path, 'w') as f:
                json.dump(self.monitoring_history, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save monitoring history: {e}")

    def _load_alerts(self) -> List[DataQualityAlert]:
        """Load active alerts from disk."""
        alerts = []
        if self.config.alerts_path.exists():
            try:
                with open(self.config.alerts_path, 'r') as f:
                    alerts_data = json.load(f)
                    for alert_data in alerts_data:
                        # Convert back to DataQualityAlert
                        alert_data["timestamp"] = datetime.fromisoformat(alert_data["timestamp"])
                        alert_data["severity"] = AlertSeverity(alert_data["severity"])
                        alert_data["status"] = AlertStatus(alert_data["status"])
                        alerts.append(DataQualityAlert(**alert_data))
            except Exception as e:
                self.logger.warning(f"Failed to load alerts: {e}")
        return alerts

    def _save_alerts(self) -> None:
        """Save alerts to disk."""
        try:
            alerts_data = [alert.to_dict() for alert in self.active_alerts]
            with open(self.config.alerts_path, 'w') as f:
                json.dump(alerts_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save alerts: {e}")

    def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Data quality monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)

        self.logger.info("Data quality monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                self.check_data_quality()
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.config.check_interval_seconds)

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform a data quality check and return results.

        Returns:
            Dictionary with quality metrics and any alerts generated
        """
        self.logger.debug("Performing data quality check")

        # Get data
        df = self._get_current_data()
        if df is None or df.empty:
            self.logger.warning("No data available for quality check")
            return {"error": "No data available"}

        # Perform quality validation
        quality_report = self.validator.validate_data_quality(df)

        # Store in history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": quality_report.metrics.__dict__,
            "issues": quality_report.issues,
            "recommendations": quality_report.recommendations,
        }
        self.monitoring_history.append(history_entry)

        # Keep only last 1000 entries
        if len(self.monitoring_history) > 1000:
            self.monitoring_history = self.monitoring_history[-1000:]

        self._save_history()

        # Check rules and generate alerts
        alerts = self._evaluate_rules(quality_report.metrics)

        # Run Great Expectations if enabled
        ge_results = None
        if self.ge_context:
            ge_results = self._run_ge_validations(df)

        # Update last check time
        self.last_check_time = datetime.now()

        result = {
            "timestamp": self.last_check_time.isoformat(),
            "metrics": quality_report.metrics.__dict__,
            "issues": quality_report.issues,
            "recommendations": quality_report.recommendations,
            "alerts_generated": len(alerts),
            "ge_results": ge_results,
        }

        self.logger.info(f"Data quality check completed. Generated {len(alerts)} alerts.")
        return result

    def _get_current_data(self) -> Optional[pd.DataFrame]:
        """Get the current data to monitor."""
        if self.config.data_source_callback:
            try:
                return self.config.data_source_callback()
            except Exception as e:
                self.logger.error(f"Error getting data from callback: {e}")
                return None

        elif self.config.data_source_path and self.config.data_source_path.exists():
            try:
                if self.config.data_source_path.suffix == '.csv':
                    return pd.read_csv(self.config.data_source_path)
                elif self.config.data_source_path.suffix == '.parquet':
                    return pd.read_parquet(self.config.data_source_path)
                elif self.config.data_source_path.suffix in ['.json', '.jsonl']:
                    return pd.read_json(self.config.data_source_path)
                else:
                    self.logger.warning(f"Unsupported file format: {self.config.data_source_path.suffix}")
                    return None
            except Exception as e:
                self.logger.error(f"Error reading data file: {e}")
                return None

        return None

    def _evaluate_rules(self, metrics: DataQualityMetrics) -> List[DataQualityAlert]:
        """Evaluate all monitoring rules and generate alerts."""
        alerts = []

        for rule in self.config.monitoring_rules:
            # Handle templated rules (e.g., column-specific rules)
            if rule.metric_name.endswith("_time") and "time" in str(rule.affected_columns):
                rule_copy = MonitoringRule(
                    rule_id=f"{rule.rule_id}_time",
                    name=rule.name,
                    description=rule.description,
                    metric_name="column_completeness_time",
                    condition=rule.condition,
                    threshold=rule.threshold,
                    severity=rule.severity,
                    enabled=rule.enabled,
                    affected_columns=rule.affected_columns,
                )
                alert = rule_copy.evaluate(metrics)
                if alert:
                    alerts.append(alert)
            else:
                alert = rule.evaluate(metrics)
                if alert:
                    alerts.append(alert)

        # Add to active alerts
        for alert in alerts:
            self.active_alerts.append(alert)

            # Trigger alert callbacks
            if self.config.enable_alerts:
                self._trigger_alert_callbacks(alert)

        # Save alerts
        if alerts:
            self._save_alerts()

        return alerts

    def _trigger_alert_callbacks(self, alert: DataQualityAlert) -> None:
        """Trigger alert callbacks."""
        for callback in self.config.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")

    def _run_ge_validations(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Run Great Expectations validations."""
        if not self.ge_context:
            return None

        try:
            # This is a simplified GE integration
            # In a real implementation, you'd have pre-defined expectations
            batch = self.ge_context.get_batch(df, "monitoring_batch")

            results = self.ge_context.run_checkpoint(
                checkpoint_name="data_quality_checkpoint",
                batch_request=batch,
            )

            return {
                "success": results.success,
                "statistics": results.run_results.get("statistics", {}),
                "validation_results": [
                    {
                        "expectation_type": vr.expectation_config.expectation_type,
                        "success": vr.success,
                        "result": vr.result,
                    }
                    for vr in results.run_results.get("validation_results", [])
                ]
            }

        except Exception as e:
            self.logger.error(f"Great Expectations validation failed: {e}")
            return {"error": str(e)}

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "is_monitoring": self.is_monitoring,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "active_alerts_count": len([a for a in self.active_alerts if a.status == AlertStatus.ACTIVE]),
            "total_alerts_count": len(self.active_alerts),
            "history_entries_count": len(self.monitoring_history),
            "great_expectations_enabled": self.ge_context is not None,
        }

    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 50,
    ) -> List[DataQualityAlert]:
        """Get alerts with optional filtering."""
        alerts = self.active_alerts

        if status:
            alerts = [a for a in alerts if a.status == status]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)

        return alerts[:limit]

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                self.logger.info(f"Alert {alert_id} acknowledged by {user}")
                self._save_alerts()
                return True
        return False

    def resolve_alert(self, alert_id: str, user: str) -> bool:
        """Resolve an alert."""
        for alert in self.active_alerts:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.RESOLVED
                self.logger.info(f"Alert {alert_id} resolved by {user}")
                self._save_alerts()
                return True
        return False

    def get_quality_trends(
        self,
        metric_name: str,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get quality metric trends over time."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        trends = []
        for entry in self.monitoring_history:
            entry_time = datetime.fromisoformat(entry["timestamp"])
            if entry_time >= cutoff_time:
                metrics = entry["metrics"]
                if metric_name in metrics:
                    trends.append({
                        "timestamp": entry_time,
                        "value": metrics[metric_name],
                    })

        return sorted(trends, key=lambda x: x["timestamp"])


# Convenience functions
def create_default_monitor(
    data_source_path: Optional[Path] = None,
    data_callback: Optional[Callable[[], pd.DataFrame]] = None,
    enable_ge: bool = False,
) -> DataQualityMonitor:
    """Create a monitor with default configuration."""
    config = MonitoringConfiguration(
        data_source_path=data_source_path,
        data_source_callback=data_callback,
        enable_ge=enable_ge,
    )
    return DataQualityMonitor(config)


def alert_to_slack_webhook(alert: DataQualityAlert, webhook_url: str) -> None:
    """Send alert to Slack webhook."""
    import requests

    payload = {
        "text": f"🚨 Data Quality Alert: {alert.title}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 {alert.severity.value.upper()} Alert: {alert.title}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{alert.description}*\n\n**Metric:** {alert.metric_name}\n**Value:** {alert.metric_value}\n**Threshold:** {alert.threshold_value}"
                }
            }
        ]
    }

    try:
        requests.post(webhook_url, json=payload)
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


def alert_to_email(alert: DataQualityAlert, smtp_config: Dict[str, Any]) -> None:
    """Send alert via email."""
    import smtplib
    from email.mime.text import MIMEText

    msg = MIMEText(f"""
Data Quality Alert: {alert.title}

{alert.description}

Metric: {alert.metric_name}
Value: {alert.metric_value}
Threshold: {alert.threshold_value}

Remediation Suggestions:
{chr(10).join(f"- {s}" for s in alert.remediation_suggestions)}
    """)

    msg['Subject'] = f"Data Quality Alert: {alert.title}"
    msg['From'] = smtp_config['from_email']
    msg['To'] = smtp_config['to_email']

    try:
        server = smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port'])
        server.starttls()
        server.login(smtp_config['username'], smtp_config['password'])
        server.sendmail(smtp_config['from_email'], smtp_config['to_email'], msg.as_string())
        server.quit()
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")

