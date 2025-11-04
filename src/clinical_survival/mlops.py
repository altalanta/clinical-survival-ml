"""Automated MLOps pipeline and model lifecycle management for clinical survival models."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from clinical_survival.logging_utils import log_function_call
from clinical_survival.models import BaseSurvivalModel
from clinical_survival.utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Model version metadata and lifecycle information."""

    version_id: str
    model_name: str
    model_type: str
    version_number: str  # e.g., "1.0.0"
    created_at: datetime
    created_by: str
    parent_version: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Model artifacts
    model_path: Path = field(default_factory=Path)
    preprocessor_path: Optional[Path] = None
    feature_names: List[str] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Lifecycle status
    status: str = "development"  # "development", "staging", "production", "archived"
    approval_status: str = "pending"  # "pending", "approved", "rejected"
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None

    # Deployment information
    deployed_at: Optional[datetime] = None
    deployed_to: List[str] = field(default_factory=list)  # environments
    traffic_percentage: float = 0.0

    # Monitoring
    monitoring_enabled: bool = True
    drift_detected: bool = False
    performance_degraded: bool = False


@dataclass
class DeploymentEnvironment:
    """Configuration for deployment environments."""

    name: str
    type: str  # "staging", "production", "development"
    url: Optional[str] = None
    api_key: Optional[str] = None
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    auto_rollback: bool = True
    rollback_threshold: float = 0.05  # Performance degradation threshold for rollback


@dataclass
class ABTestConfig:
    """Configuration for A/B testing model versions."""

    test_id: str
    test_name: str
    model_versions: List[str]  # Version IDs to compare
    traffic_split: Dict[str, float]  # Version -> percentage
    test_duration_days: int = 14
    success_metrics: List[str] = field(default_factory=lambda: ["concordance", "ibs"])
    statistical_significance_threshold: float = 0.05
    minimum_sample_size: int = 1000


@dataclass
class RetrainingTrigger:
    """Configuration for automated retraining triggers."""

    trigger_id: str
    trigger_name: str
    trigger_type: str  # "scheduled", "performance", "drift", "data_volume"
    enabled: bool = True

    # Trigger conditions
    schedule_cron: Optional[str] = None  # For scheduled triggers
    performance_threshold: Optional[float] = None  # For performance triggers
    drift_threshold: Optional[float] = None  # For drift triggers
    min_new_samples: Optional[int] = None  # For data volume triggers

    # Actions
    notify_on_trigger: bool = True
    auto_retrain: bool = False
    require_approval: bool = True


class ModelRegistry:
    """Central registry for model versions and lifecycle management."""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.models_file = registry_path / "models.json"
        self.versions_file = registry_path / "versions.json"
        self._models: Dict[str, Dict[str, Any]] = {}
        self._versions: Dict[str, ModelVersion] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load existing registry data."""
        try:
            if self.models_file.exists():
                with open(self.models_file) as f:
                    self._models = json.load(f)

            if self.versions_file.exists():
                with open(self.versions_file) as f:
                    versions_data = json.load(f)

                for version_id, version_data in versions_data.items():
                    version = ModelVersion(
                        version_id=version_id,
                        model_name=version_data["model_name"],
                        model_type=version_data["model_type"],
                        version_number=version_data["version_number"],
                        created_at=datetime.fromisoformat(version_data["created_at"]),
                        created_by=version_data["created_by"],
                        parent_version=version_data.get("parent_version"),
                        description=version_data.get("description", ""),
                        tags=version_data.get("tags", []),
                        performance_metrics=version_data.get("performance_metrics", {}),
                        validation_metrics=version_data.get("validation_metrics", {}),
                        model_path=Path(version_data.get("model_path", "")),
                        preprocessor_path=Path(version_data["preprocessor_path"]) if version_data.get("preprocessor_path") else None,
                        feature_names=version_data.get("feature_names", []),
                        hyperparameters=version_data.get("hyperparameters", {}),
                        status=version_data.get("status", "development"),
                        approval_status=version_data.get("approval_status", "pending"),
                        approved_by=version_data.get("approved_by"),
                        approved_at=datetime.fromisoformat(version_data["approved_at"]) if version_data.get("approved_at") else None,
                        deployed_at=datetime.fromisoformat(version_data["deployed_at"]) if version_data.get("deployed_at") else None,
                        deployed_to=version_data.get("deployed_to", []),
                        traffic_percentage=version_data.get("traffic_percentage", 0.0),
                        monitoring_enabled=version_data.get("monitoring_enabled", True),
                        drift_detected=version_data.get("drift_detected", False),
                        performance_degraded=version_data.get("performance_degraded", False)
                    )
                    self._versions[version_id] = version

        except Exception as e:
            logger.error(f"Failed to load model registry: {e}")
            self._models = {}
            self._versions = {}

    def _save_registry(self) -> None:
        """Save registry data to disk."""
        ensure_dir(self.registry_path)

        # Save models
        with open(self.models_file, 'w') as f:
            json.dump(self._models, f, indent=2, default=str)

        # Save versions
        versions_data = {}
        for version_id, version in self._versions.items():
            versions_data[version_id] = {
                "model_name": version.model_name,
                "model_type": version.model_type,
                "version_number": version.version_number,
                "created_at": version.created_at.isoformat(),
                "created_by": version.created_by,
                "parent_version": version.parent_version,
                "description": version.description,
                "tags": version.tags,
                "performance_metrics": version.performance_metrics,
                "validation_metrics": version.validation_metrics,
                "model_path": str(version.model_path),
                "preprocessor_path": str(version.preprocessor_path) if version.preprocessor_path else None,
                "feature_names": version.feature_names,
                "hyperparameters": version.hyperparameters,
                "status": version.status,
                "approval_status": version.approval_status,
                "approved_by": version.approved_by,
                "approved_at": version.approved_at.isoformat() if version.approved_at else None,
                "deployed_at": version.deployed_at.isoformat() if version.deployed_at else None,
                "deployed_to": version.deployed_to,
                "traffic_percentage": version.traffic_percentage,
                "monitoring_enabled": version.monitoring_enabled,
                "drift_detected": version.drift_detected,
                "performance_degraded": version.performance_degraded
            }

        with open(self.versions_file, 'w') as f:
            json.dump(versions_data, f, indent=2)

    def register_model(
        self,
        model: BaseSurvivalModel,
        model_name: str,
        version_number: str,
        created_by: str,
        description: str = "",
        tags: List[str] = None,
        parent_version: Optional[str] = None,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> ModelVersion:
        """Register a new model version."""

        version_id = str(uuid.uuid4())

        # Determine model type
        model_type = type(model).__name__

        # Get feature names if available
        feature_names = getattr(model, 'feature_names_', [])

        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            model_type=model_type,
            version_number=version_number,
            created_at=datetime.now(),
            created_by=created_by,
            parent_version=parent_version,
            description=description,
            tags=tags or [],
            model_path=model_path or Path(f"models/{model_name}_{version_id}.pkl"),
            feature_names=feature_names,
            hyperparameters=hyperparameters or {}
        )

        self._versions[version_id] = version

        # Update models index
        if model_name not in self._models:
            self._models[model_name] = {}

        self._models[model_name][version_number] = version_id

        self._save_registry()

        logger.info(f"Registered model {model_name} version {version_number} with ID {version_id}")
        return version

    def get_model_versions(self, model_name: str) -> List[ModelVersion]:
        """Get all versions of a specific model."""
        if model_name not in self._models:
            return []

        version_ids = []
        for version_num, version_id in self._models[model_name].items():
            version_ids.append(version_id)

        return [self._versions[vid] for vid in version_ids if vid in self._versions]

    def get_latest_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the latest version of a model."""
        versions = self.get_model_versions(model_name)
        if not versions:
            return None

        # Sort by creation time (most recent first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions[0]

    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._versions.get(version_id)

    def update_version_status(
        self,
        version_id: str,
        status: str,
        approved_by: Optional[str] = None
    ) -> bool:
        """Update the status of a model version."""
        if version_id not in self._versions:
            return False

        version = self._versions[version_id]
        version.status = status

        if status == "production":
            version.approval_status = "approved"
            version.approved_by = approved_by
            version.approved_at = datetime.now()
            version.deployed_at = datetime.now()
        elif status == "staging":
            version.approval_status = "approved"
            version.approved_by = approved_by
            version.approved_at = datetime.now()

        self._save_registry()
        logger.info(f"Updated model version {version_id} status to {status}")
        return True

    def promote_version(
        self,
        version_id: str,
        target_environment: str,
        approved_by: str
    ) -> bool:
        """Promote a model version to a target environment."""
        if version_id not in self._versions:
            return False

        version = self._versions[version_id]

        if target_environment not in version.deployed_to:
            version.deployed_to.append(target_environment)

        # Update status based on environment
        if target_environment == "production":
            return self.update_version_status(version_id, "production", approved_by)
        elif target_environment == "staging":
            return self.update_version_status(version_id, "staging", approved_by)
        else:
            version.deployed_at = datetime.now()
            self._save_registry()
            return True

    def record_performance(
        self,
        version_id: str,
        metrics: Dict[str, float],
        validation_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """Record performance metrics for a model version."""
        if version_id not in self._versions:
            return False

        version = self._versions[version_id]
        version.performance_metrics.update(metrics)

        if validation_metrics:
            version.validation_metrics.update(validation_metrics)

        self._save_registry()
        logger.info(f"Recorded performance metrics for version {version_id}")
        return True

    def mark_degraded(
        self,
        version_id: str,
        degradation_type: str  # "performance" or "drift"
    ) -> bool:
        """Mark a model version as degraded."""
        if version_id not in self._versions:
            return False

        version = self._versions[version_id]

        if degradation_type == "performance":
            version.performance_degraded = True
        elif degradation_type == "drift":
            version.drift_detected = True

        self._save_registry()
        logger.warning(f"Marked version {version_id} as degraded ({degradation_type})")
        return True


class AutomatedRetrainer:
    """Automated model retraining system."""

    def __init__(
        self,
        registry: ModelRegistry,
        triggers: List[RetrainingTrigger],
        default_config: Dict[str, Any]
    ):
        self.registry = registry
        self.triggers = triggers
        self.default_config = default_config
        self._last_check: Dict[str, datetime] = {}

    def check_triggers(self) -> List[str]:
        """Check all triggers and return list of triggered model names."""
        triggered_models = []

        for trigger in self.triggers:
            if not trigger.enabled:
                continue

            if trigger.trigger_type == "scheduled":
                if self._check_scheduled_trigger(trigger):
                    triggered_models.extend(self._get_models_for_trigger(trigger))

            elif trigger.trigger_type == "performance":
                if self._check_performance_trigger(trigger):
                    triggered_models.extend(self._get_models_for_trigger(trigger))

            elif trigger.trigger_type == "drift":
                if self._check_drift_trigger(trigger):
                    triggered_models.extend(self._get_models_for_trigger(trigger))

            elif trigger.trigger_type == "data_volume":
                if self._check_data_volume_trigger(trigger):
                    triggered_models.extend(self._get_models_for_trigger(trigger))

        return list(set(triggered_models))  # Remove duplicates

    def _check_scheduled_trigger(self, trigger: RetrainingTrigger) -> bool:
        """Check if scheduled trigger should fire."""
        # For now, implement simple time-based checking
        # In practice, this would use cron-like scheduling
        trigger_key = f"scheduled_{trigger.trigger_id}"

        if trigger_key not in self._last_check:
            self._last_check[trigger_key] = datetime.now()
            return True

        # Check if enough time has passed
        time_since_last = datetime.now() - self._last_check[trigger_key]
        # Simple check - would implement proper cron parsing
        if time_since_last.total_seconds() > 3600:  # Check every hour for now
            self._last_check[trigger_key] = datetime.now()
            return True

        return False

    def _check_performance_trigger(self, trigger: RetrainingTrigger) -> bool:
        """Check if performance trigger should fire."""
        # Check if any model has performance below threshold
        for model_name in self.registry._models.keys():
            latest_version = self.registry.get_latest_version(model_name)
            if latest_version and latest_version.status == "production":
                # Check if performance has degraded
                baseline_concordance = latest_version.performance_metrics.get("concordance", 1.0)
                current_concordance = baseline_concordance  # Would get from monitoring

                if current_concordance < baseline_concordance - (trigger.performance_threshold or 0.05):
                    logger.info(f"Performance trigger activated for {model_name}")
                    return True

        return False

    def _check_drift_trigger(self, trigger: RetrainingTrigger) -> bool:
        """Check if drift trigger should fire."""
        # Check if any model has detected drift
        for version in self.registry._versions.values():
            if version.status == "production" and version.drift_detected:
                drift_score = 0.1  # Would get actual drift score from monitoring

                if drift_score > (trigger.drift_threshold or 0.1):
                    logger.info(f"Drift trigger activated for version {version.version_id}")
                    return True

        return False

    def _check_data_volume_trigger(self, trigger: RetrainingTrigger) -> bool:
        """Check if data volume trigger should fire."""
        # Check if enough new data is available
        # This would integrate with data monitoring systems
        return False  # Placeholder

    def _get_models_for_trigger(self, trigger: RetrainingTrigger) -> List[str]:
        """Get list of models affected by a trigger."""
        # For now, return all models - would be more sophisticated
        return list(self.registry._models.keys())


class DeploymentManager:
    """Manage model deployments across environments."""

    def __init__(
        self,
        registry: ModelRegistry,
        environments: List[DeploymentEnvironment]
    ):
        self.registry = registry
        self.environments = {env.name: env for env in environments}
        self._active_deployments: Dict[str, Dict[str, Any]] = {}

    def deploy_model(
        self,
        version_id: str,
        environment_name: str,
        traffic_percentage: float = 100.0,
        approved_by: Optional[str] = None
    ) -> bool:
        """Deploy a model version to an environment."""

        if environment_name not in self.environments:
            logger.error(f"Environment {environment_name} not found")
            return False

        environment = self.environments[environment_name]

        if not self.registry.promote_version(version_id, environment_name, approved_by or "system"):
            return False

        # Update traffic allocation
        version = self.registry.get_version(version_id)
        if version:
            version.traffic_percentage = traffic_percentage

            # Record deployment
            deployment_id = str(uuid.uuid4())
            self._active_deployments[deployment_id] = {
                "version_id": version_id,
                "environment": environment_name,
                "deployed_at": datetime.now(),
                "traffic_percentage": traffic_percentage,
                "approved_by": approved_by
            }

            logger.info(f"Deployed model version {version_id} to {environment_name} with {traffic_percentage}% traffic")
            return True

        return False

    def rollback_deployment(
        self,
        environment_name: str,
        target_version_id: str,
        reason: str = "manual_rollback"
    ) -> bool:
        """Rollback a deployment to a previous version."""

        if environment_name not in self.environments:
            logger.error(f"Environment {environment_name} not found")
            return False

        # Find current deployment in this environment
        current_deployment = None
        for deployment in self._active_deployments.values():
            if deployment["environment"] == environment_name:
                current_deployment = deployment
                break

        if not current_deployment:
            logger.error(f"No active deployment found for environment {environment_name}")
            return False

        current_version_id = current_deployment["version_id"]

        # Switch traffic to target version
        target_version = self.registry.get_version(target_version_id)
        current_version = self.registry.get_version(current_version_id)

        if target_version and current_version:
            # Update traffic allocation
            target_version.traffic_percentage = current_version.traffic_percentage
            current_version.traffic_percentage = 0.0

            # Update deployment record
            current_deployment["version_id"] = target_version_id
            current_deployment["rolled_back_at"] = datetime.now()
            current_deployment["rollback_reason"] = reason

            logger.info(f"Rolled back deployment in {environment_name} from {current_version_id} to {target_version_id}")
            return True

        return False

    def get_deployment_status(self, environment_name: str) -> Dict[str, Any]:
        """Get deployment status for an environment."""
        if environment_name not in self.environments:
            return {"error": "Environment not found"}

        environment = self.environments[environment_name]

        # Find active deployment
        active_deployment = None
        for deployment in self._active_deployments.values():
            if deployment["environment"] == environment_name:
                active_deployment = deployment
                break

        if not active_deployment:
            return {
                "environment": environment_name,
                "status": "no_deployment",
                "active_versions": []
            }

        version = self.registry.get_version(active_deployment["version_id"])
        if not version:
            return {
                "environment": environment_name,
                "status": "deployment_error",
                "active_versions": []
            }

        return {
            "environment": environment_name,
            "status": "active",
            "current_version": {
                "version_id": version.version_id,
                "version_number": version.version_number,
                "deployed_at": active_deployment["deployed_at"],
                "traffic_percentage": active_deployment["traffic_percentage"],
                "performance_metrics": version.performance_metrics
            },
            "environment_config": {
                "auto_rollback": environment.auto_rollback,
                "rollback_threshold": environment.rollback_threshold
            }
        }


class ABTestManager:
    """Manage A/B testing for model versions."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._active_tests: Dict[str, ABTestConfig] = {}

    def create_ab_test(
        self,
        test_name: str,
        model_versions: List[str],
        traffic_split: Dict[str, float],
        test_duration_days: int = 14,
        success_metrics: List[str] = None
    ) -> str:
        """Create a new A/B test."""

        test_id = str(uuid.uuid4())

        # Validate that all versions exist
        for version_id in model_versions:
            if not self.registry.get_version(version_id):
                raise ValueError(f"Model version {version_id} not found")

        # Validate traffic split
        total_traffic = sum(traffic_split.values())
        if abs(total_traffic - 1.0) > 0.01:
            raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")

        test_config = ABTestConfig(
            test_id=test_id,
            test_name=test_name,
            model_versions=model_versions,
            traffic_split=traffic_split,
            test_duration_days=test_duration_days,
            success_metrics=success_metrics or ["concordance", "ibs"]
        )

        self._active_tests[test_id] = test_config

        # Update model traffic allocation
        for version_id, percentage in traffic_split.items():
            version = self.registry.get_version(version_id)
            if version:
                version.traffic_percentage = percentage

        self.registry._save_registry()

        logger.info(f"Created A/B test '{test_name}' with ID {test_id}")
        return test_id

    def get_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get results for an A/B test."""
        if test_id not in self._active_tests:
            return {"error": "Test not found"}

        test_config = self._active_tests[test_id]

        # Collect performance data for each version
        version_results = {}
        for version_id in test_config.model_versions:
            version = self.registry.get_version(version_id)
            if version:
                version_results[version_id] = {
                    "version_number": version.version_number,
                    "performance_metrics": version.performance_metrics,
                    "traffic_percentage": version.traffic_percentage,
                    "status": version.status
                }

        return {
            "test_id": test_id,
            "test_name": test_config.test_name,
            "status": "active",  # Would check if test is complete
            "start_date": datetime.now() - timedelta(days=test_config.test_duration_days),  # Placeholder
            "duration_days": test_config.test_duration_days,
            "version_results": version_results,
            "success_metrics": test_config.success_metrics,
            "statistical_significance": self._calculate_significance(version_results)
        }

    def _calculate_significance(self, version_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate statistical significance for A/B test results."""
        # Placeholder - would implement proper statistical testing
        return {
            "concordance_p_value": 0.05,  # Would calculate actual p-value
            "ibs_p_value": 0.03,
            "significant_difference": True
        }

    def conclude_test(self, test_id: str, winner_version_id: Optional[str] = None) -> bool:
        """Conclude an A/B test and optionally promote a winner."""

        if test_id not in self._active_tests:
            return False

        test_config = self._active_tests[test_id]

        if winner_version_id:
            # Promote winner to full production
            version = self.registry.get_version(winner_version_id)
            if version:
                version.traffic_percentage = 100.0

                # Demote other versions
                for version_id in test_config.model_versions:
                    if version_id != winner_version_id:
                        other_version = self.registry.get_version(version_id)
                        if other_version:
                            other_version.traffic_percentage = 0.0

        # Mark test as completed
        del self._active_tests[test_id]

        self.registry._save_registry()
        logger.info(f"Concluded A/B test {test_id}")
        return True


def create_mlops_config(
    registry_path: Path,
    environments: Optional[List[Dict[str, Any]]] = None,
    triggers: Optional[List[Dict[str, Any]]] = None,
    ab_test_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create MLOps configuration."""

    default_environments = [
        {
            "name": "development",
            "type": "development",
            "auto_rollback": False,
            "rollback_threshold": 0.1
        },
        {
            "name": "staging",
            "type": "staging",
            "auto_rollback": True,
            "rollback_threshold": 0.05
        },
        {
            "name": "production",
            "type": "production",
            "auto_rollback": True,
            "rollback_threshold": 0.02
        }
    ]

    default_triggers = [
        {
            "trigger_id": "daily_retrain",
            "trigger_name": "Daily Retraining",
            "trigger_type": "scheduled",
            "enabled": True,
            "schedule_cron": "0 2 * * *",  # Daily at 2 AM
            "require_approval": True
        },
        {
            "trigger_id": "performance_monitor",
            "trigger_name": "Performance Degradation",
            "trigger_type": "performance",
            "enabled": True,
            "performance_threshold": 0.05,
            "auto_retrain": False,
            "require_approval": True
        },
        {
            "trigger_id": "drift_detection",
            "trigger_name": "Data Drift",
            "trigger_type": "drift",
            "enabled": True,
            "drift_threshold": 0.1,
            "auto_retrain": False,
            "require_approval": True
        }
    ]

    return {
        "registry_path": str(registry_path),
        "environments": environments or default_environments,
        "triggers": triggers or default_triggers,
        "ab_test_config": ab_test_config or {},
        "deployment_settings": {
            "require_approval_for_production": True,
            "auto_rollback_on_failure": True,
            "max_concurrent_deployments": 3,
            "deployment_timeout_minutes": 30
        }
    }


def load_mlops_config(config_path: Path) -> Dict[str, Any]:
    """Load MLOps configuration from file."""
    try:
        with open(config_path) as f:
            config_data = json.load(f)

        return config_data

    except Exception as e:
        logger.error(f"Failed to load MLOps config: {e}")
        return create_mlops_config(Path("results/mlops"))  # Return default config


def initialize_mlops_system(config: Dict[str, Any]) -> Tuple[ModelRegistry, AutomatedRetrainer, DeploymentManager, ABTestManager]:
    """Initialize the complete MLOps system."""

    # Create registry
    registry_path = Path(config["registry_path"])
    registry = ModelRegistry(registry_path)

    # Create environments
    environments = [
        DeploymentEnvironment(**env_data)
        for env_data in config["environments"]
    ]
    deployment_manager = DeploymentManager(registry, environments)

    # Create triggers
    triggers = [
        RetrainingTrigger(**trigger_data)
        for trigger_data in config["triggers"]
    ]
    auto_retrainer = AutomatedRetrainer(registry, triggers, config)

    # Create A/B test manager
    ab_test_manager = ABTestManager(registry)

    logger.info("MLOps system initialized successfully")
    return registry, auto_retrainer, deployment_manager, ab_test_manager


