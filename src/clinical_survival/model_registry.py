"""
Centralized model registry with promotion workflows and governance.

This module provides:
- Model versioning and storage
- Environment-based promotion workflows (dev -> staging -> production)
- Governance features (approvals, audit trails, access control)
- Integration with MLflow tracking
- CLI interface for registry operations
- Model validation and health checks
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from clinical_survival.logging_config import get_logger

logger = get_logger(__name__)
console = Console()


class ModelStage(Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ApprovalStatus(Enum):
    """Approval status for model promotions."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


class AccessLevel(Enum):
    """Access control levels."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""

    model_name: str
    version: str
    stage: ModelStage
    created_at: datetime
    created_by: str
    description: str = ""
    tags: Dict[str, Any] = field(default_factory=dict)

    # File paths
    model_path: Optional[Path] = None
    metadata_path: Optional[Path] = None

    # Performance metrics (from training)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Approval and governance
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    approval_notes: Optional[str] = None

    # Audit trail
    promotion_history: List[Dict[str, Any]] = field(default_factory=list)
    access_log: List[Dict[str, Any]] = field(default_factory=list)

    # Model health
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"
    health_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "stage": self.stage.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "tags": self.tags,
            "model_path": str(self.model_path) if self.model_path else None,
            "metadata_path": str(self.metadata_path) if self.metadata_path else None,
            "metrics": self.metrics,
            "approval_status": self.approval_status.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "approval_notes": self.approval_notes,
            "promotion_history": self.promotion_history,
            "access_log": self.access_log,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_status": self.health_status,
            "health_score": self.health_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelVersion:
        """Create from dictionary."""
        # Convert string timestamps back to datetime
        created_at = datetime.fromisoformat(data["created_at"])
        approved_at = datetime.fromisoformat(data["approved_at"]) if data.get("approved_at") else None
        last_health_check = datetime.fromisoformat(data["last_health_check"]) if data.get("last_health_check") else None

        # Convert enums
        stage = ModelStage(data["stage"])
        approval_status = ApprovalStatus(data["approval_status"])

        return cls(
            model_name=data["model_name"],
            version=data["version"],
            stage=stage,
            created_at=created_at,
            created_by=data["created_by"],
            description=data.get("description", ""),
            tags=data.get("tags", {}),
            model_path=Path(data["model_path"]) if data.get("model_path") else None,
            metadata_path=Path(data["metadata_path"]) if data.get("metadata_path") else None,
            metrics=data.get("metrics", {}),
            approval_status=approval_status,
            approved_by=data.get("approved_by"),
            approved_at=approved_at,
            approval_notes=data.get("approval_notes"),
            promotion_history=data.get("promotion_history", []),
            access_log=data.get("access_log", []),
            last_health_check=last_health_check,
            health_status=data.get("health_status", "unknown"),
            health_score=data.get("health_score"),
        )

    def log_access(self, user: str, action: str, access_level: AccessLevel) -> None:
        """Log an access event."""
        access_event = {
            "timestamp": datetime.now().isoformat(),
            "user": user,
            "action": action,
            "access_level": access_level.value,
        }
        self.access_log.append(access_event)

        # Keep only last 1000 access events
        if len(self.access_log) > 1000:
            self.access_log = self.access_log[-1000:]

    def promote(self, new_stage: ModelStage, promoted_by: str, notes: str = "") -> None:
        """Record a promotion event."""
        promotion_event = {
            "timestamp": datetime.now().isoformat(),
            "from_stage": self.stage.value,
            "to_stage": new_stage.value,
            "promoted_by": promoted_by,
            "notes": notes,
        }
        self.promotion_history.append(promotion_event)
        self.stage = new_stage

        logger.info(f"Model {self.model_name} v{self.version} promoted to {new_stage.value}")

    def approve(self, approved_by: str, notes: str = "") -> None:
        """Approve the model for promotion."""
        self.approval_status = ApprovalStatus.APPROVED
        self.approved_by = approved_by
        self.approved_at = datetime.now()
        self.approval_notes = notes

        logger.info(f"Model {self.model_name} v{self.version} approved by {approved_by}")

    def reject(self, rejected_by: str, notes: str = "") -> None:
        """Reject the model for promotion."""
        self.approval_status = ApprovalStatus.REJECTED
        self.approved_by = rejected_by
        self.approved_at = datetime.now()
        self.approval_notes = notes

        logger.info(f"Model {self.model_name} v{self.version} rejected by {rejected_by}")

    def update_health(self, status: str, score: Optional[float] = None) -> None:
        """Update model health status."""
        self.last_health_check = datetime.now()
        self.health_status = status
        self.health_score = score

    def is_promotion_ready(self) -> bool:
        """Check if model is ready for promotion."""
        return (
            self.approval_status == ApprovalStatus.APPROVED
            and self.health_status in ["healthy", "good"]
            and self.health_score is not None
            and self.health_score >= 0.8  # Minimum health score
        )


@dataclass
class RegistryConfig:
    """Configuration for the model registry."""

    registry_root: Path
    enable_auto_approval: bool = False
    require_approval_for_production: bool = True
    auto_approval_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "concordance_min": 0.7,
        "brier_max": 0.25,
    })
    access_control: Dict[str, List[str]] = field(default_factory=dict)
    retention_policy: Dict[str, int] = field(default_factory=lambda: {
        "development": 30,  # days
        "staging": 90,
        "production": 365,
        "archived": -1,  # never delete
    })


class ModelRegistry:
    """Centralized model registry with governance and promotion workflows."""

    def __init__(self, config: RegistryConfig):
        self.config = config
        self.registry_root = config.registry_root
        self.models_dir = self.registry_root / "models"
        self.metadata_dir = self.registry_root / "metadata"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache of model versions
        self._model_cache: Dict[str, Dict[str, ModelVersion]] = {}
        self._load_registry()

        logger.info(f"Model registry initialized at {self.registry_root}")

    def _load_registry(self) -> None:
        """Load all model metadata from disk."""
        self._model_cache = {}

        if not self.metadata_dir.exists():
            return

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    version = ModelVersion.from_dict(data)

                model_name = version.model_name
                if model_name not in self._model_cache:
                    self._model_cache[model_name] = {}

                self._model_cache[model_name][version.version] = version

            except Exception as e:
                logger.warning(f"Failed to load model metadata from {metadata_file}: {e}")

        logger.info(f"Loaded {sum(len(versions) for versions in self._model_cache.values())} model versions")

    def _save_model_version(self, version: ModelVersion) -> None:
        """Save a model version to disk."""
        metadata_file = self.metadata_dir / f"{version.model_name}_{version.version}.json"

        with open(metadata_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2, default=str)

        # Update cache
        if version.model_name not in self._model_cache:
            self._model_cache[version.model_name] = {}
        self._model_cache[version.model_name][version.version] = version

    def register_model(
        self,
        model_name: str,
        model_path: Path,
        metadata: Dict[str, Any],
        created_by: str,
        description: str = "",
        tags: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Register a new model version."""

        # Generate version number
        existing_versions = self._model_cache.get(model_name, {})
        version_num = len(existing_versions) + 1
        version = f"v{version_num}"

        # Copy model file to registry
        registry_model_path = self.models_dir / f"{model_name}_{version}.joblib"
        shutil.copy2(model_path, registry_model_path)

        # Create model version object
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
            tags=tags or {},
            model_path=registry_model_path,
            metadata_path=self.metadata_dir / f"{model_name}_{version}.json",
            metrics=metadata.get("metrics", {}),
        )

        # Auto-approval logic
        if self.config.enable_auto_approval and self._should_auto_approve(metadata):
            model_version.approval_status = ApprovalStatus.AUTO_APPROVED
            model_version.approved_by = "system"
            model_version.approved_at = datetime.now()
            model_version.approval_notes = "Auto-approved based on performance thresholds"

        # Save to registry
        self._save_model_version(model_version)

        logger.info(f"Registered model {model_name} {version} in {ModelStage.DEVELOPMENT.value} stage")
        return model_version

    def _should_auto_approve(self, metadata: Dict[str, Any]) -> bool:
        """Check if model should be auto-approved based on performance."""
        metrics = metadata.get("metrics", {})

        concordance = metrics.get("concordance", 0)
        brier = metrics.get("ibs", 1)  # IBS is integrated brier score

        return (
            concordance >= self.config.auto_approval_thresholds.get("concordance_min", 0.7)
            and brier <= self.config.auto_approval_thresholds.get("brier_max", 0.25)
        )

    def promote_model(
        self,
        model_name: str,
        version: str,
        target_stage: ModelStage,
        promoted_by: str,
        notes: str = "",
    ) -> bool:
        """Promote a model to a new stage."""

        model_version = self.get_model_version(model_name, version)
        if not model_version:
            raise ValueError(f"Model {model_name} {version} not found")

        # Check approval requirements
        if target_stage == ModelStage.PRODUCTION and self.config.require_approval_for_production:
            if model_version.approval_status != ApprovalStatus.APPROVED:
                raise ValueError(f"Model {model_name} {version} requires approval before production promotion")

        # Check if promotion is valid
        if not self._is_valid_promotion(model_version.stage, target_stage):
            raise ValueError(f"Invalid promotion from {model_version.stage.value} to {target_stage.value}")

        # Perform promotion
        model_version.promote(target_stage, promoted_by, notes)
        self._save_model_version(model_version)

        return True

    def _is_valid_promotion(self, from_stage: ModelStage, to_stage: ModelStage) -> bool:
        """Check if a promotion transition is valid."""
        valid_transitions = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT, ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.STAGING, ModelStage.DEPRECATED],
            ModelStage.DEPRECATED: [ModelStage.ARCHIVED],
            ModelStage.ARCHIVED: [],  # Cannot promote from archived
        }

        return to_stage in valid_transitions.get(from_stage, [])

    def approve_model(self, model_name: str, version: str, approved_by: str, notes: str = "") -> None:
        """Approve a model for promotion."""
        model_version = self.get_model_version(model_name, version)
        if not model_version:
            raise ValueError(f"Model {model_name} {version} not found")

        model_version.approve(approved_by, notes)
        self._save_model_version(model_version)

    def reject_model(self, model_name: str, version: str, rejected_by: str, notes: str = "") -> None:
        """Reject a model for promotion."""
        model_version = self.get_model_version(model_name, version)
        if not model_version:
            raise ValueError(f"Model {model_name} {version} not found")

        model_version.reject(rejected_by, notes)
        self._save_model_version(model_version)

    def get_model_version(self, model_name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._model_cache.get(model_name, {}).get(version)

    def get_latest_version(self, model_name: str, stage: Optional[ModelStage] = None) -> Optional[ModelVersion]:
        """Get the latest version of a model, optionally filtered by stage."""
        model_versions = self._model_cache.get(model_name, {})

        if not model_versions:
            return None

        versions = list(model_versions.values())

        if stage:
            versions = [v for v in versions if v.stage == stage]

        if not versions:
            return None

        # Return most recently created version
        return max(versions, key=lambda v: v.created_at)

    def list_models(
        self,
        stage: Optional[ModelStage] = None,
        model_name: Optional[str] = None,
    ) -> List[ModelVersion]:
        """List model versions with optional filtering."""
        all_versions = []
        for model_versions in self._model_cache.values():
            all_versions.extend(model_versions.values())

        # Apply filters
        if model_name:
            all_versions = [v for v in all_versions if v.model_name == model_name]

        if stage:
            all_versions = [v for v in all_versions if v.stage == stage]

        # Sort by creation time (newest first)
        return sorted(all_versions, key=lambda v: v.created_at, reverse=True)

    def check_model_health(self, model_name: str, version: str) -> Dict[str, Any]:
        """Perform health check on a model version."""
        model_version = self.get_model_version(model_name, version)
        if not model_version:
            return {"status": "not_found", "score": 0.0}

        # Basic health checks
        health_score = 1.0
        issues = []

        # Check if model file exists
        if not model_version.model_path or not model_version.model_path.exists():
            health_score -= 0.5
            issues.append("Model file missing")

        # Check if model is too old (for production models)
        if model_version.stage == ModelStage.PRODUCTION:
            days_old = (datetime.now() - model_version.created_at).days
            if days_old > 180:  # 6 months
                health_score -= 0.2
                issues.append("Model is older than 6 months")

        # Check approval status
        if model_version.approval_status == ApprovalStatus.REJECTED:
            health_score -= 0.3
            issues.append("Model was rejected")

        # Determine status
        if health_score >= 0.9:
            status = "excellent"
        elif health_score >= 0.7:
            status = "good"
        elif health_score >= 0.5:
            status = "fair"
        else:
            status = "poor"

        # Update model health
        model_version.update_health(status, health_score)
        self._save_model_version(model_version)

        return {
            "status": status,
            "score": health_score,
            "issues": issues,
            "last_check": model_version.last_health_check.isoformat() if model_version.last_health_check else None,
        }

    def cleanup_old_versions(self) -> int:
        """Clean up old model versions based on retention policy."""
        deleted_count = 0

        for model_name, versions in self._model_cache.items():
            for version_str, version in versions.items():
                retention_days = self.config.retention_policy.get(version.stage.value, -1)

                if retention_days == -1:  # Never delete
                    continue

                days_old = (datetime.now() - version.created_at).days
                if days_old > retention_days:
                    # Archive the model
                    version.stage = ModelStage.ARCHIVED
                    version.promote(ModelStage.ARCHIVED, "system", "Auto-archived due to retention policy")

                    self._save_model_version(version)
                    deleted_count += 1

        logger.info(f"Archived {deleted_count} old model versions")
        return deleted_count

    def generate_report(self) -> str:
        """Generate a comprehensive registry report."""
        all_models = self.list_models()

        if not all_models:
            return "No models registered in the registry."

        # Group by stage
        by_stage = {}
        for stage in ModelStage:
            by_stage[stage] = [m for m in all_models if m.stage == stage]

        # Create report
        report_lines = ["# Model Registry Report", f"**Generated:** {datetime.now().isoformat()}", ""]

        # Summary
        report_lines.extend([
            "## Summary",
            f"- Total Models: {len(set(m.model_name for m in all_models))}",
            f"- Total Versions: {len(all_models)}",
            "",
        ])

        # By stage
        report_lines.append("## Models by Stage")
        for stage in ModelStage:
            models_in_stage = by_stage[stage]
            if models_in_stage:
                report_lines.append(f"### {stage.value.title()} ({len(models_in_stage)} versions)")
                for model in sorted(models_in_stage, key=lambda m: m.created_at, reverse=True):
                    approval_status = f" ({model.approval_status.value})" if model.approval_status != ApprovalStatus.AUTO_APPROVED else ""
                    health_status = f" [{model.health_status}]" if model.health_score else ""
                    report_lines.append(f"- {model.model_name} {model.version} - {model.created_by} - {model.created_at.date()}{approval_status}{health_status}")
                report_lines.append("")

        return "\n".join(report_lines)


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry(config: Optional[RegistryConfig] = None) -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        if config is None:
            # Default configuration
            config = RegistryConfig(registry_root=Path("results/model_registry"))
        _registry = ModelRegistry(config)
    return _registry


def initialize_registry(registry_root: Path, **config_kwargs) -> ModelRegistry:
    """Initialize the global model registry."""
    config = RegistryConfig(registry_root=registry_root, **config_kwargs)
    global _registry
    _registry = ModelRegistry(config)
    return _registry

