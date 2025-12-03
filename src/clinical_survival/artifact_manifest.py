"""
Artifact manifest and versioning system.

This module provides comprehensive tracking of pipeline artifacts:
- Manifest generation for reproducibility
- Artifact versioning and checksums
- Configuration snapshots
- Lineage tracking
- Export/import capabilities

Usage:
    from clinical_survival.artifact_manifest import ArtifactManifest, ManifestManager
    
    manager = ManifestManager(output_dir)
    
    # During pipeline execution
    manager.start_run(params_config, features_config, grid_config)
    manager.add_artifact("model", model_path, category="models")
    manager.add_metric("concordance", 0.75, model="coxph")
    
    # At the end
    manifest = manager.finalize()
    manager.save()
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from clinical_survival.logging_config import get_logger

# Module logger
logger = get_logger(__name__)


@dataclass
class ArtifactInfo:
    """Information about a single artifact."""
    name: str
    path: str
    category: str
    checksum: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "category": self.category,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class ArtifactManifest:
    """
    Complete manifest for a pipeline run.
    
    Contains all information needed to reproduce or understand a run:
    - Run metadata (timestamp, duration, etc.)
    - Configuration snapshots
    - Environment information
    - List of artifacts with checksums
    - Metrics and results
    """
    
    # Run identification
    run_id: str
    run_name: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    status: str = "running"
    
    # Correlation
    correlation_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    
    # Configuration
    params_config: Dict[str, Any] = field(default_factory=dict)
    features_config: Dict[str, Any] = field(default_factory=dict)
    grid_config: Dict[str, Any] = field(default_factory=dict)
    
    # Environment
    environment: Dict[str, Any] = field(default_factory=dict)
    
    # Artifacts
    artifacts: List[ArtifactInfo] = field(default_factory=list)
    
    # Metrics
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Pipeline info
    pipeline_steps: List[str] = field(default_factory=list)
    step_durations: Dict[str, float] = field(default_factory=dict)
    
    # Best model selection
    best_model: Optional[str] = None
    best_model_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Notes and tags
    notes: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_artifact(self, artifact: ArtifactInfo) -> None:
        """Add an artifact to the manifest."""
        self.artifacts.append(artifact)
    
    def add_metric(self, model: str, metric_name: str, value: float) -> None:
        """Add a metric value."""
        if model not in self.metrics:
            self.metrics[model] = {}
        self.metrics[model][metric_name] = value
    
    def add_note(self, note: str) -> None:
        """Add a note to the manifest."""
        self.notes.append(f"[{datetime.utcnow().isoformat()}] {note}")
    
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag."""
        self.tags[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "manifest_version": "1.0",
            "run_id": self.run_id,
            "run_name": self.run_name,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "correlation_id": self.correlation_id,
            "parent_run_id": self.parent_run_id,
            "configuration": {
                "params": self.params_config,
                "features": self.features_config,
                "grid": self.grid_config,
            },
            "environment": self.environment,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "metrics": self.metrics,
            "pipeline": {
                "steps": self.pipeline_steps,
                "step_durations": self.step_durations,
            },
            "best_model": {
                "name": self.best_model,
                "metrics": self.best_model_metrics,
            } if self.best_model else None,
            "notes": self.notes,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactManifest":
        """Create manifest from dictionary."""
        manifest = cls(
            run_id=data["run_id"],
            run_name=data.get("run_name"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            completed_at=data.get("completed_at"),
            duration_seconds=data.get("duration_seconds"),
            status=data.get("status", "unknown"),
            correlation_id=data.get("correlation_id"),
            parent_run_id=data.get("parent_run_id"),
        )
        
        # Configuration
        config = data.get("configuration", {})
        manifest.params_config = config.get("params", {})
        manifest.features_config = config.get("features", {})
        manifest.grid_config = config.get("grid", {})
        
        # Environment
        manifest.environment = data.get("environment", {})
        
        # Artifacts
        for artifact_data in data.get("artifacts", []):
            artifact = ArtifactInfo(
                name=artifact_data["name"],
                path=artifact_data["path"],
                category=artifact_data["category"],
                checksum=artifact_data.get("checksum"),
                size_bytes=artifact_data.get("size_bytes"),
                created_at=artifact_data.get("created_at"),
                metadata=artifact_data.get("metadata", {}),
            )
            manifest.artifacts.append(artifact)
        
        # Metrics
        manifest.metrics = data.get("metrics", {})
        
        # Pipeline
        pipeline = data.get("pipeline", {})
        manifest.pipeline_steps = pipeline.get("steps", [])
        manifest.step_durations = pipeline.get("step_durations", {})
        
        # Best model
        best = data.get("best_model")
        if best:
            manifest.best_model = best.get("name")
            manifest.best_model_metrics = best.get("metrics", {})
        
        # Notes and tags
        manifest.notes = data.get("notes", [])
        manifest.tags = data.get("tags", {})
        
        return manifest


class ManifestManager:
    """
    Manager for creating and maintaining artifact manifests.
    
    Usage:
        manager = ManifestManager(Path("results"))
        manager.start_run(params_config, features_config, grid_config)
        
        # Add artifacts during pipeline execution
        manager.add_artifact("coxph_model", model_path, category="models")
        manager.add_metric("concordance", 0.75, model="coxph")
        
        # Finalize and save
        manifest = manager.finalize(success=True)
        manager.save()
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        run_name: Optional[str] = None,
    ):
        """
        Initialize the manifest manager.
        
        Args:
            output_dir: Directory for output artifacts
            run_name: Optional name for this run
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._manifest: Optional[ArtifactManifest] = None
        self._start_time: Optional[datetime] = None
        self._run_name = run_name
    
    def start_run(
        self,
        params_config: Any = None,
        features_config: Any = None,
        grid_config: Dict[str, Any] = None,
        correlation_id: Optional[str] = None,
    ) -> ArtifactManifest:
        """
        Start a new run and initialize the manifest.
        
        Args:
            params_config: ParamsConfig object
            features_config: FeaturesConfig object  
            grid_config: Model grid configuration
            correlation_id: Optional correlation ID
            
        Returns:
            Initialized ArtifactManifest
        """
        self._start_time = datetime.utcnow()
        run_id = self._generate_run_id()
        
        self._manifest = ArtifactManifest(
            run_id=run_id,
            run_name=self._run_name or f"run_{run_id[:8]}",
            correlation_id=correlation_id,
        )
        
        # Store configuration snapshots
        if params_config:
            self._manifest.params_config = self._serialize_config(params_config)
            self._manifest.pipeline_steps = list(params_config.pipeline) if hasattr(params_config, 'pipeline') else []
        
        if features_config:
            self._manifest.features_config = self._serialize_config(features_config)
        
        if grid_config:
            self._manifest.grid_config = grid_config
        
        # Capture environment information
        self._manifest.environment = self._capture_environment()
        
        logger.info(
            f"Started manifest for run {run_id}",
            extra={"run_id": run_id, "run_name": self._manifest.run_name},
        )
        
        return self._manifest
    
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        import uuid
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        unique = uuid.uuid4().hex[:8]
        return f"{timestamp}_{unique}"
    
    def _serialize_config(self, config: Any) -> Dict[str, Any]:
        """Serialize a Pydantic config to dictionary."""
        if hasattr(config, "model_dump"):
            return config.model_dump()
        elif hasattr(config, "dict"):
            return config.dict()
        elif isinstance(config, dict):
            return config
        else:
            return {"raw": str(config)}
    
    def _capture_environment(self) -> Dict[str, Any]:
        """Capture environment information."""
        env = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cwd": os.getcwd(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        }
        
        # Try to get git info
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            env["git_commit"] = git_hash
            
            git_branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            env["git_branch"] = git_branch
            
            # Check for uncommitted changes
            git_status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            env["git_dirty"] = bool(git_status)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        # Package versions
        try:
            import numpy
            import pandas
            import sklearn
            
            env["package_versions"] = {
                "numpy": numpy.__version__,
                "pandas": pandas.__version__,
                "sklearn": sklearn.__version__,
            }
            
            try:
                import xgboost
                env["package_versions"]["xgboost"] = xgboost.__version__
            except ImportError:
                pass
            
            try:
                import sksurv
                env["package_versions"]["scikit-survival"] = sksurv.__version__
            except ImportError:
                pass
                
        except Exception:
            pass
        
        return env
    
    def add_artifact(
        self,
        name: str,
        path: Union[str, Path],
        category: str = "general",
        compute_checksum: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactInfo:
        """
        Add an artifact to the manifest.
        
        Args:
            name: Artifact name
            path: Path to the artifact file
            category: Category (models, metrics, plots, data, etc.)
            compute_checksum: Whether to compute file checksum
            metadata: Additional metadata
            
        Returns:
            ArtifactInfo for the added artifact
        """
        if self._manifest is None:
            raise RuntimeError("Must call start_run() before adding artifacts")
        
        path = Path(path)
        
        artifact = ArtifactInfo(
            name=name,
            path=str(path),
            category=category,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )
        
        if path.exists():
            artifact.size_bytes = path.stat().st_size
            
            if compute_checksum and path.is_file():
                artifact.checksum = self._compute_checksum(path)
        
        self._manifest.add_artifact(artifact)
        
        logger.debug(
            f"Added artifact: {name}",
            extra={"artifact": name, "category": category, "path": str(path)},
        )
        
        return artifact
    
    def _compute_checksum(self, path: Path, algorithm: str = "sha256") -> str:
        """Compute file checksum."""
        hasher = hashlib.new(algorithm)
        
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        
        return f"{algorithm}:{hasher.hexdigest()}"
    
    def add_metric(
        self,
        metric_name: str,
        value: float,
        model: str = "global",
    ) -> None:
        """Add a metric value."""
        if self._manifest is None:
            raise RuntimeError("Must call start_run() before adding metrics")
        
        self._manifest.add_metric(model, metric_name, value)
    
    def add_step_duration(self, step_name: str, duration_seconds: float) -> None:
        """Record duration for a pipeline step."""
        if self._manifest is None:
            raise RuntimeError("Must call start_run() before adding step durations")
        
        self._manifest.step_durations[step_name] = duration_seconds
    
    def set_best_model(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Set the best model selection."""
        if self._manifest is None:
            raise RuntimeError("Must call start_run() first")
        
        self._manifest.best_model = model_name
        self._manifest.best_model_metrics = metrics
    
    def add_note(self, note: str) -> None:
        """Add a note to the manifest."""
        if self._manifest is None:
            raise RuntimeError("Must call start_run() first")
        
        self._manifest.add_note(note)
    
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the manifest."""
        if self._manifest is None:
            raise RuntimeError("Must call start_run() first")
        
        self._manifest.set_tag(key, value)
    
    def finalize(self, success: bool = True) -> ArtifactManifest:
        """
        Finalize the manifest.
        
        Args:
            success: Whether the run completed successfully
            
        Returns:
            Finalized ArtifactManifest
        """
        if self._manifest is None:
            raise RuntimeError("Must call start_run() first")
        
        self._manifest.completed_at = datetime.utcnow().isoformat()
        self._manifest.status = "completed" if success else "failed"
        
        if self._start_time:
            delta = datetime.utcnow() - self._start_time
            self._manifest.duration_seconds = delta.total_seconds()
        
        logger.info(
            f"Finalized manifest for run {self._manifest.run_id}",
            extra={
                "run_id": self._manifest.run_id,
                "status": self._manifest.status,
                "duration_seconds": self._manifest.duration_seconds,
                "n_artifacts": len(self._manifest.artifacts),
            },
        )
        
        return self._manifest
    
    def save(self, path: Optional[Path] = None) -> Path:
        """
        Save the manifest to a JSON file.
        
        Args:
            path: Optional custom path
            
        Returns:
            Path to saved manifest file
        """
        if self._manifest is None:
            raise RuntimeError("No manifest to save")
        
        if path is None:
            manifest_dir = self.output_dir / "artifacts" / "manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            path = manifest_dir / f"manifest_{self._manifest.run_id}.json"
        
        path = Path(path)
        
        with open(path, "w") as f:
            json.dump(self._manifest.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved manifest to {path}")
        
        return path
    
    @staticmethod
    def load(path: Union[str, Path]) -> ArtifactManifest:
        """Load a manifest from file."""
        with open(path) as f:
            data = json.load(f)
        
        return ArtifactManifest.from_dict(data)
    
    @property
    def manifest(self) -> Optional[ArtifactManifest]:
        """Get current manifest."""
        return self._manifest


def create_manifest_from_run(
    output_dir: Path,
    params_config: Any,
    features_config: Any,
    grid_config: Dict[str, Any],
    metrics: Dict[str, Dict[str, float]],
    model_paths: Dict[str, Path],
    best_model: Optional[str] = None,
) -> ArtifactManifest:
    """
    Convenience function to create a manifest from run results.
    
    Args:
        output_dir: Output directory
        params_config: Parameters configuration
        features_config: Features configuration
        grid_config: Grid configuration
        metrics: Model metrics dictionary
        model_paths: Dictionary of model names to saved model paths
        best_model: Name of best model
        
    Returns:
        Complete ArtifactManifest
    """
    manager = ManifestManager(output_dir)
    manager.start_run(params_config, features_config, grid_config)
    
    # Add model artifacts
    for model_name, model_path in model_paths.items():
        manager.add_artifact(
            name=f"{model_name}_model",
            path=model_path,
            category="models",
            metadata={"model_name": model_name},
        )
    
    # Add metrics
    for model_name, model_metrics in metrics.items():
        for metric_name, value in model_metrics.items():
            manager.add_metric(metric_name, value, model=model_name)
    
    # Set best model
    if best_model and best_model in metrics:
        manager.set_best_model(best_model, metrics[best_model])
    
    manifest = manager.finalize(success=True)
    manager.save()
    
    return manifest

