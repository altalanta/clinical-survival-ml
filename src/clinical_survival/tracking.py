"""
MLflow tracking with resilience patterns for graceful degradation.

This module provides MLflow integration that:
- Continues working even if MLflow server is unavailable
- Uses circuit breaker to prevent cascading failures
- Retries transient failures with exponential backoff
- Logs locally when MLflow is unavailable
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from clinical_survival.logging_config import get_logger
from clinical_survival.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    graceful_degradation,
    retry_with_backoff,
)

# Get module logger
logger = get_logger(__name__)

# Try to import mlflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None  # type: ignore
    MlflowClient = None  # type: ignore
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Tracking will be disabled.")


class NullTracker:
    """
    Null implementation of tracker interface for graceful degradation.
    
    All methods are no-ops that return None or empty values.
    """
    
    def __init__(self) -> None:
        self.is_enabled = False
    
    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        yield None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        pass
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        pass
    
    def register_model(self, model_uri: str, name: str) -> None:
        pass
    
    def set_tag(self, key: str, value: str) -> None:
        pass
    
    def end_run(self) -> None:
        pass


class LocalFallbackTracker:
    """
    Local file-based fallback tracker when MLflow is unavailable.
    
    Logs metrics and parameters to local JSON files so they can be
    recovered later when MLflow becomes available.
    """
    
    def __init__(self, fallback_dir: Path):
        self.fallback_dir = Path(fallback_dir)
        self.fallback_dir.mkdir(parents=True, exist_ok=True)
        self.is_enabled = True
        self._current_run: Optional[Dict[str, Any]] = None
        self._run_count = 0
        
        logger.info(
            f"Local fallback tracker initialized at {self.fallback_dir}",
            extra={"fallback_dir": str(self.fallback_dir)},
        )
    
    @contextmanager
    def start_run(self, run_name: str) -> Generator[None, None, None]:
        self._run_count += 1
        self._current_run = {
            "run_name": run_name,
            "params": {},
            "metrics": [],
            "artifacts": [],
            "tags": {},
        }
        logger.debug(f"Started local fallback run: {run_name}")
        try:
            yield None
        finally:
            self._save_run()
            self._current_run = None
    
    def _save_run(self) -> None:
        if self._current_run is None:
            return
        
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_file = self.fallback_dir / f"run_{timestamp}_{self._run_count}.json"
        
        with open(run_file, "w") as f:
            json.dump(self._current_run, f, indent=2, default=str)
        
        logger.info(
            f"Saved fallback run to {run_file}",
            extra={"path": str(run_file)},
        )
    
    def log_params(self, params: Dict[str, Any]) -> None:
        if self._current_run is not None:
            self._current_run["params"].update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._current_run is not None:
            self._current_run["metrics"].append({"values": metrics, "step": step})
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        if self._current_run is not None:
            self._current_run["artifacts"].append({
                "local_path": local_path,
                "artifact_path": artifact_path,
            })
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        self.log_artifact(local_dir, artifact_path)
    
    def register_model(self, model_uri: str, name: str) -> None:
        if self._current_run is not None:
            self._current_run["registered_model"] = {"uri": model_uri, "name": name}
    
    def set_tag(self, key: str, value: str) -> None:
        if self._current_run is not None:
            self._current_run["tags"][key] = value
    
    def end_run(self) -> None:
        pass


class MLflowTracker:
    """
    Resilient MLflow tracker with graceful degradation.
    
    Features:
    - Circuit breaker to prevent cascading failures
    - Automatic retry with exponential backoff
    - Local fallback when MLflow is unavailable
    - Graceful degradation for non-critical operations
    
    Usage:
        tracker = MLflowTracker(config)
        
        with tracker.start_run("training"):
            tracker.log_params({"learning_rate": 0.01})
            # ... training code ...
            tracker.log_metrics({"accuracy": 0.95})
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        fallback_dir: Optional[Path] = None,
    ):
        """
        Initialize the MLflow tracker.
        
        Args:
            config: Configuration dictionary with keys:
                - enabled: Whether tracking is enabled
                - tracking_uri: MLflow tracking URI
                - experiment_name: Experiment name
            fallback_dir: Directory for local fallback storage
        """
        self.config = config or {}
        self.is_enabled = self.config.get("enabled", False) and MLFLOW_AVAILABLE
        self._fallback_dir = fallback_dir or Path("artifacts/mlflow_fallback")
        self._fallback_tracker: Optional[LocalFallbackTracker] = None
        self._circuit_breaker: Optional[CircuitBreaker] = None
        self._active_run = False
        
        if self.is_enabled:
            self._initialize_mlflow()
    
    def _initialize_mlflow(self) -> None:
        """Initialize MLflow with circuit breaker protection."""
        self._circuit_breaker = CircuitBreaker(
            name="mlflow",
            failure_threshold=3,
            success_threshold=1,
            recovery_timeout=60.0,
        )
        
        try:
            with self._circuit_breaker:
                tracking_uri = self.config.get("tracking_uri", "file:./mlruns")
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(
                    self.config.get("experiment_name", "clinical-survival-ml")
                )
                logger.info(
                    "MLflow initialized successfully",
                    extra={"tracking_uri": tracking_uri},
                )
        except Exception as e:
            logger.warning(
                f"Failed to initialize MLflow: {e}. Using local fallback.",
                extra={"error": str(e)},
            )
            self._use_fallback()
    
    def _use_fallback(self) -> None:
        """Switch to local fallback tracker."""
        if self._fallback_tracker is None:
            self._fallback_tracker = LocalFallbackTracker(self._fallback_dir)
        logger.info("Switched to local fallback tracker")
    
    def _get_tracker(self) -> Any:
        """Get the active tracker (MLflow or fallback)."""
        if self._fallback_tracker is not None:
            return self._fallback_tracker
        return mlflow
    
    @contextmanager
    def start_run(self, run_name: str) -> Generator[Optional[Any], None, None]:
        """
        Start an MLflow run with graceful degradation.
        
        Args:
            run_name: Name for the run
            
        Yields:
            MLflow ActiveRun or None
        """
        if not self.is_enabled:
            yield None
            return
        
        # Try MLflow first
        if self._circuit_breaker is not None and self._fallback_tracker is None:
            try:
                with self._circuit_breaker:
                    run = mlflow.start_run(run_name=run_name)
                    self._active_run = True
                    try:
                        yield run
                    finally:
                        self._active_run = False
                        mlflow.end_run()
                    return
            except (CircuitOpenError, Exception) as e:
                logger.warning(
                    f"MLflow unavailable, using fallback: {e}",
                    extra={"error_type": type(e).__name__},
                )
                self._use_fallback()
        
        # Use fallback tracker
        if self._fallback_tracker is not None:
            with self._fallback_tracker.start_run(run_name):
                yield None
            return
        
        yield None
    
    @graceful_degradation(default=None)
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters with graceful degradation."""
        if not self.is_enabled:
            return
        
        if self._fallback_tracker is not None:
            self._fallback_tracker.log_params(params)
            return
        
        if self._circuit_breaker is not None:
            try:
                with self._circuit_breaker:
                    mlflow.log_params(params)
            except CircuitOpenError:
                self._use_fallback()
                self._fallback_tracker.log_params(params)  # type: ignore
    
    @graceful_degradation(default=None)
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics with graceful degradation."""
        if not self.is_enabled:
            return
        
        if self._fallback_tracker is not None:
            self._fallback_tracker.log_metrics(metrics, step)
            return
        
        if self._circuit_breaker is not None:
            try:
                with self._circuit_breaker:
                    mlflow.log_metrics(metrics, step=step)
            except CircuitOpenError:
                self._use_fallback()
                self._fallback_tracker.log_metrics(metrics, step)  # type: ignore
    
    @graceful_degradation(default=None)
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log an artifact with graceful degradation."""
        if not self.is_enabled:
            return
        
        if not Path(local_path).exists():
            logger.warning(f"Artifact path does not exist: {local_path}")
            return
        
        if self._fallback_tracker is not None:
            self._fallback_tracker.log_artifact(local_path, artifact_path)
            return
        
        if self._circuit_breaker is not None:
            try:
                with self._circuit_breaker:
                    mlflow.log_artifact(local_path, artifact_path)
            except CircuitOpenError:
                self._use_fallback()
                self._fallback_tracker.log_artifact(local_path, artifact_path)  # type: ignore
    
    @graceful_degradation(default=None)
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None) -> None:
        """Log artifacts directory with graceful degradation."""
        if not self.is_enabled:
            return
        
        if not Path(local_dir).is_dir():
            logger.warning(f"Artifacts directory does not exist: {local_dir}")
            return
        
        if self._fallback_tracker is not None:
            self._fallback_tracker.log_artifacts(local_dir, artifact_path)
            return
        
        if self._circuit_breaker is not None:
            try:
                with self._circuit_breaker:
                    mlflow.log_artifacts(local_dir, artifact_path)
            except CircuitOpenError:
                self._use_fallback()
                self._fallback_tracker.log_artifacts(local_dir, artifact_path)  # type: ignore
    
    @graceful_degradation(default=None)
    def register_model(self, model_uri: str, name: str) -> Optional[Any]:
        """Register a model with graceful degradation."""
        if not self.is_enabled:
            return None
        
        if self._fallback_tracker is not None:
            self._fallback_tracker.register_model(model_uri, name)
            return None
        
        if self._circuit_breaker is not None:
            try:
                with self._circuit_breaker:
                    return mlflow.register_model(model_uri, name)
            except CircuitOpenError:
                self._use_fallback()
                self._fallback_tracker.register_model(model_uri, name)  # type: ignore
        
        return None
    
    @graceful_degradation(default=None)
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag with graceful degradation."""
        if not self.is_enabled:
            return
        
        if self._fallback_tracker is not None:
            self._fallback_tracker.set_tag(key, value)
            return
        
        if self._circuit_breaker is not None:
            try:
                with self._circuit_breaker:
                    mlflow.set_tag(key, value)
            except CircuitOpenError:
                self._use_fallback()
                self._fallback_tracker.set_tag(key, value)  # type: ignore
    
    def end_run(self) -> None:
        """End the current run."""
        if mlflow is not None and mlflow.active_run():
            mlflow.end_run()
        self._active_run = False
    
    @property
    def is_degraded(self) -> bool:
        """Check if tracker is in degraded mode (using fallback)."""
        return self._fallback_tracker is not None
    
    def get_circuit_state(self) -> Optional[str]:
        """Get the current circuit breaker state."""
        if self._circuit_breaker is not None:
            return self._circuit_breaker.state.value
        return None
