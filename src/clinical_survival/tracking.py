"""
Enhanced MLflow tracking with experiment comparison, visualization, and lineage.

This module provides comprehensive experiment tracking that:
- Continues working even if MLflow server is unavailable
- Uses circuit breaker to prevent cascading failures
- Retries transient failures with exponential backoff
- Logs locally when MLflow is unavailable
- Provides advanced experiment comparison and visualization
- Tracks experiment lineage and performance trends
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

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    pd = None
    px = None
    go = None
    make_subplots = None
    PLOTLY_AVAILABLE = False

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


class ExperimentComparator:
    """
    Advanced experiment comparison and analysis.

    Provides statistical comparison, performance trends, and visualization
    of multiple experiments.
    """

    def __init__(self, tracker: MLflowTracker):
        self.tracker = tracker
        self.logger = get_logger(f"{__name__}.ExperimentComparator")

    def compare_experiments(
        self,
        experiment_ids: list[str],
        metrics: Optional[list[str]] = None,
        parameters: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments with statistical analysis.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Metrics to include in comparison (default: all numeric metrics)
            parameters: Parameters to include in comparison

        Returns:
            Dictionary containing comparison results and statistics
        """
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLflow not available for experiment comparison")
            return {}

        if not self.tracker.is_enabled:
            self.logger.warning("MLflow tracker not enabled")
            return {}

        try:
            client = MlflowClient()

            all_runs = []
            for exp_id in experiment_ids:
                runs = client.search_runs(
                    experiment_ids=[exp_id],
                    filter_string="",
                    order_by=["start_time DESC"]
                )
                all_runs.extend(runs)

            if not all_runs:
                return {"error": "No runs found for specified experiments"}

            # Extract metrics and parameters
            comparison_data = []
            for run in all_runs:
                run_data = {
                    "run_id": run.info.run_id,
                    "experiment_id": run.info.experiment_id,
                    "start_time": run.info.start_time,
                    "status": run.info.status,
                }

                # Add metrics
                if run.data.metrics:
                    run_data.update(run.data.metrics)

                # Add parameters
                if run.data.params:
                    for param_name, param_value in run.data.params.items():
                        # Try to convert to numeric if possible
                        try:
                            run_data[f"param_{param_name}"] = float(param_value)
                        except (ValueError, TypeError):
                            run_data[f"param_{param_name}"] = param_value

                # Add tags
                if run.data.tags:
                    for tag_name, tag_value in run.data.tags.items():
                        run_data[f"tag_{tag_name}"] = tag_value

                comparison_data.append(run_data)

            df = pd.DataFrame(comparison_data)

            # Statistical analysis
            stats = self._compute_comparison_statistics(df, metrics or [])

            return {
                "runs_count": len(comparison_data),
                "experiments_count": len(experiment_ids),
                "statistics": stats,
                "raw_data": df.to_dict('records'),
                "summary": self._generate_comparison_summary(df, metrics or []),
            }

        except Exception as e:
            self.logger.error(f"Failed to compare experiments: {e}")
            return {"error": str(e)}

    def _compute_comparison_statistics(self, df: pd.DataFrame, metrics: list[str]) -> Dict[str, Any]:
        """Compute statistical comparisons between experiments."""
        stats = {}

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # If specific metrics requested, filter to those
        if metrics:
            metric_cols = [col for col in numeric_cols if any(m in col for m in metrics)]
        else:
            # Auto-detect metrics (exclude param_ and tag_ columns)
            metric_cols = [col for col in numeric_cols if not col.startswith(('param_', 'tag_'))]

        for metric in metric_cols:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 1:
                    stats[metric] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "count": len(values),
                        "quartiles": {
                            "25%": float(values.quantile(0.25)),
                            "50%": float(values.quantile(0.50)),
                            "75%": float(values.quantile(0.75)),
                        }
                    }

        return stats

    def _generate_comparison_summary(self, df: pd.DataFrame, metrics: list[str]) -> Dict[str, Any]:
        """Generate a human-readable comparison summary."""
        summary = {
            "total_runs": len(df),
            "experiments": df.get('experiment_id', pd.Series()).nunique() if 'experiment_id' in df.columns else 0,
            "best_performers": {},
            "trends": {},
        }

        # Find best performers for each metric
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if not col.startswith(('param_', 'tag_'))]

        for metric in metric_cols:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    best_run_idx = values.idxmax() if 'concordance' in metric.lower() else values.idxmin()
                    best_run = df.loc[best_run_idx]
                    summary["best_performers"][metric] = {
                        "value": float(best_run[metric]),
                        "run_id": best_run.get('run_id', 'unknown'),
                        "experiment_id": best_run.get('experiment_id', 'unknown'),
                    }

        return summary

    def plot_comparison(self, comparison_result: Dict[str, Any], output_path: Optional[Path] = None):
        """
        Generate visualization plots for experiment comparison.

        Args:
            comparison_result: Result from compare_experiments()
            output_path: Path to save plots (shows interactively if None)
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available for visualization")
            return

        if "error" in comparison_result:
            self.logger.error(f"Cannot plot comparison: {comparison_result['error']}")
            return

        df = pd.DataFrame(comparison_result["raw_data"])

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Metric Distributions", "Parameter Correlations", "Performance Trends", "Experiment Comparison"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )

        # 1. Metric distributions (histogram)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if not col.startswith(('param_', 'tag_'))]

        if metric_cols:
            for i, metric in enumerate(metric_cols[:3]):  # Show first 3 metrics
                fig.add_trace(
                    go.Histogram(x=df[metric], name=metric, opacity=0.7),
                    row=1, col=1
                )

        # 2. Parameter correlations (scatter)
        param_cols = [col for col in df.columns if col.startswith('param_')]
        if len(param_cols) >= 2 and metric_cols:
            primary_metric = metric_cols[0]
            fig.add_trace(
                go.Scatter(
                    x=df[param_cols[0]],
                    y=df[primary_metric],
                    mode='markers',
                    name=f"{param_cols[0]} vs {primary_metric}",
                    text=df.get('run_id', df.index)
                ),
                row=1, col=2
            )

        # 3. Performance trends over time
        if 'start_time' in df.columns and metric_cols:
            df_sorted = df.sort_values('start_time')
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(df_sorted['start_time'], unit='ms'),
                    y=df_sorted[metric_cols[0]],
                    mode='lines+markers',
                    name=f"{metric_cols[0]} trend"
                ),
                row=2, col=1
            )

        # 4. Experiment comparison (bar chart)
        if 'experiment_id' in df.columns and metric_cols:
            exp_means = df.groupby('experiment_id')[metric_cols[0]].mean()
            fig.add_trace(
                go.Bar(x=exp_means.index, y=exp_means.values, name=f"Mean {metric_cols[0]}"),
                row=2, col=2
            )

        fig.update_layout(
            title="Experiment Comparison Dashboard",
            showlegend=True,
            height=800,
        )

        if output_path:
            fig.write_html(str(output_path))
            self.logger.info(f"Comparison plot saved to {output_path}")
        else:
            fig.show()


class ExperimentVisualizer:
    """
    Advanced visualization for experiment tracking and analysis.

    Provides interactive plots, performance dashboards, and trend analysis.
    """

    def __init__(self, tracker: MLflowTracker):
        self.tracker = tracker
        self.logger = get_logger(f"{__name__}.ExperimentVisualizer")

    def create_performance_dashboard(
        self,
        experiment_id: str,
        metrics: Optional[list[str]] = None,
        output_path: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Create an interactive performance dashboard for an experiment.

        Args:
            experiment_id: MLflow experiment ID
            metrics: Metrics to include in dashboard
            output_path: Path to save dashboard HTML

        Returns:
            Path to saved dashboard or None if failed
        """
        if not PLOTLY_AVAILABLE or not MLFLOW_AVAILABLE:
            self.logger.warning("Required dependencies not available for dashboard creation")
            return None

        try:
            client = MlflowClient()

            # Get all runs for the experiment
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="",
                order_by=["start_time DESC"]
            )

            if not runs:
                self.logger.warning(f"No runs found for experiment {experiment_id}")
                return None

            # Extract data
            runs_data = []
            for run in runs:
                run_data = {
                    "run_id": run.info.run_id,
                    "start_time": pd.to_datetime(run.info.start_time, unit='ms'),
                    "status": run.info.status,
                }

                # Add metrics
                if run.data.metrics:
                    run_data.update(run.data.metrics)

                # Add parameters
                if run.data.params:
                    for param_name, param_value in run.data.params.items():
                        try:
                            run_data[f"param_{param_name}"] = float(param_value)
                        except (ValueError, TypeError):
                            run_data[f"param_{param_name}"] = param_value

                runs_data.append(run_data)

            df = pd.DataFrame(runs_data)

            # Create dashboard
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    "Metric Trends Over Time",
                    "Parameter Importance",
                    "Performance Distribution",
                    "Run Status Summary",
                    "Parameter Correlations",
                    "Best Runs Table"
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "bar"}],
                    [{"type": "histogram"}, {"type": "pie"}],
                    [{"type": "scatter"}, {"type": "table"}]
                ]
            )

            # 1. Metric trends
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            metric_cols = [col for col in numeric_cols if not col.startswith('param_')]

            for metric in metric_cols[:3]:  # Show first 3 metrics
                fig.add_trace(
                    go.Scatter(
                        x=df['start_time'],
                        y=df[metric],
                        mode='lines+markers',
                        name=metric,
                        text=df['run_id']
                    ),
                    row=1, col=1
                )

            # 2. Parameter importance (simplified - could use feature importance)
            param_cols = [col for col in df.columns if col.startswith('param_')]
            if param_cols and metric_cols:
                # Calculate correlation between parameters and primary metric
                primary_metric = metric_cols[0]
                correlations = {}
                for param in param_cols[:5]:  # Top 5 parameters
                    corr = df[param].corr(df[primary_metric])
                    if not pd.isna(corr):
                        correlations[param] = abs(corr)

                if correlations:
                    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
                    params, corrs = zip(*sorted_params)
                    fig.add_trace(
                        go.Bar(x=params, y=corrs, name="Parameter Correlation"),
                        row=1, col=2
                    )

            # 3. Performance distribution
            if metric_cols:
                fig.add_trace(
                    go.Histogram(x=df[metric_cols[0]], name=f"{metric_cols[0]} Distribution"),
                    row=2, col=1
                )

            # 4. Run status summary
            status_counts = df['status'].value_counts()
            fig.add_trace(
                go.Pie(labels=status_counts.index, values=status_counts.values, name="Run Status"),
                row=2, col=2
            )

            # 5. Parameter correlations
            if len(param_cols) >= 2 and metric_cols:
                fig.add_trace(
                    go.Scatter(
                        x=df[param_cols[0]],
                        y=df[param_cols[1]],
                        mode='markers',
                        name="Parameter Correlation",
                        text=df[metric_cols[0]].round(3)
                    ),
                    row=3, col=1
                )

            # 6. Best runs table
            if metric_cols:
                # Sort by primary metric and show top 5
                top_runs = df.nlargest(5, metric_cols[0])[['run_id', metric_cols[0]] + param_cols[:2]]
                table_data = []
                for col in top_runs.columns:
                    table_data.append(top_runs[col].tolist())

                fig.add_trace(
                    go.Table(
                        header=dict(values=top_runs.columns.tolist()),
                        cells=dict(values=table_data)
                    ),
                    row=3, col=2
                )

            fig.update_layout(
                title=f"Experiment {experiment_id} Performance Dashboard",
                showlegend=True,
                height=1200,
            )

            if output_path:
                fig.write_html(str(output_path))
                self.logger.info(f"Performance dashboard saved to {output_path}")
                return str(output_path)
            else:
                fig.show()
                return None

        except Exception as e:
            self.logger.error(f"Failed to create performance dashboard: {e}")
            return None

    def plot_metric_trends(
        self,
        experiment_ids: list[str],
        metrics: list[str],
        output_path: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Plot metric trends across multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Metrics to plot
            output_path: Path to save plot

        Returns:
            Path to saved plot or None
        """
        if not PLOTLY_AVAILABLE or not MLFLOW_AVAILABLE:
            return None

        try:
            client = MlflowClient()

            all_data = []
            for exp_id in experiment_ids:
                runs = client.search_runs(experiment_ids=[exp_id])
                for run in runs:
                    run_data = {
                        "experiment_id": exp_id,
                        "run_id": run.info.run_id,
                        "start_time": pd.to_datetime(run.info.start_time, unit='ms'),
                    }
                    if run.data.metrics:
                        run_data.update(run.data.metrics)
                    all_data.append(run_data)

            if not all_data:
                return None

            df = pd.DataFrame(all_data)

            fig = go.Figure()

            for metric in metrics:
                if metric in df.columns:
                    for exp_id in experiment_ids:
                        exp_data = df[df['experiment_id'] == exp_id].sort_values('start_time')
                        fig.add_trace(
                            go.Scatter(
                                x=exp_data['start_time'],
                                y=exp_data[metric],
                                mode='lines+markers',
                                name=f"{exp_id}: {metric}"
                            )
                        )

            fig.update_layout(
                title="Metric Trends Across Experiments",
                xaxis_title="Time",
                yaxis_title="Metric Value",
                showlegend=True,
            )

            if output_path:
                fig.write_html(str(output_path))
                return str(output_path)
            else:
                fig.show()
                return None

        except Exception as e:
            self.logger.error(f"Failed to plot metric trends: {e}")
            return None


# Convenience functions
def compare_experiments(
    tracker: MLflowTracker,
    experiment_ids: list[str],
    **kwargs
) -> Dict[str, Any]:
    """Convenience function for experiment comparison."""
    comparator = ExperimentComparator(tracker)
    return comparator.compare_experiments(experiment_ids, **kwargs)


def create_performance_dashboard(
    tracker: MLflowTracker,
    experiment_id: str,
    **kwargs
) -> Optional[str]:
    """Convenience function for creating performance dashboards."""
    visualizer = ExperimentVisualizer(tracker)
    return visualizer.create_performance_dashboard(experiment_id, **kwargs)
