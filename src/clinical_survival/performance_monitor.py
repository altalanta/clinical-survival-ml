"""
Systematic performance profiling and optimization recommendations for the clinical survival pipeline.

This module provides:
- Performance profiling during pipeline execution
- Bottleneck identification and analysis
- Optimization recommendations based on profiling data
- Memory usage tracking and recommendations
- GPU utilization monitoring (when applicable)
- Performance regression detection
"""

from __future__ import annotations

import json
import time
import psutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from clinical_survival.logging_config import get_logger

logger = get_logger(__name__)
console = Console()


@dataclass
class PerformanceMetrics:
    """Container for performance metrics collected during execution."""

    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # CPU and Memory metrics
    cpu_percent_avg: float = 0.0
    cpu_percent_max: float = 0.0
    memory_mb_avg: float = 0.0
    memory_mb_max: float = 0.0
    memory_percent_avg: float = 0.0
    memory_percent_max: float = 0.0

    # Disk I/O
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0

    # GPU metrics (if available)
    gpu_memory_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None

    # Step-level metrics
    step_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Data processing metrics
    data_loading_time: float = 0.0
    preprocessing_time: float = 0.0
    training_time: float = 0.0
    evaluation_time: float = 0.0

    # Data sizes
    input_data_size_mb: float = 0.0
    processed_data_size_mb: float = 0.0
    model_size_mb: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Represents a performance optimization recommendation."""

    category: str  # "memory", "cpu", "io", "data", "model"
    priority: str  # "high", "medium", "low"
    title: str
    description: str
    impact_estimate: str  # "high", "medium", "low"
    implementation_effort: str  # "low", "medium", "high"
    suggested_action: str
    expected_benefit: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "priority": self.priority,
            "title": self.title,
            "description": self.description,
            "impact_estimate": self.impact_estimate,
            "implementation_effort": self.implementation_effort,
            "suggested_action": self.suggested_action,
            "expected_benefit": self.expected_benefit,
        }


class PerformanceProfiler:
    """Advanced performance profiler with optimization recommendations."""

    def __init__(self, enable_gpu_monitoring: bool = False):
        self.metrics = PerformanceMetrics(start_time=datetime.now())
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.system_metrics_history: List[Dict[str, Any]] = []

        # Check for GPU monitoring capability
        self.gpu_available = self._check_gpu_availability()

        logger.info("Performance profiler initialized", extra={
            "gpu_monitoring": enable_gpu_monitoring,
            "gpu_available": self.gpu_available,
        })

    def _check_gpu_availability(self) -> bool:
        """Check if GPU monitoring is available."""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except ImportError:
            return False

    def start_monitoring(self) -> None:
        """Start background system monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_system_metrics, daemon=True)
        self.monitoring_thread.start()

        logger.debug("System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background system monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)

        self.metrics.end_time = datetime.now()
        self.metrics.duration_seconds = (self.metrics.end_time - self.metrics.start_time).total_seconds()

        # Calculate averages from collected metrics
        self._calculate_averages()

        logger.debug("System monitoring stopped", extra={
            "duration_seconds": self.metrics.duration_seconds,
            "metrics_collected": len(self.system_metrics_history),
        })

    def _monitor_system_metrics(self) -> None:
        """Background thread to monitor system metrics."""
        process = psutil.Process()

        while self.monitoring_active:
            try:
                # CPU and Memory
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()

                # Disk I/O
                io_counters = process.io_counters()
                disk_read_mb = io_counters.read_bytes / (1024 * 1024) if io_counters else 0
                disk_write_mb = io_counters.write_bytes / (1024 * 1024) if io_counters else 0

                # GPU metrics
                gpu_memory_mb = None
                gpu_utilization = None
                if self.gpu_available and self.enable_gpu_monitoring:
                    try:
                        import GPUtil
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Monitor first GPU
                            gpu_memory_mb = gpu.memoryUsed
                            gpu_utilization = gpu.load * 100
                    except Exception as e:
                        logger.debug(f"GPU monitoring failed: {e}")

                metrics_point = {
                    "timestamp": datetime.now().isoformat(),
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_info.rss / (1024 * 1024),
                    "memory_percent": memory_percent,
                    "disk_read_mb": disk_read_mb,
                    "disk_write_mb": disk_write_mb,
                    "gpu_memory_mb": gpu_memory_mb,
                    "gpu_utilization_percent": gpu_utilization,
                }

                self.system_metrics_history.append(metrics_point)

            except Exception as e:
                logger.warning(f"System monitoring error: {e}")

            time.sleep(0.1)  # Sample every 100ms

    def _calculate_averages(self) -> None:
        """Calculate average metrics from collected history."""
        if not self.system_metrics_history:
            return

        df = pd.DataFrame(self.system_metrics_history)

        self.metrics.cpu_percent_avg = df['cpu_percent'].mean()
        self.metrics.cpu_percent_max = df['cpu_percent'].max()
        self.metrics.memory_mb_avg = df['memory_mb'].mean()
        self.metrics.memory_mb_max = df['memory_mb'].max()
        self.metrics.memory_percent_avg = df['memory_percent'].mean()
        self.metrics.memory_percent_max = df['memory_percent'].max()

        # Calculate total I/O
        self.metrics.disk_read_mb = df['disk_read_mb'].max() if not df.empty else 0
        self.metrics.disk_write_mb = df['disk_write_mb'].max() if not df.empty else 0

        # GPU metrics (take last reading)
        if self.gpu_available and not df['gpu_memory_mb'].empty:
            self.metrics.gpu_memory_mb = df['gpu_memory_mb'].dropna().iloc[-1] if not df['gpu_memory_mb'].dropna().empty else None
            self.metrics.gpu_utilization_percent = df['gpu_utilization_percent'].dropna().iloc[-1] if not df['gpu_utilization_percent'].dropna().empty else None

    def record_step_start(self, step_name: str) -> None:
        """Record the start of a pipeline step."""
        self.metrics.step_metrics[step_name] = {
            "start_time": datetime.now(),
            "start_memory_mb": psutil.Process().memory_info().rss / (1024 * 1024),
        }

    def record_step_end(self, step_name: str, **additional_metrics: Any) -> None:
        """Record the end of a pipeline step."""
        if step_name not in self.metrics.step_metrics:
            logger.warning(f"Step {step_name} end recorded without start")
            return

        step_info = self.metrics.step_metrics[step_name]
        step_info["end_time"] = datetime.now()
        step_info["end_memory_mb"] = psutil.Process().memory_info().rss / (1024 * 1024)
        step_info["duration_seconds"] = (step_info["end_time"] - step_info["start_time"]).total_seconds()
        step_info["memory_delta_mb"] = step_info["end_memory_mb"] - step_info["start_memory_mb"]

        # Add any additional metrics
        step_info.update(additional_metrics)

        logger.debug(f"Step {step_name} completed", extra={
            "duration_seconds": step_info["duration_seconds"],
            "memory_delta_mb": step_info["memory_delta_mb"],
        })

    def analyze_performance(self) -> List[OptimizationRecommendation]:
        """Analyze collected metrics and generate optimization recommendations."""
        recommendations = []

        # Memory usage analysis
        recommendations.extend(self._analyze_memory_usage())

        # CPU usage analysis
        recommendations.extend(self._analyze_cpu_usage())

        # Step performance analysis
        recommendations.extend(self._analyze_step_performance())

        # Data processing analysis
        recommendations.extend(self._analyze_data_processing())

        # I/O analysis
        recommendations.extend(self._analyze_io_performance())

        return recommendations

    def _analyze_memory_usage(self) -> List[OptimizationRecommendation]:
        """Analyze memory usage patterns and recommend optimizations."""
        recommendations = []

        memory_percent_max = self.metrics.memory_percent_max
        memory_mb_max = self.metrics.memory_mb_max

        # High memory usage
        if memory_percent_max > 80:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="high",
                title="High Memory Usage Detected",
                description=f"Peak memory usage reached {memory_percent_max:.1f}% ({memory_mb_max:.0f} MB), indicating potential memory bottlenecks.",
                impact_estimate="high",
                implementation_effort="medium",
                suggested_action="Consider using Dask for distributed processing, implement data chunking, or use memory-efficient data types.",
                expected_benefit="Reduce memory usage by 30-50%, prevent out-of-memory errors.",
            ))

        # Memory growth analysis
        if self.system_metrics_history:
            df = pd.DataFrame(self.system_metrics_history)
            memory_trend = np.polyfit(range(len(df)), df['memory_mb'], 1)[0]
            if memory_trend > 10:  # Memory growing by >10MB per sample
                recommendations.append(OptimizationRecommendation(
                    category="memory",
                    priority="medium",
                    title="Memory Leak Suspected",
                    description="Memory usage shows consistent upward trend, indicating potential memory leaks or inefficient data handling.",
                    impact_estimate="medium",
                    implementation_effort="high",
                    suggested_action="Review data pipeline for proper cleanup, implement garbage collection calls, check for reference cycles.",
                    expected_benefit="Stabilize memory usage and prevent long-term memory growth.",
                ))

        return recommendations

    def _analyze_cpu_usage(self) -> List[OptimizationRecommendation]:
        """Analyze CPU usage patterns."""
        recommendations = []

        cpu_max = self.metrics.cpu_percent_max

        if cpu_max < 50:
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                priority="medium",
                title="Underutilized CPU Resources",
                description=f"Peak CPU usage was only {cpu_max:.1f}%, suggesting computational bottlenecks or inefficient parallelization.",
                impact_estimate="medium",
                implementation_effort="medium",
                suggested_action="Consider increasing parallel processing (n_jobs), using GPU acceleration, or optimizing single-threaded bottlenecks.",
                expected_benefit="Improve processing speed by 2-5x depending on workload.",
            ))

        return recommendations

    def _analyze_step_performance(self) -> List[OptimizationRecommendation]:
        """Analyze individual step performance."""
        recommendations = []

        for step_name, step_info in self.metrics.step_metrics.items():
            duration = step_info.get("duration_seconds", 0)

            # Long-running steps
            if duration > 300:  # 5 minutes
                recommendations.append(OptimizationRecommendation(
                    category="performance",
                    priority="high" if duration > 1800 else "medium",  # High if >30min
                    title=f"Long-running Step: {step_name}",
                    description=f"Step '{step_name}' took {duration:.1f} seconds, which may benefit from optimization.",
                    impact_estimate="high" if duration > 1800 else "medium",
                    implementation_effort="medium",
                    suggested_action=f"Profile {step_name} step in detail, consider algorithm optimization, data sampling, or distributed processing.",
                    expected_benefit="Reduce execution time by 20-60% depending on the bottleneck.",
                ))

            # Memory-intensive steps
            memory_delta = step_info.get("memory_delta_mb", 0)
            if memory_delta > 1000:  # >1GB increase
                recommendations.append(OptimizationRecommendation(
                    category="memory",
                    priority="medium",
                    title=f"Memory-intensive Step: {step_name}",
                    description=f"Step '{step_name}' increased memory usage by {memory_delta:.0f} MB.",
                    impact_estimate="medium",
                    implementation_effort="medium",
                    suggested_action=f"Implement streaming processing for {step_name}, use memory-mapped arrays, or process data in chunks.",
                    expected_benefit="Reduce peak memory usage and improve scalability.",
                ))

        return recommendations

    def _analyze_data_processing(self) -> List[OptimizationRecommendation]:
        """Analyze data processing performance."""
        recommendations = []

        # Data size analysis
        if self.metrics.input_data_size_mb > 1000:  # >1GB
            recommendations.append(OptimizationRecommendation(
                category="data",
                priority="medium",
                title="Large Dataset Processing",
                description=f"Processing {self.metrics.input_data_size_mb:.0f} MB input data. Consider optimization for large-scale processing.",
                impact_estimate="medium",
                implementation_effort="high",
                suggested_action="Implement data chunking, use Dask for distributed processing, or consider data sampling for development.",
                expected_benefit="Improve processing speed and memory efficiency for large datasets.",
            ))

        # Training time analysis
        if self.metrics.training_time > 3600:  # >1 hour
            recommendations.append(OptimizationRecommendation(
                category="model",
                priority="high",
                title="Long Training Time",
                description=f"Model training took {self.metrics.training_time:.1f} seconds, indicating potential for optimization.",
                impact_estimate="high",
                implementation_effort="medium",
                suggested_action="Consider early stopping, hyperparameter optimization, model architecture changes, or distributed training.",
                expected_benefit="Reduce training time by 50-80% while maintaining model performance.",
            ))

        return recommendations

    def _analyze_io_performance(self) -> List[OptimizationRecommendation]:
        """Analyze I/O performance patterns."""
        recommendations = []

        total_io_mb = self.metrics.disk_read_mb + self.metrics.disk_write_mb

        if total_io_mb > 1000:  # >1GB I/O
            recommendations.append(OptimizationRecommendation(
                category="io",
                priority="low",
                title="High I/O Activity",
                description=f"Total disk I/O was {total_io_mb:.0f} MB, which may impact performance.",
                impact_estimate="low",
                implementation_effort="medium",
                suggested_action="Consider using faster storage, implementing data caching, or reducing unnecessary I/O operations.",
                expected_benefit="Improve I/O performance and reduce processing time.",
            ))

        return recommendations

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate a comprehensive performance report."""
        recommendations = self.analyze_performance()

        # Create rich console report
        report_lines = []

        # Header
        report_lines.append("# Performance Analysis Report")
        report_lines.append(f"**Generated:** {datetime.now().isoformat()}")
        report_lines.append(f"**Total Duration:** {self.metrics.duration_seconds:.2f} seconds")
        report_lines.append("")

        # System Metrics Summary
        report_lines.append("## System Metrics Summary")
        report_lines.append(f"- **CPU Usage:** Avg {self.metrics.cpu_percent_avg:.1f}%, Max {self.metrics.cpu_percent_max:.1f}%")
        report_lines.append(f"- **Memory Usage:** Avg {self.metrics.memory_mb_avg:.0f} MB ({self.metrics.memory_percent_avg:.1f}%), Max {self.metrics.memory_mb_max:.0f} MB ({self.metrics.memory_percent_max:.1f}%)")
        report_lines.append(f"- **Disk I/O:** Read {self.metrics.disk_read_mb:.0f} MB, Write {self.metrics.disk_write_mb:.0f} MB")

        if self.metrics.gpu_memory_mb is not None:
            report_lines.append(f"- **GPU Memory:** {self.metrics.gpu_memory_mb:.0f} MB")
        if self.metrics.gpu_utilization_percent is not None:
            report_lines.append(f"- **GPU Utilization:** {self.metrics.gpu_utilization_percent:.1f}%")

        report_lines.append("")

        # Step Performance
        if self.metrics.step_metrics:
            report_lines.append("## Step Performance")
            for step_name, step_info in self.metrics.step_metrics.items():
                duration = step_info.get("duration_seconds", 0)
                memory_delta = step_info.get("memory_delta_mb", 0)
                report_lines.append(f"- **{step_name}:** {duration:.2f}s, Memory Δ: {memory_delta:+.0f} MB")
            report_lines.append("")

        # Recommendations
        if recommendations:
            report_lines.append("## Optimization Recommendations")
            report_lines.append("")

            # Group by priority
            by_priority = {"high": [], "medium": [], "low": []}
            for rec in recommendations:
                by_priority[rec.priority].append(rec)

            for priority in ["high", "medium", "low"]:
                if by_priority[priority]:
                    report_lines.append(f"### {priority.title()} Priority")
                    for rec in by_priority[priority]:
                        report_lines.append(f"**{rec.title}**")
                        report_lines.append(f"- *Impact:* {rec.impact_estimate} | *Effort:* {rec.implementation_effort}")
                        report_lines.append(f"- {rec.description}")
                        report_lines.append(f"- *Suggested Action:* {rec.suggested_action}")
                        report_lines.append(f"- *Expected Benefit:* {rec.expected_benefit}")
                        report_lines.append("")
        else:
            report_lines.append("## Optimization Recommendations")
            report_lines.append("✅ No optimization recommendations at this time.")
            report_lines.append("")

        report_content = "\n".join(report_lines)

        # Display to console
        console.print(Panel(report_content, title="Performance Analysis Report", border_style="blue"))

        # Save to file if requested
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Performance report saved to {output_path}")

        return report_content

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "start_time": self.metrics.start_time.isoformat(),
            "end_time": self.metrics.end_time.isoformat() if self.metrics.end_time else None,
            "duration_seconds": self.metrics.duration_seconds,
            "cpu_percent_avg": self.metrics.cpu_percent_avg,
            "cpu_percent_max": self.metrics.cpu_percent_max,
            "memory_mb_avg": self.metrics.memory_mb_avg,
            "memory_mb_max": self.metrics.memory_mb_max,
            "memory_percent_avg": self.metrics.memory_percent_avg,
            "memory_percent_max": self.metrics.memory_percent_max,
            "disk_read_mb": self.metrics.disk_read_mb,
            "disk_write_mb": self.metrics.disk_write_mb,
            "gpu_memory_mb": self.metrics.gpu_memory_mb,
            "gpu_utilization_percent": self.metrics.gpu_utilization_percent,
            "step_metrics": self.metrics.step_metrics,
            "data_loading_time": self.metrics.data_loading_time,
            "preprocessing_time": self.metrics.preprocessing_time,
            "training_time": self.metrics.training_time,
            "evaluation_time": self.metrics.evaluation_time,
            "input_data_size_mb": self.metrics.input_data_size_mb,
            "processed_data_size_mb": self.metrics.processed_data_size_mb,
            "model_size_mb": self.metrics.model_size_mb,
        }

    def save_metrics(self, output_path: Path) -> None:
        """Save performance metrics to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Performance metrics saved to {output_path}")


# Global profiler instance
_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = PerformanceProfiler()
    return _profiler


def start_performance_monitoring(enable_gpu: bool = False) -> PerformanceProfiler:
    """Start performance monitoring for the current execution."""
    global _profiler
    _profiler = PerformanceProfiler(enable_gpu_monitoring=enable_gpu)
    _profiler.start_monitoring()
    return _profiler


def stop_performance_monitoring() -> Optional[PerformanceProfiler]:
    """Stop performance monitoring and return the profiler."""
    global _profiler
    if _profiler:
        _profiler.stop_monitoring()
    return _profiler


def get_performance_report(output_path: Optional[Path] = None) -> str:
    """Generate and return a performance report."""
    profiler = get_profiler()
    return profiler.generate_report(output_path)


def record_step_performance(step_name: str, **metrics: Any) -> None:
    """Record performance metrics for a pipeline step."""
    profiler = get_profiler()
    if hasattr(profiler, 'record_step_end'):
        profiler.record_step_end(step_name, **metrics)

