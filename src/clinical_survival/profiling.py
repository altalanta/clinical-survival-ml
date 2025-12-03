"""
Pipeline performance profiling and metrics collection.

This module provides:
- Decorators for timing function execution
- Memory profiling utilities
- Pipeline step performance tracking
- Performance report generation
- Integration with logging system
"""

from __future__ import annotations

import functools
import gc
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

from rich.console import Console
from rich.table import Table

from clinical_survival.logging_config import get_logger

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Module logger
logger = get_logger(__name__)

# Console for rich output
console = Console()


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StepProfile:
    """Performance profile for a single pipeline step."""
    
    step_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    memory_start_mb: float = 0.0
    memory_peak_mb: float = 0.0
    memory_end_mb: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_delta_mb(self) -> float:
        """Memory change from start to end."""
        return self.memory_end_mb - self.memory_start_mb


@dataclass
class PipelineProfile:
    """Complete performance profile for a pipeline run."""
    
    pipeline_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: List[StepProfile] = field(default_factory=list)
    total_duration_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    success: bool = True
    correlation_id: Optional[str] = None
    
    def add_step(self, step: StepProfile) -> None:
        """Add a step profile to the pipeline."""
        self.steps.append(step)
        if step.memory_peak_mb > self.peak_memory_mb:
            self.peak_memory_mb = step.memory_peak_mb
    
    def get_slowest_steps(self, n: int = 5) -> List[StepProfile]:
        """Get the N slowest steps."""
        return sorted(self.steps, key=lambda s: s.duration_seconds, reverse=True)[:n]
    
    def get_memory_intensive_steps(self, n: int = 5) -> List[StepProfile]:
        """Get the N most memory-intensive steps."""
        return sorted(self.steps, key=lambda s: s.memory_peak_mb, reverse=True)[:n]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary for serialization."""
        return {
            "pipeline_name": self.pipeline_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_seconds": self.total_duration_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "success": self.success,
            "correlation_id": self.correlation_id,
            "n_steps": len(self.steps),
            "steps": [
                {
                    "name": s.step_name,
                    "duration_seconds": s.duration_seconds,
                    "memory_peak_mb": s.memory_peak_mb,
                    "success": s.success,
                }
                for s in self.steps
            ],
        }


# =============================================================================
# Profiler Class
# =============================================================================


class PipelineProfiler:
    """
    Tracks performance metrics across pipeline execution.
    
    Usage:
        profiler = PipelineProfiler("training_pipeline")
        
        with profiler.profile_step("load_data"):
            df = load_data(path)
        
        with profiler.profile_step("preprocess"):
            X, y = preprocess(df)
        
        profiler.finish()
        profiler.print_summary()
    """
    
    def __init__(
        self,
        pipeline_name: str,
        correlation_id: Optional[str] = None,
        track_memory: bool = True,
    ):
        """
        Initialize the profiler.
        
        Args:
            pipeline_name: Name of the pipeline being profiled
            correlation_id: Optional correlation ID for tracking
            track_memory: Whether to track memory usage (adds overhead)
        """
        self.pipeline_name = pipeline_name
        self.correlation_id = correlation_id
        self.track_memory = track_memory
        self._profile = PipelineProfile(
            pipeline_name=pipeline_name,
            start_time=datetime.utcnow(),
            correlation_id=correlation_id,
        )
        self._current_step: Optional[StepProfile] = None
        
        if track_memory:
            tracemalloc.start()
        
        logger.debug(
            f"Started profiling pipeline '{pipeline_name}'",
            extra={"correlation_id": correlation_id},
        )
    
    @contextmanager
    def profile_step(self, step_name: str, **metadata: Any):
        """
        Context manager to profile a single step.
        
        Args:
            step_name: Name of the step being profiled
            **metadata: Additional metadata to attach to the step profile
        """
        step = StepProfile(
            step_name=step_name,
            start_time=datetime.utcnow(),
            metadata=metadata,
        )
        
        # Capture starting memory
        if self.track_memory:
            gc.collect()
            current, _ = tracemalloc.get_traced_memory()
            step.memory_start_mb = current / 1024 / 1024
        
        start_time = time.perf_counter()
        self._current_step = step
        
        try:
            yield step
            step.success = True
        except Exception as e:
            step.success = False
            step.error_message = str(e)
            raise
        finally:
            # Capture duration
            step.duration_seconds = time.perf_counter() - start_time
            step.end_time = datetime.utcnow()
            
            # Capture ending memory
            if self.track_memory:
                gc.collect()
                current, peak = tracemalloc.get_traced_memory()
                step.memory_end_mb = current / 1024 / 1024
                step.memory_peak_mb = peak / 1024 / 1024
            
            self._profile.add_step(step)
            self._current_step = None
            
            # Log step completion
            logger.info(
                f"Step '{step_name}' completed",
                extra={
                    "step": step_name,
                    "duration_seconds": round(step.duration_seconds, 3),
                    "memory_peak_mb": round(step.memory_peak_mb, 1),
                    "success": step.success,
                },
            )
    
    def finish(self) -> PipelineProfile:
        """
        Finish profiling and return the complete profile.
        
        Returns:
            Complete pipeline profile
        """
        self._profile.end_time = datetime.utcnow()
        self._profile.total_duration_seconds = sum(
            s.duration_seconds for s in self._profile.steps
        )
        self._profile.success = all(s.success for s in self._profile.steps)
        
        if self.track_memory:
            tracemalloc.stop()
        
        logger.info(
            f"Pipeline '{self.pipeline_name}' profiling complete",
            extra={
                "total_duration_seconds": round(self._profile.total_duration_seconds, 3),
                "n_steps": len(self._profile.steps),
                "peak_memory_mb": round(self._profile.peak_memory_mb, 1),
                "success": self._profile.success,
            },
        )
        
        return self._profile
    
    def print_summary(self) -> None:
        """Print a formatted summary of the pipeline profile."""
        profile = self._profile
        
        # Header
        console.print()
        console.print(
            f"[bold cyan]Pipeline Profile: {profile.pipeline_name}[/bold cyan]"
        )
        console.print(f"Correlation ID: {profile.correlation_id or 'N/A'}")
        console.print(f"Status: {'✅ Success' if profile.success else '❌ Failed'}")
        console.print()
        
        # Summary table
        summary_table = Table(title="Summary", show_header=False)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")
        
        summary_table.add_row("Total Duration", f"{profile.total_duration_seconds:.2f}s")
        summary_table.add_row("Peak Memory", f"{profile.peak_memory_mb:.1f} MB")
        summary_table.add_row("Steps Executed", str(len(profile.steps)))
        summary_table.add_row("Start Time", profile.start_time.strftime("%H:%M:%S"))
        if profile.end_time:
            summary_table.add_row("End Time", profile.end_time.strftime("%H:%M:%S"))
        
        console.print(summary_table)
        console.print()
        
        # Steps table
        steps_table = Table(title="Step Breakdown")
        steps_table.add_column("Step", style="cyan")
        steps_table.add_column("Duration", justify="right")
        steps_table.add_column("% of Total", justify="right")
        steps_table.add_column("Memory Peak", justify="right")
        steps_table.add_column("Status", justify="center")
        
        for step in profile.steps:
            pct = (step.duration_seconds / profile.total_duration_seconds * 100 
                   if profile.total_duration_seconds > 0 else 0)
            status = "✅" if step.success else "❌"
            steps_table.add_row(
                step.step_name,
                f"{step.duration_seconds:.2f}s",
                f"{pct:.1f}%",
                f"{step.memory_peak_mb:.1f} MB",
                status,
            )
        
        console.print(steps_table)
        console.print()
    
    def save_report(self, output_path: Path) -> None:
        """
        Save the profile as a JSON report.
        
        Args:
            output_path: Path to save the report
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w") as f:
            json.dump(self._profile.to_dict(), f, indent=2)
        
        logger.info(f"Profile report saved to {output_path}")


# =============================================================================
# Decorators
# =============================================================================


def profile_function(
    name: Optional[str] = None,
    track_memory: bool = False,
    log_args: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to profile a function's execution time and optionally memory.
    
    Args:
        name: Custom name for the profile (defaults to function name)
        track_memory: Whether to track memory usage
        log_args: Whether to log function arguments
        
    Returns:
        Decorated function
        
    Example:
        @profile_function(track_memory=True)
        def expensive_computation(data):
            ...
    """
    def decorator(func: F) -> F:
        profile_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            extra = {"function": profile_name}
            if log_args:
                extra["args_count"] = len(args)
                extra["kwargs_keys"] = list(kwargs.keys())
            
            # Start timing
            start_time = time.perf_counter()
            
            # Start memory tracking if enabled
            if track_memory:
                gc.collect()
                tracemalloc.start()
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration = time.perf_counter() - start_time
                extra["duration_seconds"] = round(duration, 4)
                extra["success"] = True
                
                # Get memory stats if tracking
                if track_memory:
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    extra["memory_peak_mb"] = round(peak / 1024 / 1024, 2)
                
                logger.debug(f"Function '{profile_name}' completed", extra=extra)
                
                return result
                
            except Exception as e:
                duration = time.perf_counter() - start_time
                extra["duration_seconds"] = round(duration, 4)
                extra["success"] = False
                extra["error"] = str(e)
                
                if track_memory:
                    tracemalloc.stop()
                
                logger.warning(f"Function '{profile_name}' failed", extra=extra)
                raise
        
        return wrapper  # type: ignore
    
    return decorator


def timed(func: F) -> F:
    """
    Simple timing decorator that logs execution time.
    
    Example:
        @timed
        def my_function():
            ...
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(
                f"{func.__name__} completed in {elapsed:.4f}s",
                extra={"function": func.__name__, "duration_seconds": elapsed},
            )
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.debug(
                f"{func.__name__} failed after {elapsed:.4f}s",
                extra={"function": func.__name__, "duration_seconds": elapsed},
            )
            raise
    
    return wrapper  # type: ignore


# =============================================================================
# Utility Functions
# =============================================================================


@contextmanager
def profile_block(name: str, track_memory: bool = False):
    """
    Context manager for profiling a code block.
    
    Args:
        name: Name for the profiled block
        track_memory: Whether to track memory
        
    Example:
        with profile_block("data_loading", track_memory=True) as stats:
            df = pd.read_csv(path)
        print(f"Loaded in {stats['duration']}s")
    """
    stats: Dict[str, Any] = {"name": name}
    start_time = time.perf_counter()
    
    if track_memory:
        gc.collect()
        tracemalloc.start()
    
    try:
        yield stats
    finally:
        stats["duration"] = time.perf_counter() - start_time
        
        if track_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            stats["memory_peak_mb"] = peak / 1024 / 1024
        
        logger.debug(
            f"Block '{name}' completed",
            extra={
                "block": name,
                "duration_seconds": round(stats["duration"], 4),
                **({
                    "memory_peak_mb": round(stats.get("memory_peak_mb", 0), 2)
                } if track_memory else {}),
            },
        )


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import sys
    
    # Try to get process memory from psutil if available
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        pass
    
    # Fallback to sys.getsizeof for main objects
    return sum(sys.getsizeof(obj) for obj in gc.get_objects()) / 1024 / 1024


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to human-readable string."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f}μs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

