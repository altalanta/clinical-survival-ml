"""
Pre-flight health check and diagnostics system.

This module provides comprehensive environment validation before pipeline execution:
- Dependency availability checks
- GPU/hardware detection
- File system access validation
- Configuration completeness verification
- Memory and disk space checks

Usage:
    from clinical_survival.diagnostics import run_health_checks, HealthCheckResult
    
    results = run_health_checks(params_config)
    if not results.all_passed:
        print(results.summary())
        sys.exit(1)
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from clinical_survival.logging_config import get_logger

# Module logger
logger = get_logger(__name__)

# Console for output
console = Console()


class CheckStatus(Enum):
    """Status of a health check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single health check."""
    name: str
    status: CheckStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None


@dataclass
class HealthCheckResult:
    """Aggregate result of all health checks."""
    checks: List[CheckResult] = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        """Check if all checks passed (warnings are OK)."""
        return all(c.status in (CheckStatus.PASSED, CheckStatus.WARNING, CheckStatus.SKIPPED) 
                   for c in self.checks)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(c.status == CheckStatus.WARNING for c in self.checks)
    
    @property
    def failed_checks(self) -> List[CheckResult]:
        """Get list of failed checks."""
        return [c for c in self.checks if c.status == CheckStatus.FAILED]
    
    @property
    def warning_checks(self) -> List[CheckResult]:
        """Get list of warning checks."""
        return [c for c in self.checks if c.status == CheckStatus.WARNING]
    
    def add(self, check: CheckResult) -> None:
        """Add a check result."""
        self.checks.append(check)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "all_passed": self.all_passed,
            "has_warnings": self.has_warnings,
            "total_checks": len(self.checks),
            "passed": len([c for c in self.checks if c.status == CheckStatus.PASSED]),
            "warnings": len(self.warning_checks),
            "failed": len(self.failed_checks),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "suggestion": c.suggestion,
                }
                for c in self.checks
            ],
        }


# =============================================================================
# Individual Health Checks
# =============================================================================


def check_python_version() -> CheckResult:
    """Check Python version compatibility."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        return CheckResult(
            name="Python Version",
            status=CheckStatus.FAILED,
            message=f"Python {version_str} is not supported",
            details={"current": version_str, "required": ">=3.10"},
            suggestion="Upgrade to Python 3.10 or later",
        )
    
    return CheckResult(
        name="Python Version",
        status=CheckStatus.PASSED,
        message=f"Python {version_str}",
        details={"version": version_str},
    )


def check_core_dependencies() -> CheckResult:
    """Check that core dependencies are available."""
    required = [
        ("numpy", "np"),
        ("pandas", "pd"),
        ("sklearn", "sklearn"),
        ("sksurv", "sksurv"),
        ("pydantic", "pydantic"),
        ("rich", "rich"),
        ("yaml", "yaml"),
    ]
    
    missing = []
    versions = {}
    
    for module_name, import_name in required:
        try:
            mod = importlib.import_module(module_name)
            versions[module_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            missing.append(module_name)
    
    if missing:
        return CheckResult(
            name="Core Dependencies",
            status=CheckStatus.FAILED,
            message=f"Missing: {', '.join(missing)}",
            details={"missing": missing, "installed": versions},
            suggestion=f"Install missing packages: pip install {' '.join(missing)}",
        )
    
    return CheckResult(
        name="Core Dependencies",
        status=CheckStatus.PASSED,
        message=f"{len(versions)} core packages installed",
        details={"versions": versions},
    )


def check_optional_dependencies() -> CheckResult:
    """Check optional dependencies for enhanced functionality."""
    optional = [
        ("xgboost", "GPU-accelerated models"),
        ("mlflow", "Experiment tracking"),
        ("optuna", "Hyperparameter tuning"),
        ("shap", "Model explainability"),
        ("great_expectations", "Data validation"),
        ("streamlit", "Interactive dashboard"),
        ("fastapi", "REST API"),
        ("evidently", "Drift monitoring"),
    ]
    
    available = []
    missing = []
    
    for module_name, feature in optional:
        try:
            importlib.import_module(module_name)
            available.append(module_name)
        except ImportError:
            missing.append((module_name, feature))
    
    if missing:
        features_msg = ", ".join(f"{m} ({f})" for m, f in missing[:3])
        return CheckResult(
            name="Optional Dependencies",
            status=CheckStatus.WARNING,
            message=f"Some optional features unavailable: {features_msg}",
            details={"available": available, "missing": [m for m, _ in missing]},
            suggestion="Install optional packages for full functionality",
        )
    
    return CheckResult(
        name="Optional Dependencies",
        status=CheckStatus.PASSED,
        message=f"All {len(optional)} optional packages available",
        details={"available": available},
    )


def check_gpu_availability() -> CheckResult:
    """Check GPU availability for accelerated training."""
    gpu_info = {"cuda_available": False, "devices": []}
    
    # Check CUDA via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["cuda_available"] = True
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                gpu_info["devices"].append({
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
                })
    except ImportError:
        pass
    
    # Check XGBoost GPU
    try:
        import xgboost as xgb
        gpu_info["xgboost_gpu"] = "gpu_hist" in xgb.XGBClassifier().get_params().get("tree_method", "")
    except (ImportError, Exception):
        gpu_info["xgboost_gpu"] = False
    
    if gpu_info["cuda_available"]:
        return CheckResult(
            name="GPU Availability",
            status=CheckStatus.PASSED,
            message=f"{gpu_info['device_count']} GPU(s) available",
            details=gpu_info,
        )
    
    return CheckResult(
        name="GPU Availability",
        status=CheckStatus.WARNING,
        message="No GPU detected, using CPU only",
        details=gpu_info,
        suggestion="GPU training will be disabled. Install CUDA for acceleration.",
    )


def check_memory_available() -> CheckResult:
    """Check available system memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        
        details = {
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "percent_used": mem.percent,
        }
        
        if available_gb < 2:
            return CheckResult(
                name="System Memory",
                status=CheckStatus.WARNING,
                message=f"Low memory: {available_gb:.1f}GB available",
                details=details,
                suggestion="Close other applications or reduce batch sizes",
            )
        
        return CheckResult(
            name="System Memory",
            status=CheckStatus.PASSED,
            message=f"{available_gb:.1f}GB available of {total_gb:.1f}GB",
            details=details,
        )
    except ImportError:
        return CheckResult(
            name="System Memory",
            status=CheckStatus.SKIPPED,
            message="psutil not available for memory check",
            suggestion="Install psutil for memory monitoring",
        )


def check_disk_space(output_dir: Optional[Path] = None) -> CheckResult:
    """Check available disk space in output directory."""
    check_path = output_dir or Path(".")
    
    try:
        total, used, free = shutil.disk_usage(check_path)
        free_gb = free / (1024 ** 3)
        total_gb = total / (1024 ** 3)
        
        details = {
            "path": str(check_path),
            "total_gb": round(total_gb, 2),
            "free_gb": round(free_gb, 2),
            "percent_used": round((used / total) * 100, 1),
        }
        
        if free_gb < 1:
            return CheckResult(
                name="Disk Space",
                status=CheckStatus.WARNING,
                message=f"Low disk space: {free_gb:.1f}GB free",
                details=details,
                suggestion="Free up disk space or change output directory",
            )
        
        return CheckResult(
            name="Disk Space",
            status=CheckStatus.PASSED,
            message=f"{free_gb:.1f}GB free of {total_gb:.1f}GB",
            details=details,
        )
    except Exception as e:
        return CheckResult(
            name="Disk Space",
            status=CheckStatus.WARNING,
            message=f"Could not check disk space: {e}",
        )


def check_data_file(data_path: Optional[Path]) -> CheckResult:
    """Check that the data file exists and is readable."""
    if data_path is None:
        return CheckResult(
            name="Data File",
            status=CheckStatus.SKIPPED,
            message="No data path specified",
        )
    
    path = Path(data_path)
    
    if not path.exists():
        return CheckResult(
            name="Data File",
            status=CheckStatus.FAILED,
            message=f"Data file not found: {path}",
            details={"path": str(path)},
            suggestion=f"Check the path in config or run 'dvc pull' if using DVC",
        )
    
    if not path.is_file():
        return CheckResult(
            name="Data File",
            status=CheckStatus.FAILED,
            message=f"Path is not a file: {path}",
            details={"path": str(path)},
        )
    
    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    
    # Try to read header
    try:
        import pandas as pd
        df_head = pd.read_csv(path, nrows=5)
        n_cols = len(df_head.columns)
        
        return CheckResult(
            name="Data File",
            status=CheckStatus.PASSED,
            message=f"Found {path.name} ({size_mb:.1f}MB, {n_cols} columns)",
            details={
                "path": str(path),
                "size_mb": round(size_mb, 2),
                "columns": list(df_head.columns),
            },
        )
    except Exception as e:
        return CheckResult(
            name="Data File",
            status=CheckStatus.WARNING,
            message=f"File exists but could not read: {e}",
            details={"path": str(path), "error": str(e)},
        )


def check_output_directory(output_dir: Optional[Path]) -> CheckResult:
    """Check that output directory is writable."""
    if output_dir is None:
        output_dir = Path("results")
    
    path = Path(output_dir)
    
    try:
        path.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        test_file = path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        
        return CheckResult(
            name="Output Directory",
            status=CheckStatus.PASSED,
            message=f"Writable: {path}",
            details={"path": str(path)},
        )
    except PermissionError:
        return CheckResult(
            name="Output Directory",
            status=CheckStatus.FAILED,
            message=f"No write permission: {path}",
            details={"path": str(path)},
            suggestion="Change output directory or fix permissions",
        )
    except Exception as e:
        return CheckResult(
            name="Output Directory",
            status=CheckStatus.FAILED,
            message=f"Cannot create directory: {e}",
            details={"path": str(path), "error": str(e)},
        )


def check_mlflow_connection(tracking_uri: Optional[str] = None) -> CheckResult:
    """Check MLflow server connection."""
    try:
        import mlflow
    except ImportError:
        return CheckResult(
            name="MLflow Connection",
            status=CheckStatus.SKIPPED,
            message="MLflow not installed",
        )
    
    if not tracking_uri:
        return CheckResult(
            name="MLflow Connection",
            status=CheckStatus.SKIPPED,
            message="MLflow tracking not configured",
        )
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        # Try to list experiments as a connection test
        mlflow.search_experiments(max_results=1)
        
        return CheckResult(
            name="MLflow Connection",
            status=CheckStatus.PASSED,
            message=f"Connected to {tracking_uri}",
            details={"uri": tracking_uri},
        )
    except Exception as e:
        return CheckResult(
            name="MLflow Connection",
            status=CheckStatus.WARNING,
            message=f"Cannot connect to MLflow: {e}",
            details={"uri": tracking_uri, "error": str(e)},
            suggestion="MLflow tracking will fall back to local storage",
        )


# =============================================================================
# Main Health Check Runner
# =============================================================================


def run_health_checks(
    params_config: Optional[Any] = None,
    verbose: bool = True,
) -> HealthCheckResult:
    """
    Run all health checks and return aggregated results.
    
    Args:
        params_config: Optional ParamsConfig for context-aware checks
        verbose: Whether to print results to console
        
    Returns:
        HealthCheckResult with all check results
    """
    result = HealthCheckResult()
    
    # Core checks
    result.add(check_python_version())
    result.add(check_core_dependencies())
    result.add(check_optional_dependencies())
    result.add(check_gpu_availability())
    result.add(check_memory_available())
    
    # Config-dependent checks
    if params_config is not None:
        data_path = Path(params_config.paths.data_csv) if hasattr(params_config, 'paths') else None
        output_dir = Path(params_config.paths.outdir) if hasattr(params_config, 'paths') else None
        
        result.add(check_data_file(data_path))
        result.add(check_output_directory(output_dir))
        result.add(check_disk_space(output_dir))
        
        # MLflow check
        if hasattr(params_config, 'mlflow_tracking') and params_config.mlflow_tracking.enabled:
            result.add(check_mlflow_connection(params_config.mlflow_tracking.tracking_uri))
    else:
        result.add(check_output_directory(None))
        result.add(check_disk_space(None))
    
    # Log results
    for check in result.checks:
        log_level = {
            CheckStatus.PASSED: "debug",
            CheckStatus.WARNING: "warning",
            CheckStatus.FAILED: "error",
            CheckStatus.SKIPPED: "debug",
        }.get(check.status, "info")
        
        getattr(logger, log_level)(
            f"Health check '{check.name}': {check.status.value}",
            extra={"check": check.name, "status": check.status.value, "message": check.message},
        )
    
    if verbose:
        print_health_check_results(result)
    
    return result


def print_health_check_results(result: HealthCheckResult) -> None:
    """Print health check results in a formatted table."""
    # Create status indicators
    status_icons = {
        CheckStatus.PASSED: "[green]✓[/green]",
        CheckStatus.WARNING: "[yellow]⚠[/yellow]",
        CheckStatus.FAILED: "[red]✗[/red]",
        CheckStatus.SKIPPED: "[dim]○[/dim]",
    }
    
    # Create table
    table = Table(title="🏥 Health Check Results", show_header=True)
    table.add_column("Status", width=3, justify="center")
    table.add_column("Check", style="cyan")
    table.add_column("Result")
    
    for check in result.checks:
        icon = status_icons[check.status]
        table.add_row(icon, check.name, check.message)
    
    console.print()
    console.print(table)
    
    # Summary
    if result.all_passed:
        if result.has_warnings:
            console.print(
                Panel(
                    "[yellow]All checks passed with warnings. Pipeline can proceed.[/yellow]",
                    title="Summary",
                    border_style="yellow",
                )
            )
        else:
            console.print(
                Panel(
                    "[green]All checks passed! Pipeline ready to run.[/green]",
                    title="Summary",
                    border_style="green",
                )
            )
    else:
        failed_names = [c.name for c in result.failed_checks]
        suggestions = [c.suggestion for c in result.failed_checks if c.suggestion]
        
        text = Text()
        text.append(f"Failed checks: {', '.join(failed_names)}\n\n", style="red")
        text.append("Suggestions:\n", style="yellow")
        for suggestion in suggestions:
            text.append(f"  → {suggestion}\n", style="dim")
        
        console.print(
            Panel(text, title="❌ Health Check Failed", border_style="red")
        )
    
    console.print()









