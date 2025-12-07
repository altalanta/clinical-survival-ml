"""
Data profiling module for exploratory data analysis.

This module provides automated data profiling:
- Summary statistics for all columns
- Missing value analysis
- Survival-specific statistics (event rates, time distributions)
- Correlation analysis
- Feature distribution plots
- Data quality report generation

Usage:
    from clinical_survival.data_profiling import DataProfiler
    
    profiler = DataProfiler(df, time_col="time", event_col="event")
    report = profiler.generate_report()
    profiler.save_report("data_profile.html")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from clinical_survival.logging_config import get_logger
from clinical_survival.utils import ensure_dir

# Module logger
logger = get_logger(__name__)


@dataclass
class ColumnProfile:
    """Profile for a single column."""
    
    name: str
    dtype: str
    n_total: int
    n_missing: int
    missing_pct: float
    n_unique: int
    
    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Categorical stats
    top_values: Optional[List[Tuple[str, int]]] = None
    
    # Flags
    is_constant: bool = False
    is_highly_missing: bool = False
    is_highly_cardinal: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class SurvivalProfile:
    """Survival-specific statistics."""
    
    n_events: int = 0
    n_censored: int = 0
    event_rate: float = 0.0
    
    time_min: float = 0.0
    time_max: float = 0.0
    time_mean: float = 0.0
    time_median: float = 0.0
    time_std: float = 0.0
    
    # Time by event status
    event_time_mean: Optional[float] = None
    censored_time_mean: Optional[float] = None
    
    # Kaplan-Meier estimates
    km_median_survival: Optional[float] = None
    km_1year_survival: Optional[float] = None
    km_5year_survival: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class DataProfile:
    """Complete data profile."""
    
    n_rows: int = 0
    n_columns: int = 0
    memory_mb: float = 0.0
    
    column_profiles: List[ColumnProfile] = field(default_factory=list)
    survival_profile: Optional[SurvivalProfile] = None
    
    # Correlations
    numeric_correlations: Optional[pd.DataFrame] = None
    
    # Warnings and issues
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    generated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": {
                "n_rows": self.n_rows,
                "n_columns": self.n_columns,
                "memory_mb": self.memory_mb,
                "generated_at": self.generated_at,
            },
            "columns": [c.to_dict() for c in self.column_profiles],
            "survival": self.survival_profile.to_dict() if self.survival_profile else None,
            "warnings": self.warnings,
        }


class DataProfiler:
    """
    Automated data profiling for survival analysis datasets.
    
    Usage:
        profiler = DataProfiler(
            df,
            time_col="time",
            event_col="event",
            categorical_cols=["stage", "treatment"],
        )
        
        # Generate profile
        profile = profiler.generate_profile()
        
        # Print summary
        profiler.print_summary()
        
        # Save report
        profiler.save_report("data_profile.html")
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: Optional[str] = None,
        event_col: Optional[str] = None,
        categorical_cols: Optional[List[str]] = None,
        id_col: Optional[str] = None,
    ):
        """
        Initialize the profiler.
        
        Args:
            df: DataFrame to profile
            time_col: Name of time-to-event column
            event_col: Name of event indicator column
            categorical_cols: List of categorical column names
            id_col: Column to exclude from analysis (ID column)
        """
        self.df = df.copy()
        self.time_col = time_col
        self.event_col = event_col
        self.categorical_cols = categorical_cols or []
        self.id_col = id_col
        
        self._profile: Optional[DataProfile] = None
    
    def generate_profile(self) -> DataProfile:
        """
        Generate complete data profile.
        
        Returns:
            DataProfile with all statistics
        """
        from datetime import datetime
        
        logger.info(f"Profiling data: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        profile = DataProfile(
            n_rows=len(self.df),
            n_columns=len(self.df.columns),
            memory_mb=self.df.memory_usage(deep=True).sum() / 1024 / 1024,
            generated_at=datetime.utcnow().isoformat(),
        )
        
        # Profile each column
        for col in self.df.columns:
            if col == self.id_col:
                continue
            
            col_profile = self._profile_column(col)
            profile.column_profiles.append(col_profile)
            
            # Add warnings
            if col_profile.is_highly_missing:
                profile.warnings.append(f"Column '{col}' has {col_profile.missing_pct:.1f}% missing values")
            if col_profile.is_constant:
                profile.warnings.append(f"Column '{col}' has constant value")
        
        # Survival-specific profile
        if self.time_col and self.event_col:
            profile.survival_profile = self._profile_survival()
        
        # Correlations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            profile.numeric_correlations = self.df[numeric_cols].corr()
        
        self._profile = profile
        
        logger.info("Data profiling complete")
        
        return profile
    
    def _profile_column(self, col: str) -> ColumnProfile:
        """Profile a single column."""
        series = self.df[col]
        
        profile = ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            n_total=len(series),
            n_missing=series.isna().sum(),
            missing_pct=series.isna().mean() * 100,
            n_unique=series.nunique(),
        )
        
        # Flags
        profile.is_highly_missing = profile.missing_pct > 50
        profile.is_constant = profile.n_unique <= 1
        profile.is_highly_cardinal = profile.n_unique > len(series) * 0.9
        
        # Type-specific stats
        if pd.api.types.is_numeric_dtype(series):
            self._add_numeric_stats(profile, series)
        else:
            self._add_categorical_stats(profile, series)
        
        return profile
    
    def _add_numeric_stats(self, profile: ColumnProfile, series: pd.Series) -> None:
        """Add numeric statistics to column profile."""
        clean = series.dropna()
        
        if len(clean) == 0:
            return
        
        profile.mean = float(clean.mean())
        profile.std = float(clean.std())
        profile.min = float(clean.min())
        profile.max = float(clean.max())
        profile.median = float(clean.median())
        profile.q25 = float(clean.quantile(0.25))
        profile.q75 = float(clean.quantile(0.75))
        
        try:
            profile.skewness = float(clean.skew())
            profile.kurtosis = float(clean.kurtosis())
        except Exception:
            pass
    
    def _add_categorical_stats(self, profile: ColumnProfile, series: pd.Series) -> None:
        """Add categorical statistics to column profile."""
        value_counts = series.value_counts()
        
        # Top 10 values
        top_n = min(10, len(value_counts))
        profile.top_values = [
            (str(val), int(count)) 
            for val, count in value_counts.head(top_n).items()
        ]
    
    def _profile_survival(self) -> SurvivalProfile:
        """Generate survival-specific profile."""
        profile = SurvivalProfile()
        
        time = self.df[self.time_col]
        event = self.df[self.event_col]
        
        profile.n_events = int(event.sum())
        profile.n_censored = int((event == 0).sum())
        profile.event_rate = profile.n_events / len(event) if len(event) > 0 else 0
        
        profile.time_min = float(time.min())
        profile.time_max = float(time.max())
        profile.time_mean = float(time.mean())
        profile.time_median = float(time.median())
        profile.time_std = float(time.std())
        
        # Time by event status
        profile.event_time_mean = float(time[event == 1].mean()) if profile.n_events > 0 else None
        profile.censored_time_mean = float(time[event == 0].mean()) if profile.n_censored > 0 else None
        
        # Kaplan-Meier estimates
        try:
            from lifelines import KaplanMeierFitter
            
            kmf = KaplanMeierFitter()
            kmf.fit(time, event_observed=event)
            
            profile.km_median_survival = float(kmf.median_survival_time_)
            
            # Survival at 1 year and 5 years
            for t, attr in [(365, "km_1year_survival"), (1825, "km_5year_survival")]:
                if t <= time.max():
                    prob = kmf.survival_function_at_times(t).values[0]
                    setattr(profile, attr, float(prob))
        except Exception as e:
            logger.warning(f"Could not compute KM estimates: {e}")
        
        return profile
    
    def print_summary(self) -> None:
        """Print a summary of the data profile."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        if self._profile is None:
            self.generate_profile()
        
        profile = self._profile
        
        # Header
        console.print()
        console.print(Panel(
            f"[bold]Data Profile Summary[/bold]\n"
            f"Rows: {profile.n_rows:,}  |  Columns: {profile.n_columns}  |  "
            f"Memory: {profile.memory_mb:.2f} MB",
            title="📊 Dataset Overview",
        ))
        
        # Survival summary
        if profile.survival_profile:
            sp = profile.survival_profile
            console.print()
            console.print("[bold cyan]Survival Statistics:[/bold cyan]")
            console.print(f"  Events: {sp.n_events:,} ({sp.event_rate:.1%})")
            console.print(f"  Censored: {sp.n_censored:,} ({1-sp.event_rate:.1%})")
            console.print(f"  Time range: {sp.time_min:.0f} - {sp.time_max:.0f} days")
            console.print(f"  Median follow-up: {sp.time_median:.0f} days")
            
            if sp.km_median_survival:
                console.print(f"  KM median survival: {sp.km_median_survival:.0f} days")
            if sp.km_1year_survival:
                console.print(f"  1-year survival: {sp.km_1year_survival:.1%}")
            if sp.km_5year_survival:
                console.print(f"  5-year survival: {sp.km_5year_survival:.1%}")
        
        # Column summary table
        console.print()
        table = Table(title="Column Summary")
        table.add_column("Column", style="cyan")
        table.add_column("Type")
        table.add_column("Missing", justify="right")
        table.add_column("Unique", justify="right")
        table.add_column("Stats")
        
        for col in profile.column_profiles:
            missing_style = "red" if col.is_highly_missing else ""
            
            if col.mean is not None:
                stats = f"mean={col.mean:.2f}, std={col.std:.2f}"
            elif col.top_values:
                top = col.top_values[0]
                stats = f"top: {top[0]} ({top[1]})"
            else:
                stats = ""
            
            table.add_row(
                col.name,
                col.dtype,
                f"[{missing_style}]{col.missing_pct:.1f}%[/{missing_style}]" if missing_style else f"{col.missing_pct:.1f}%",
                str(col.n_unique),
                stats[:50],
            )
        
        console.print(table)
        
        # Warnings
        if profile.warnings:
            console.print()
            console.print("[bold yellow]⚠ Warnings:[/bold yellow]")
            for warning in profile.warnings[:10]:
                console.print(f"  • {warning}")
        
        console.print()
    
    def save_report(
        self,
        output_path: Union[str, Path],
        format: str = "html",
    ) -> Path:
        """
        Save profile report to file.
        
        Args:
            output_path: Output file path
            format: Output format (html, json, or csv)
            
        Returns:
            Path to saved report
        """
        if self._profile is None:
            self.generate_profile()
        
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(self._profile.to_dict(), f, indent=2, default=str)
        
        elif format == "csv":
            # Save column profiles as CSV
            rows = [col.to_dict() for col in self._profile.column_profiles]
            pd.DataFrame(rows).to_csv(output_path, index=False)
        
        elif format == "html":
            self._save_html_report(output_path)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved profile report to {output_path}")
        
        return output_path
    
    def _save_html_report(self, output_path: Path) -> None:
        """Generate and save HTML report."""
        profile = self._profile
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Profile Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card .value {{ font-size: 2em; font-weight: bold; color: #2980b9; }}
        .summary-card .label {{ color: #7f8c8d; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .warning {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px 15px; margin: 10px 0; }}
        .bar {{ background: #3498db; height: 8px; border-radius: 4px; }}
        .bar-bg {{ background: #ecf0f1; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Data Profile Report</h1>
        <p>Generated: {profile.generated_at}</p>
        
        <div class="summary-grid">
            <div class="summary-card">
                <div class="value">{profile.n_rows:,}</div>
                <div class="label">Rows</div>
            </div>
            <div class="summary-card">
                <div class="value">{profile.n_columns}</div>
                <div class="label">Columns</div>
            </div>
            <div class="summary-card">
                <div class="value">{profile.memory_mb:.1f} MB</div>
                <div class="label">Memory</div>
            </div>
            <div class="summary-card">
                <div class="value">{profile.survival_profile.event_rate:.1%}</div>
                <div class="label">Event Rate</div>
            </div>
        </div>
"""
        
        # Survival section
        if profile.survival_profile:
            sp = profile.survival_profile
            html += f"""
        <h2>Survival Statistics</h2>
        <table>
            <tr><td>Events</td><td>{sp.n_events:,}</td></tr>
            <tr><td>Censored</td><td>{sp.n_censored:,}</td></tr>
            <tr><td>Time Range</td><td>{sp.time_min:.0f} - {sp.time_max:.0f} days</td></tr>
            <tr><td>Median Follow-up</td><td>{sp.time_median:.0f} days</td></tr>
            {"<tr><td>KM Median Survival</td><td>" + f"{sp.km_median_survival:.0f} days</td></tr>" if sp.km_median_survival else ""}
        </table>
"""
        
        # Columns table
        html += """
        <h2>Column Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Missing</th>
                    <th>Unique</th>
                    <th>Summary</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for col in profile.column_profiles:
            if col.mean is not None:
                summary = f"μ={col.mean:.2f}, σ={col.std:.2f}"
            elif col.top_values:
                summary = f"Top: {col.top_values[0][0]}"
            else:
                summary = ""
            
            missing_style = 'color: red;' if col.is_highly_missing else ''
            
            html += f"""
                <tr>
                    <td>{col.name}</td>
                    <td>{col.dtype}</td>
                    <td style="{missing_style}">{col.missing_pct:.1f}%</td>
                    <td>{col.n_unique}</td>
                    <td>{summary}</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
"""
        
        # Warnings
        if profile.warnings:
            html += "<h2>⚠️ Warnings</h2>"
            for warning in profile.warnings:
                html += f'<div class="warning">{warning}</div>'
        
        html += """
    </div>
</body>
</html>
"""
        
        with open(output_path, "w") as f:
            f.write(html)


def profile_data(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    event_col: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    print_summary: bool = True,
) -> DataProfile:
    """
    Convenience function to profile a DataFrame.
    
    Args:
        df: DataFrame to profile
        time_col: Time-to-event column name
        event_col: Event indicator column name
        output_path: Optional path to save report
        print_summary: Whether to print summary to console
        
    Returns:
        DataProfile object
    """
    profiler = DataProfiler(df, time_col=time_col, event_col=event_col)
    profile = profiler.generate_profile()
    
    if print_summary:
        profiler.print_summary()
    
    if output_path:
        profiler.save_report(output_path)
    
    return profile


