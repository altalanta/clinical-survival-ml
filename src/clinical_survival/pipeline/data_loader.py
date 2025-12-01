"""
Data loading pipeline step with schema validation and error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from clinical_survival.config import ParamsConfig
from clinical_survival.error_handling import wrap_step_errors
from clinical_survival.errors import DataLoadError, MissingColumnError
from clinical_survival.logging_config import get_logger
from clinical_survival.pipeline.schemas import (
    DataLoaderInput,
    DataLoaderOutput,
    validate_pipeline_step,
)

# Get module logger
logger = get_logger(__name__)


def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file with error handling.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        DataLoadError: If the file cannot be loaded
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        raise DataLoadError(
            f"Data file not found: {path}",
            path=str(path),
        )
    except pd.errors.EmptyDataError:
        raise DataLoadError(
            f"Data file is empty: {path}",
            path=str(path),
        )
    except pd.errors.ParserError as e:
        raise DataLoadError(
            f"Failed to parse CSV file: {e}",
            path=str(path),
        )
    except PermissionError:
        raise DataLoadError(
            f"Permission denied reading file: {path}",
            path=str(path),
        )
    except Exception as e:
        raise DataLoadError(
            f"Unexpected error loading data: {e}",
            path=str(path),
        )


def _validate_required_columns(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
    path: Path,
) -> None:
    """
    Validate that required columns exist in the DataFrame.
    
    Args:
        df: DataFrame to validate
        time_col: Name of the time column
        event_col: Name of the event column
        path: Path to the data file (for error context)
        
    Raises:
        MissingColumnError: If required columns are missing
    """
    required = {time_col, event_col}
    available = set(df.columns)
    missing = required - available
    
    if missing:
        raise MissingColumnError(
            f"Required columns missing from data file: {missing}",
            missing_columns=list(missing),
            available_columns=list(df.columns),
            path=str(path),
        )


@validate_pipeline_step(
    input_schema=DataLoaderInput,
    output_schema=DataLoaderOutput,
)
@wrap_step_errors("data_loading")
def load_raw_data(params_config: ParamsConfig, **_kwargs: Any) -> Dict[str, Any]:
    """
    Pipeline step to load the raw data from the specified path.
    
    This step:
    1. Loads the CSV file from the configured path
    2. Validates that required columns (time, event) exist
    3. Logs data quality summary information
    
    Input Schema (DataLoaderInput):
        - params_config: ParamsConfig - Main parameters configuration
        
    Output Schema (DataLoaderOutput):
        - raw_df: pd.DataFrame - Loaded raw data
    
    Args:
        params_config: Main parameters configuration
        **_kwargs: Additional context (unused)
        
    Returns:
        Dictionary with 'raw_df' key containing the loaded DataFrame
        
    Raises:
        DataLoadError: If the file cannot be loaded
        MissingColumnError: If required columns are missing
        SchemaValidationError: If input/output validation fails
    """
    data_path = Path(params_config.paths.data_csv)
    
    logger.info(
        "Loading raw data",
        extra={"path": str(data_path)},
    )
    
    # Load the data
    df = _load_csv(data_path)
    
    # Validate required columns
    _validate_required_columns(
        df,
        time_col=params_config.time_col,
        event_col=params_config.event_col,
        path=data_path,
    )
    
    # Log success with details
    logger.info(
        "Raw data loaded successfully",
        extra={
            "shape": df.shape,
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        },
    )
    
    # Log data quality summary
    missing_pct = (df.isnull().sum().sum() / df.size) * 100 if df.size > 0 else 0
    event_rate = df[params_config.event_col].mean() if params_config.event_col in df.columns else None
    
    logger.debug(
        "Data quality summary",
        extra={
            "missing_percentage": round(missing_pct, 2),
            "event_rate": round(event_rate, 3) if event_rate is not None else None,
            "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        },
    )
    
    return {"raw_df": df}
