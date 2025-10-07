"""Configuration validation using JSON Schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:
    jsonschema = None

from clinical_survival.utils import load_yaml

# JSON Schemas for configuration files
PARAMS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "seed": {"type": "integer", "minimum": 0},
        "n_splits": {"type": "integer", "minimum": 2, "maximum": 10},
        "inner_splits": {"type": "integer", "minimum": 2, "maximum": 5},
        "test_split": {"type": "number", "minimum": 0.1, "maximum": 0.5},
        "time_col": {"type": "string"},
        "event_col": {"type": "string"},
        "id_col": {"type": "string"},
        "scoring": {
            "type": "object",
            "properties": {
                "primary": {"type": "string"},
                "secondary": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["primary"],
        },
        "calibration": {
            "type": "object",
            "properties": {
                "times_days": {"type": "array", "items": {"type": "number", "minimum": 1}},
                "bins": {"type": "integer", "minimum": 5, "maximum": 20},
            },
        },
        "decision_curve": {
            "type": "object",
            "properties": {
                "times_days": {"type": "array", "items": {"type": "number", "minimum": 1}},
                "thresholds": {
                    "type": "array",
                    "items": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
        "missing": {
            "type": "object",
            "properties": {
                "strategy": {"type": "string", "enum": ["iterative", "simple"]},
                "max_iter": {"type": "integer", "minimum": 1, "maximum": 50},
                "initial_strategy": {
                    "type": "string",
                    "enum": ["mean", "median", "most_frequent", "constant"],
                },
            },
        },
        "models": {
            "type": "array",
            "items": {"type": "string", "enum": ["coxph", "rsf", "xgb_cox", "xgb_aft"]},
            "minItems": 1,
            "uniqueItems": True,
        },
        "paths": {
            "type": "object",
            "properties": {
                "data_csv": {"type": "string"},
                "metadata": {"type": "string"},
                "external_csv": {"type": ["string", "null"]},
                "outdir": {"type": "string"},
                "features": {"type": "string"},
            },
            "required": ["data_csv", "metadata", "outdir"],
        },
        "evaluation": {
            "type": "object",
            "properties": {"bootstrap": {"type": "integer", "minimum": 10, "maximum": 1000}},
        },
        "external": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "group_column": {"type": "string"},
                "train_value": {"type": "string"},
                "external_value": {"type": "string"},
            },
        },
        "explain": {
            "type": "object",
            "properties": {
                "shap_samples": {"type": "integer", "minimum": 10, "maximum": 1000},
                "pdp_features": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "required": ["seed", "models", "paths"],
}

MODEL_GRID_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "patternProperties": {
        "^(coxph|rsf|xgb_cox|xgb_aft)$": {
            "type": "object",
            "additionalProperties": {
                "oneOf": [
                    {"type": "array"},
                    {"type": "number"},
                    {"type": "string"},
                    {"type": "boolean"},
                ]
            },
        }
    },
    "additionalProperties": False,
}

FEATURES_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "numeric": {"type": "array", "items": {"type": "string"}},
        "categorical": {"type": "array", "items": {"type": "string"}},
        "drop": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}


def validate_config_file(config_path: str | Path, schema: dict[str, Any]) -> list[str]:
    """Validate a configuration file against its schema.

    Args:
        config_path: Path to the configuration file
        schema: JSON schema to validate against

    Returns:
        List of validation error messages (empty if valid)
    """
    if jsonschema is None:
        # If jsonschema is not available, do basic validation
        return _basic_validation(config_path, schema)

    config_path = Path(config_path)

    if not config_path.exists():
        return [f"Configuration file not found: {config_path}"]

    try:
        config_data = load_yaml(config_path)
    except Exception as e:
        return [f"Failed to parse YAML file {config_path}: {e}"]

    try:
        jsonschema.validate(config_data, schema)
        return []
    except jsonschema.ValidationError as e:
        return [_format_validation_error(e)]


def _basic_validation(config_path: str | Path, schema: dict[str, Any]) -> list[str]:
    """Basic validation when jsonschema is not available."""
    config_path = Path(config_path)
    errors = []

    if not config_path.exists():
        errors.append(f"Configuration file not found: {config_path}")
        return errors

    try:
        config_data = load_yaml(config_path)
    except Exception as e:
        errors.append(f"Failed to parse YAML file {config_path}: {e}")
        return errors

    # Basic structure checks
    if not isinstance(config_data, dict):
        errors.append(f"Configuration file must contain a dictionary, got {type(config_data)}")
        return errors

    return []


def _format_validation_error(error: jsonschema.ValidationError) -> str:
    """Format a validation error into a user-friendly message."""
    path = " -> ".join(str(p) for p in error.absolute_path) if error.absolute_path else "root"
    return f"Validation error at '{path}': {error.message}"


def validate_params_config(config_path: str | Path) -> list[str]:
    """Validate params.yaml configuration file."""
    return validate_config_file(config_path, PARAMS_SCHEMA)


def validate_model_grid_config(config_path: str | Path) -> list[str]:
    """Validate model_grid.yaml configuration file."""
    return validate_config_file(config_path, MODEL_GRID_SCHEMA)


def validate_features_config(config_path: str | Path) -> list[str]:
    """Validate features.yaml configuration file."""
    return validate_config_file(config_path, FEATURES_SCHEMA)


def validate_all_configs(
    params_path: str | Path, model_grid_path: str | Path, features_path: str | Path
) -> dict[str, list[str]]:
    """Validate all configuration files.

    Returns:
        Dictionary mapping file names to lists of error messages
    """
    return {
        "params.yaml": validate_params_config(params_path),
        "model_grid.yaml": validate_model_grid_config(model_grid_path),
        "features.yaml": validate_features_config(features_path),
    }


def print_validation_errors(errors: dict[str, list[str]]) -> None:
    """Print validation errors in a user-friendly format."""
    has_errors = False

    for filename, file_errors in errors.items():
        if file_errors:
            has_errors = True
            print(f"\n❌ {filename}:")
            for error in file_errors:
                print(f"  • {error}")

    if not has_errors:
        print("✅ All configuration files are valid!")
    else:
        print(f"\n❌ Found {sum(len(errors) for errors in errors.values())} validation error(s)")
        print("\n💡 Tip: Use 'clinical-ml validate-config --help' for more information")
