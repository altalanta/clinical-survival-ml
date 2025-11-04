from __future__ import annotations


class ClinicalSurvivalError(Exception):
    """Base exception for all custom errors in clinical_survival_ml."""


class ConfigurationError(ClinicalSurvivalError):
    """Exception raised for issues in configuration files."""


class DataError(ClinicalSurvivalError):
    """Exception raised for issues related to data loading or processing."""


class ModelError(ClinicalSurvivalError):
    """Exception raised for issues during model training or evaluation."""


class ReportError(ClinicalSurvivalError):
    """Exception raised for issues during report generation."""






