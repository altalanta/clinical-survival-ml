"""Clinical Survival ML package."""

from __future__ import annotations

try:
    from ._version import version as __version__
except ImportError:
    # Fallback for development installations
    __version__ = "0.1.0"

from clinical_survival.automl import AutoSurvivalML, create_automl_study
from clinical_survival.counterfactual import (
    CausalInference,
    CounterfactualExplainer,
    create_counterfactual_explainer,
)
from clinical_survival.gpu_utils import GPUAccelerator, create_gpu_accelerator

__all__ = [
    "__version__",
    "AutoSurvivalML",
    "create_automl_study",
    "CounterfactualExplainer",
    "create_counterfactual_explainer",
    "CausalInference",
    "GPUAccelerator",
    "create_gpu_accelerator",
]
