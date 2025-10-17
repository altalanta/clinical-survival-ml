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
from clinical_survival.incremental_learning import (
    IncrementalLearner,
    IncrementalLearningManager,
    IncrementalUpdateConfig,
    create_incremental_learner,
    load_incremental_learning_config,
)
from clinical_survival.distributed import (
    DistributedBenchmarker,
    DistributedClient,
    DistributedConfig,
    DistributedDataset,
    DistributedEvaluator,
    DistributedMetrics,
    DistributedTrainer,
    create_distributed_config,
    load_distributed_config,
)

__all__ = [
    "__version__",
    "AutoSurvivalML",
    "create_automl_study",
    "CounterfactualExplainer",
    "create_counterfactual_explainer",
    "CausalInference",
    "GPUAccelerator",
    "create_gpu_accelerator",
    "IncrementalLearner",
    "IncrementalLearningManager",
    "IncrementalUpdateConfig",
    "create_incremental_learner",
    "load_incremental_learning_config",
    "DistributedClient",
    "DistributedConfig",
    "DistributedDataset",
    "DistributedTrainer",
    "DistributedEvaluator",
    "DistributedBenchmarker",
    "DistributedMetrics",
    "create_distributed_config",
    "load_distributed_config",
]
