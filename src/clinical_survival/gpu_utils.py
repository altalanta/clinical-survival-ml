"""GPU acceleration and parallel processing utilities."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


class GPUAccelerator:
    """GPU acceleration utilities for survival models."""

    def __init__(self, use_gpu: bool = True, gpu_id: int = 0, n_jobs: int | None = None):
        """Initialize GPU accelerator.

        Args:
            use_gpu: Whether to use GPU acceleration when available
            gpu_id: GPU device ID to use
            n_jobs: Number of parallel jobs for CPU processing (-1 for all cores)
        """
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()

        # Detect available hardware
        self._detect_hardware()

    def _detect_hardware(self) -> None:
        """Detect available GPU and CPU resources."""
        self.gpu_available = False
        self.cuda_available = False

        if TORCH_AVAILABLE:
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                self.gpu_count = torch.cuda.device_count()
                self.gpu_name = torch.cuda.get_device_name(self.gpu_id) if self.gpu_id < self.gpu_count else None
            else:
                self.gpu_count = 0
                self.gpu_name = None

        # XGBoost GPU support (via CUDA)
        self.xgb_gpu_available = XGB_AVAILABLE and self._check_xgb_gpu()

    def _check_xgb_gpu(self) -> bool:
        """Check if XGBoost has GPU support compiled in."""
        if not XGB_AVAILABLE:
            return False

        try:
            # Try to create a simple XGBoost model with GPU
            temp_data = np.random.random((10, 5))
            temp_target = np.random.randint(0, 2, 10)

            dtrain = xgb.DMatrix(temp_data, label=temp_target)

            params = {
                "tree_method": "gpu_hist",
                "gpu_id": self.gpu_id,
                "max_depth": 3,
                "n_estimators": 2
            }

            model = xgb.train(params, dtrain, num_boost_round=1)
            return True
        except Exception:
            return False

    def get_gpu_params(self, model_type: str = "xgb") -> dict[str, Any]:
        """Get GPU-specific parameters for different model types.

        Args:
            model_type: Type of model ('xgb' for XGBoost)

        Returns:
            Dictionary of GPU parameters
        """
        if not self.use_gpu:
            return {}

        if model_type == "xgb":
            if self.xgb_gpu_available:
                return {
                    "tree_method": "gpu_hist",
                    "gpu_id": self.gpu_id,
                }
            else:
                # Fallback to CPU with parallel processing
                return {
                    "nthread": self.n_jobs,
                }

        return {}

    def parallel_fit(
        self,
        models_and_data: list[tuple[Callable, pd.DataFrame, np.ndarray]],
        n_jobs: int | None = None,
    ) -> list[Any]:
        """Fit multiple models in parallel.

        Args:
            models_and_data: List of (model_constructor, X, y) tuples
            n_jobs: Number of parallel jobs (None uses instance default)

        Returns:
            List of fitted models
        """
        if n_jobs is None:
            n_jobs = self.n_jobs

        def fit_model(model_constructor, X, y):
            model = model_constructor()
            return model.fit(X, y)

        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(fit_model)(model_constructor, X, y)
            for model_constructor, X, y in models_and_data
        )

        return results

    def parallel_cross_validate(
        self,
        model_constructor: Callable,
        X: pd.DataFrame,
        y: np.ndarray,
        cv: int = 5,
        scoring: str | Callable = "neg_mean_squared_error",
        n_jobs: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Perform cross-validation in parallel.

        Args:
            model_constructor: Function that creates a model instance
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary with 'scores' key containing CV scores
        """
        if n_jobs is None:
            n_jobs = self.n_jobs

        scores = cross_val_score(
            model_constructor(),
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,  # Let joblib handle parallelism
        )

        return {"scores": scores}

    def get_optimal_config(self) -> dict[str, Any]:
        """Get optimal configuration based on available hardware.

        Returns:
            Dictionary with recommended settings
        """
        config = {
            "use_gpu": self.use_gpu,
            "gpu_id": self.gpu_id,
            "n_jobs": self.n_jobs,
            "gpu_available": self.gpu_available,
            "cuda_available": self.cuda_available,
            "xgb_gpu_available": self.xgb_gpu_available,
        }

        if self.cuda_available:
            config.update({
                "recommended_gpu_id": self.gpu_id if self.gpu_id < self.gpu_count else 0,
                "gpu_memory_gb": torch.cuda.get_device_properties(self.gpu_id).total_memory / 1e9 if self.gpu_id < self.gpu_count else 0,
            })

        return config

    def benchmark_hardware(self, model_constructor: Callable, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
        """Benchmark model training on available hardware.

        Args:
            model_constructor: Function that creates a model instance
            X: Feature matrix for benchmarking
            y: Target vector for benchmarking

        Returns:
            Dictionary with timing results
        """
        import time

        results = {}

        # CPU benchmark
        start_time = time.time()
        model_cpu = model_constructor()
        model_cpu.fit(X, y)
        cpu_time = time.time() - start_time
        results["cpu_time"] = cpu_time

        # GPU benchmark (if available)
        if self.gpu_available and self.xgb_gpu_available:
            try:
                start_time = time.time()
                model_gpu = model_constructor()
                # Modify model to use GPU
                if hasattr(model_gpu, 'params'):
                    model_gpu.params.update(self.get_gpu_params("xgb"))
                elif hasattr(model_gpu, 'model') and hasattr(model_gpu.model, 'params'):
                    model_gpu.model.params.update(self.get_gpu_params("xgb"))

                model_gpu.fit(X, y)
                gpu_time = time.time() - start_time
                results["gpu_time"] = gpu_time
                results["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 1.0
            except Exception as e:
                results["gpu_error"] = str(e)

        return results


def create_gpu_accelerator(
    use_gpu: bool | None = None,
    gpu_id: int = 0,
    n_jobs: int | None = None,
) -> GPUAccelerator:
    """Create GPU accelerator with auto-detection.

    Args:
        use_gpu: Whether to use GPU (None for auto-detection)
        gpu_id: GPU device ID
        n_jobs: Number of CPU jobs

    Returns:
        Configured GPUAccelerator instance
    """
    if use_gpu is None:
        # Auto-detect GPU availability
        use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()

    return GPUAccelerator(
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        n_jobs=n_jobs,
    )

