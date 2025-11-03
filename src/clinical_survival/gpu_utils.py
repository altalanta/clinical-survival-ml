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

try:
    import cuml
    import cudf
    from cuml.ensemble import RandomForestClassifier as cuMLRandomForest
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

try:
    import dask
    import dask_cudf
    DASK_CUDA_AVAILABLE = True
except ImportError:
    DASK_CUDA_AVAILABLE = False


class MemoryManager:
    """Intelligent memory management for large datasets."""

    def __init__(self, max_memory_gb: float | None = None, chunk_size_mb: int = 100):
        """Initialize memory manager.

        Args:
            max_memory_gb: Maximum memory to use in GB (None for auto-detection)
            chunk_size_mb: Size of data chunks in MB for processing
        """
        self.max_memory_gb = max_memory_gb
        self.chunk_size_mb = chunk_size_mb

        if max_memory_gb is None:
            # Auto-detect available memory
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.max_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 * 0.8  # Use 80% of GPU memory
            else:
                import psutil
                self.max_memory_gb = psutil.virtual_memory().available / 1e9 * 0.8  # Use 80% of system memory

    def estimate_dataset_memory(self, n_samples: int, n_features: int, dtype_size: int = 8) -> float:
        """Estimate memory usage for a dataset in GB.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            dtype_size: Size of each element in bytes (8 for float64)

        Returns:
            Estimated memory usage in GB
        """
        # Rough estimate: samples * features * dtype_size + overhead
        estimated_bytes = n_samples * n_features * dtype_size * 1.5  # 1.5x overhead factor
        return estimated_bytes / 1e9

    def should_partition(self, n_samples: int, n_features: int) -> bool:
        """Check if dataset should be partitioned for memory efficiency.

        Args:
            n_samples: Number of samples
            n_features: Number of features

        Returns:
            Whether to use partitioning
        """
        estimated_memory = self.estimate_dataset_memory(n_samples, n_features)
        return estimated_memory > self.max_memory_gb

    def get_optimal_chunk_size(self, n_samples: int, n_features: int) -> int:
        """Get optimal chunk size for processing.

        Args:
            n_samples: Number of samples
            n_features: Number of features

        Returns:
            Optimal number of samples per chunk
        """
        estimated_memory = self.estimate_dataset_memory(n_samples, n_features)
        max_samples_per_chunk = int((self.max_memory_gb * 1e9) / (n_features * 8 * 1.5))

        # Ensure minimum chunk size
        return max(1000, min(max_samples_per_chunk, n_samples // 4))

    def partition_dataframe(self, df: pd.DataFrame, chunk_size: int | None = None) -> list[pd.DataFrame]:
        """Partition dataframe into chunks for memory-efficient processing.

        Args:
            df: DataFrame to partition
            chunk_size: Number of rows per chunk (None for auto-calculation)

        Returns:
            List of DataFrame chunks
        """
        if chunk_size is None:
            chunk_size = self.get_optimal_chunk_size(df.shape[0], df.shape[1])

        chunks = []
        for i in range(0, len(df), chunk_size):
            chunks.append(df.iloc[i:i + chunk_size].copy())

        return chunks


class GPUAccelerator:
    """GPU acceleration utilities for survival models."""

    def __init__(self, use_gpu: bool = True, gpu_id: int = 0, n_jobs: int | None = None, max_memory_gb: float | None = None):
        """Initialize GPU accelerator.

        Args:
            use_gpu: Whether to use GPU acceleration when available
            gpu_id: GPU device ID to use
            n_jobs: Number of parallel jobs for CPU processing (-1 for all cores)
        """
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.n_jobs = n_jobs if n_jobs is not None else mp.cpu_count()
        self.max_memory_gb = max_memory_gb

        # Initialize memory manager
        self.memory_manager = MemoryManager(max_memory_gb=max_memory_gb)

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

        # cuML GPU support (for Random Survival Forest and ensembles)
        self.cuml_available = CUML_AVAILABLE and self.cuda_available

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

    def profile_memory_usage(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> dict[str, Any]:
        """Profile memory usage for a dataset and model training.

        Args:
            X: Feature matrix
            y: Target vector (optional)

        Returns:
            Dictionary with memory profiling information
        """
        if isinstance(X, np.ndarray):
            n_samples, n_features = X.shape
        else:
            n_samples, n_features = X.shape

        profile = {
            "dataset_samples": n_samples,
            "dataset_features": n_features,
            "estimated_memory_gb": self.memory_manager.estimate_dataset_memory(n_samples, n_features),
            "should_partition": self.memory_manager.should_partition(n_samples, n_features),
            "optimal_chunk_size": self.memory_manager.get_optimal_chunk_size(n_samples, n_features),
            "max_memory_gb": self.memory_manager.max_memory_gb,
        }

        if y is not None:
            profile["target_memory_mb"] = y.nbytes / 1e6 if hasattr(y, 'nbytes') else "unknown"

        return profile

    def optimize_for_memory(self, X: pd.DataFrame | np.ndarray, model_type: str = "auto") -> dict[str, Any]:
        """Get memory optimization recommendations for a dataset.

        Args:
            X: Feature matrix
            model_type: Type of model ('auto', 'xgb', 'rsf', 'ensemble')

        Returns:
            Dictionary with optimization recommendations
        """
        profile = self.profile_memory_usage(X)

        recommendations = {
            "use_partitioning": profile["should_partition"],
            "chunk_size": profile["optimal_chunk_size"],
            "batch_size": min(profile["optimal_chunk_size"], 10000),
            "memory_warnings": [],
        }

        if profile["estimated_memory_gb"] > self.memory_manager.max_memory_gb * 0.9:
            recommendations["memory_warnings"].append(
                f"Dataset may exceed available memory ({profile['estimated_memory_gb']".2f"}GB vs {self.memory_manager.max_memory_gb".2f"}GB)"
            )

        # Model-specific recommendations
        if model_type == "xgb":
            recommendations.update({
                "xgb_tree_method": "gpu_hist" if self.xgb_gpu_available else "hist",
                "xgb_max_bin": 64 if profile["estimated_memory_gb"] > 10 else 256,
            })
        elif model_type == "rsf":
            recommendations.update({
                "rsf_use_cuml": self.cuml_available,
                "rsf_max_samples": min(profile["optimal_chunk_size"], 100000),
            })

        return recommendations


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
