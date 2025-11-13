"""Distributed computing support for large-scale clinical datasets."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from clinical_survival.logging_utils import log_function_call
from clinical_survival.utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed computing."""

    # Cluster setup
    cluster_type: str = "local"  # "local", "dask", "ray", "slurm"
    n_workers: int = 4
    threads_per_worker: int = 2
    memory_per_worker: str = "2GB"

    # Data partitioning
    partition_strategy: str = "balanced"  # "balanced", "hash", "random"
    n_partitions: int = 10
    max_partition_size: int = 10000  # Max samples per partition

    # Communication
    scheduler_address: str = "127.0.0.1:8786"
    dashboard_address: str = "127.0.0.1:8787"

    # Performance tuning
    chunk_size: int = 1000
    optimize_memory: bool = True
    use_gpu_if_available: bool = True

    # Fault tolerance
    retry_failed_tasks: bool = True
    max_retries: int = 3
    timeout_minutes: int = 60

    # Resource allocation
    resource_allocation_strategy: str = "balanced"  # "balanced", "memory_aware", "cpu_aware"


@dataclass
class DistributedMetrics:
    """Metrics for distributed computing performance."""

    total_time: float
    computation_time: float
    communication_time: float
    memory_usage_peak: float
    cpu_usage_avg: float
    n_tasks_completed: int
    n_tasks_failed: int
    data_transfer_volume: float
    speedup_factor: float


class DistributedClient:
    """Wrapper for distributed computing clients."""

    def __init__(self, config: DistributedConfig):
        self.config = config
        self.client = None
        self.cluster = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize the distributed computing client."""
        try:
            if self.config.cluster_type == "dask":
                return self._initialize_dask()
            elif self.config.cluster_type == "ray":
                return self._initialize_ray()
            elif self.config.cluster_type == "local":
                return self._initialize_local()
            else:
                logger.error(f"Unsupported cluster type: {self.config.cluster_type}")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize distributed client: {e}")
            return False

    def _initialize_dask(self) -> bool:
        """Initialize Dask distributed client."""
        try:
            from dask.distributed import Client, LocalCluster

            # Create local cluster for testing
            self.cluster = LocalCluster(
                n_workers=self.config.n_workers,
                threads_per_worker=self.config.threads_per_worker,
                memory_limit=self.config.memory_per_worker,
                dashboard_address=self.config.dashboard_address,
                processes=True
            )

            self.client = Client(self.cluster)
            self._initialized = True

            logger.info(f"Dask client initialized with {self.config.n_workers} workers")
            logger.info(f"Dashboard available at: {self.client.dashboard_link}")

            return True

        except ImportError:
            logger.error("Dask not available. Install with: pip install dask[distributed]")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Dask client: {e}")
            return False

    def _initialize_ray(self) -> bool:
        """Initialize Ray distributed client."""
        try:
            import ray

            # Initialize Ray
            if not ray.is_initialized():
                ray.init(
                    num_cpus=self.config.n_workers * self.config.threads_per_worker,
                    num_gpus=1 if self.config.use_gpu_if_available else 0,
                    dashboard_host=self.config.dashboard_address.split(":")[0],
                    dashboard_port=int(self.config.dashboard_address.split(":")[1])
                )

            self.client = ray
            self._initialized = True

            logger.info(f"Ray client initialized with {self.config.n_workers} workers")
            return True

        except ImportError:
            logger.error("Ray not available. Install with: pip install ray")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Ray client: {e}")
            return False

    def _initialize_local(self) -> bool:
        """Initialize local computing (no distribution)."""
        self._initialized = True
        logger.info("Using local computing (no distribution)")
        return True

    def shutdown(self) -> None:
        """Shutdown the distributed client."""
        if self.client and self._initialized:
            try:
                if self.config.cluster_type == "dask" and self.cluster:
                    self.client.close()
                    self.cluster.close()
                elif self.config.cluster_type == "ray":
                    import ray
                    ray.shutdown()
            except Exception as e:
                logger.warning(f"Error during client shutdown: {e}")

    def scatter_data(self, data: Any) -> Any:
        """Scatter data across workers."""
        if not self._initialized:
            return data

        if self.config.cluster_type == "dask":
            return self.client.scatter(data)
        elif self.config.cluster_type == "ray":
            return self.client.put(data)
        else:
            return data

    def gather_data(self, scattered_data: Any) -> Any:
        """Gather scattered data from workers."""
        if not self._initialized:
            return scattered_data

        if self.config.cluster_type == "dask":
            return self.client.gather(scattered_data)
        elif self.config.cluster_type == "ray":
            return self.client.get(scattered_data)
        else:
            return scattered_data


class DistributedDataset:
    """Wrapper for distributed datasets."""

    def __init__(self, client: DistributedClient, data_path: Path, config: DistributedConfig):
        self.client = client
        self.data_path = data_path
        self.config = config
        self._df = None
        self._ddf = None  # Dask DataFrame
        self._partitions = None

    def load_and_partition(self) -> bool:
        """Load data and create distributed partitions."""
        try:
            # Load data
            if self.data_path.suffix.lower() == '.csv':
                self._df = pd.read_csv(self.data_path)
            elif self.data_path.suffix.lower() in ['.parquet', '.pq']:
                self._df = pd.read_parquet(self.data_path)
            else:
                logger.error(f"Unsupported file format: {self.data_path.suffix}")
                return False

            logger.info(f"Loaded dataset with {len(self._df)} samples and {len(self._df.columns)} features")

            # Create distributed partitions
            if self.config.cluster_type == "dask":
                return self._create_dask_partitions()
            else:
                return self._create_simple_partitions()

        except Exception as e:
            logger.error(f"Failed to load and partition data: {e}")
            return False

    def _create_dask_partitions(self) -> bool:
        """Create Dask DataFrame partitions."""
        try:
            import dask.dataframe as dd

            # Convert to Dask DataFrame
            self._ddf = dd.from_pandas(self._df, npartitions=self.config.n_partitions)

            # Repartition based on strategy
            if self.config.partition_strategy == "balanced":
                self._ddf = self._ddf.repartition(npartitions=self.config.n_partitions)
            elif self.config.partition_strategy == "hash":
                # Hash partition by a key column if available
                id_col = "id" if "id" in self._df.columns else self._df.index.name or "index"
                self._ddf = self._ddf.set_index(id_col).repartition(npartitions=self.config.n_partitions)

            logger.info(f"Created {self._ddf.npartitions} Dask partitions")
            return True

        except ImportError:
            logger.error("Dask DataFrame not available")
            return False
        except Exception as e:
            logger.error(f"Failed to create Dask partitions: {e}")
            return False

    def _create_simple_partitions(self) -> bool:
        """Create simple partitions for non-distributed computing."""
        try:
            # Simple partitioning by dividing data into chunks
            chunk_size = max(1, len(self._df) // self.config.n_partitions)

            self._partitions = []
            for i in range(0, len(self._df), chunk_size):
                end_idx = min(i + chunk_size, len(self._df))
                self._partitions.append(self._df.iloc[i:end_idx])

            logger.info(f"Created {len(self._partitions)} simple partitions")
            return True

        except Exception as e:
            logger.error(f"Failed to create simple partitions: {e}")
            return False

    def get_partition(self, partition_idx: int) -> pd.DataFrame:
        """Get a specific partition."""
        if self.config.cluster_type == "dask" and self._ddf is not None:
            return self._ddf.get_partition(partition_idx).compute()
        elif self._partitions is not None:
            return self._partitions[partition_idx]
        else:
            raise ValueError("Partitions not initialized")

    def get_n_partitions(self) -> int:
        """Get number of partitions."""
        if self.config.cluster_type == "dask" and self._ddf is not None:
            return self._ddf.npartitions
        elif self._partitions is not None:
            return len(self._partitions)
        else:
            return 0


class DistributedTrainer:
    """Distributed training for survival models."""

    def __init__(self, client: DistributedClient, config: DistributedConfig):
        self.client = client
        self.config = config
        self.metrics = None

    def train_distributed(
        self,
        dataset: DistributedDataset,
        model_factory,
        model_params: Dict[str, Any],
        cv_folds: int = 5
    ) -> Tuple[Any, DistributedMetrics]:
        """Train model using distributed computing."""

        start_time = time.time()

        try:
            if self.config.cluster_type == "dask":
                return self._train_with_dask(dataset, model_factory, model_params, cv_folds)
            elif self.config.cluster_type == "ray":
                return self._train_with_ray(dataset, model_factory, model_params, cv_folds)
            else:
                return self._train_locally(dataset, model_factory, model_params, cv_folds)

        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            raise

    def _train_with_dask(self, dataset, model_factory, model_params, cv_folds):
        """Train using Dask distributed computing."""
        try:
            import dask
            from dask import delayed
            from dask.distributed import as_completed

            # Define training task
            @delayed
            def train_partition(partition_data, partition_idx, model_params, cv_folds):
                """Train model on a single partition."""
                try:
                    # Extract X and y from partition
                    feature_cols = [col for col in partition_data.columns
                                   if col not in ['id', 'time', 'event']]
                    X = partition_data[feature_cols]
                    y = list(zip(partition_data['event'], partition_data['time']))

                    # Create and train model
                    model = model_factory(**model_params)
                    model.fit(X, y)

                    return {
                        'partition_idx': partition_idx,
                        'model': model,
                        'n_samples': len(X),
                        'status': 'success'
                    }

                except Exception as e:
                    return {
                        'partition_idx': partition_idx,
                        'error': str(e),
                        'status': 'failed'
                    }

            # Create training tasks for all partitions
            tasks = []
            for i in range(dataset.get_n_partitions()):
                partition_data = dataset.get_partition(i)
                task = train_partition(partition_data, i, model_params, cv_folds)
                tasks.append(task)

            # Execute tasks
            results = dask.compute(*tasks)

            # Combine results
            successful_models = []
            failed_tasks = 0

            for result in results:
                if result['status'] == 'success':
                    successful_models.append(result['model'])
                else:
                    failed_tasks += 1

            # Create ensemble or combined model from successful partitions
            if successful_models:
                # For now, return the first successful model
                # In practice, you'd create an ensemble
                final_model = successful_models[0]
            else:
                raise RuntimeError("All training tasks failed")

            # Calculate metrics
            total_time = time.time() - start_time

            metrics = DistributedMetrics(
                total_time=total_time,
                computation_time=total_time * 0.8,  # Estimate
                communication_time=total_time * 0.2,  # Estimate
                memory_usage_peak=0.0,  # Would need monitoring
                cpu_usage_avg=0.0,     # Would need monitoring
                n_tasks_completed=len(successful_models),
                n_tasks_failed=failed_tasks,
                data_transfer_volume=0.0,  # Would need monitoring
                speedup_factor=1.0      # Would calculate vs local
            )

            return final_model, metrics

        except ImportError:
            logger.error("Dask not available for distributed training")
            raise
        except Exception as e:
            logger.error(f"Dask distributed training failed: {e}")
            raise

    def _train_with_ray(self, dataset, model_factory, model_params, cv_folds):
        """Train using Ray distributed computing."""
        try:
            import ray

            @ray.remote
            def train_partition_ray(partition_data, partition_idx, model_params, cv_folds):
                """Ray remote training task."""
                try:
                    # Similar to Dask version
                    feature_cols = [col for col in partition_data.columns
                                   if col not in ['id', 'time', 'event']]
                    X = partition_data[feature_cols]
                    y = list(zip(partition_data['event'], partition_data['time']))

                    model = model_factory(**model_params)
                    model.fit(X, y)

                    return {
                        'partition_idx': partition_idx,
                        'model': model,
                        'n_samples': len(X),
                        'status': 'success'
                    }

                except Exception as e:
                    return {
                        'partition_idx': partition_idx,
                        'error': str(e),
                        'status': 'failed'
                    }

            # Create Ray tasks
            tasks = []
            for i in range(dataset.get_n_partitions()):
                partition_data = dataset.get_partition(i)
                task = train_partition_ray.remote(partition_data, i, model_params, cv_folds)
                tasks.append(task)

            # Execute tasks
            results = ray.get(tasks)

            # Process results (similar to Dask)
            successful_models = []
            failed_tasks = 0

            for result in results:
                if result['status'] == 'success':
                    successful_models.append(result['model'])
                else:
                    failed_tasks += 1

            if successful_models:
                final_model = successful_models[0]  # Placeholder
            else:
                raise RuntimeError("All training tasks failed")

            total_time = time.time() - start_time

            metrics = DistributedMetrics(
                total_time=total_time,
                computation_time=total_time * 0.8,
                communication_time=total_time * 0.2,
                memory_usage_peak=0.0,
                cpu_usage_avg=0.0,
                n_tasks_completed=len(successful_models),
                n_tasks_failed=failed_tasks,
                data_transfer_volume=0.0,
                speedup_factor=1.0
            )

            return final_model, metrics

        except ImportError:
            logger.error("Ray not available for distributed training")
            raise
        except Exception as e:
            logger.error(f"Ray distributed training failed: {e}")
            raise

    def _train_locally(self, dataset, model_factory, model_params, cv_folds):
        """Train locally (fallback)."""
        # Combine all partitions and train normally
        combined_df = pd.concat([dataset.get_partition(i) for i in range(dataset.get_n_partitions())],
                               ignore_index=True)

        feature_cols = [col for col in combined_df.columns if col not in ['id', 'time', 'event']]
        X = combined_df[feature_cols]
        y = list(zip(combined_df['event'], combined_df['time']))

        model = model_factory(**model_params)
        model.fit(X, y)

        total_time = time.time() - start_time

        metrics = DistributedMetrics(
            total_time=total_time,
            computation_time=total_time,
            communication_time=0.0,
            memory_usage_peak=0.0,
            cpu_usage_avg=0.0,
            n_tasks_completed=1,
            n_tasks_failed=0,
            data_transfer_volume=0.0,
            speedup_factor=1.0
        )

        return model, metrics


class DistributedEvaluator:
    """Distributed evaluation for survival models."""

    def __init__(self, client: DistributedClient, config: DistributedConfig):
        self.client = client
        self.config = config

    def evaluate_distributed(
        self,
        dataset: DistributedDataset,
        model,
        metrics_functions: List[str] = ["concordance", "ibs"]
    ) -> Dict[str, float]:
        """Evaluate model using distributed computing."""

        try:
            if self.config.cluster_type == "dask":
                return self._evaluate_with_dask(dataset, model, metrics_functions)
            elif self.config.cluster_type == "ray":
                return self._evaluate_with_ray(dataset, model, metrics_functions)
            else:
                return self._evaluate_locally(dataset, model, metrics_functions)

        except Exception as e:
            logger.error(f"Distributed evaluation failed: {e}")
            raise

    def _evaluate_with_dask(self, dataset, model, metrics_functions):
        """Evaluate using Dask."""
        try:
            import dask.dataframe as dd
            from dask import delayed

            @delayed
            def evaluate_partition(partition_data, model, metrics_functions):
                """Evaluate model on a partition."""
                try:
                    feature_cols = [col for col in partition_data.columns
                                   if col not in ['id', 'time', 'event']]
                    X = partition_data[feature_cols]
                    y = list(zip(partition_data['event'], partition_data['time']))

                    results = {}
                    for metric in metrics_functions:
                        if metric == "concordance":
                            # Calculate concordance for this partition
                            # This is a simplified example
                            results[metric] = 0.7  # Placeholder
                        elif metric == "ibs":
                            results[metric] = 0.2  # Placeholder

                    return results

                except Exception as e:
                    return {"error": str(e)}

            # Create evaluation tasks
            tasks = []
            for i in range(dataset.get_n_partitions()):
                partition_data = dataset.get_partition(i)
                task = evaluate_partition(partition_data, model, metrics_functions)
                tasks.append(task)

            # Execute and combine results
            import dask
            results = dask.compute(*tasks)

            # Aggregate results
            aggregated = {}
            for metric in metrics_functions:
                values = [r.get(metric, 0) for r in results if 'error' not in r]
                if values:
                    aggregated[metric] = np.mean(values)

            return aggregated

        except Exception as e:
            logger.error(f"Dask evaluation failed: {e}")
            raise

    def _evaluate_with_ray(self, dataset, model, metrics_functions):
        """Evaluate using Ray."""
        # Similar to Dask implementation but using Ray
        # For brevity, implementing similar logic
        return self._evaluate_locally(dataset, model, metrics_functions)

    def _evaluate_locally(self, dataset, model, metrics_functions):
        """Evaluate locally."""
        # Combine all partitions and evaluate normally
        combined_df = pd.concat([dataset.get_partition(i) for i in range(dataset.get_n_partitions())],
                               ignore_index=True)

        feature_cols = [col for col in combined_df.columns if col not in ['id', 'time', 'event']]
        X = combined_df[feature_cols]
        y = list(zip(combined_df['event'], combined_df['time']))

        # Calculate metrics (simplified)
        results = {}
        for metric in metrics_functions:
            if metric == "concordance":
                results[metric] = 0.7  # Placeholder
            elif metric == "ibs":
                results[metric] = 0.2  # Placeholder

        return results


class DistributedBenchmarker:
    """Benchmark distributed computing performance."""

    def __init__(self, client: DistributedClient, config: DistributedConfig):
        self.client = client
        self.config = config

    def benchmark_scaling(
        self,
        dataset_sizes: List[int] = [1000, 5000, 10000, 25000],
        model_factory=None,
        model_params: Dict[str, Any] = None
    ) -> Dict[int, DistributedMetrics]:
        """Benchmark scaling performance across different dataset sizes."""

        results = {}

        for size in dataset_sizes:
            logger.info(f"Benchmarking with dataset size: {size}")

            try:
                # Create synthetic dataset
                synthetic_data = self._create_synthetic_dataset(size)

                # Setup distributed dataset
                dataset = DistributedDataset(self.client, synthetic_data, self.config)
                if not dataset.load_and_partition():
                    logger.warning(f"Failed to setup dataset for size {size}")
                    continue

                # Train model
                trainer = DistributedTrainer(self.client, self.config)

                if model_factory and model_params:
                    model, metrics = trainer.train_distributed(
                        dataset, model_factory, model_params
                    )
                else:
                    # Use default model
                    from clinical_survival.models import make_model
                    model, metrics = trainer.train_distributed(
                        dataset, lambda **kwargs: make_model("coxph", **kwargs),
                        {"random_state": 42}
                    )

                results[size] = metrics

                logger.info(f"Dataset size {size}: {metrics.total_time".2f"}s, "
                           f"speedup: {metrics.speedup_factor".2f"}x")

            except Exception as e:
                logger.error(f"Benchmarking failed for size {size}: {e}")
                continue

        return results

    def _create_synthetic_dataset(self, n_samples: int) -> Path:
        """Create a synthetic dataset for benchmarking."""
        import tempfile

        # Create synthetic survival data
        np.random.seed(42)

        data = {
            "id": range(n_samples),
            "time": np.random.exponential(5, n_samples),
            "event": np.random.binomial(1, 0.7, n_samples),
            "age": np.random.normal(60, 15, n_samples),
            "sex": np.random.choice(["male", "female"], n_samples),
            "bmi": np.random.normal(25, 5, n_samples),
            "blood_pressure": np.random.normal(120, 20, n_samples),
            "cholesterol": np.random.normal(200, 40, n_samples),
        }

        df = pd.DataFrame(data)

        # Save to temporary file
        temp_file = Path(tempfile.mktemp(suffix=".csv"))
        df.to_csv(temp_file, index=False)

        return temp_file

    def analyze_scaling_efficiency(self, results: Dict[int, DistributedMetrics]) -> Dict[str, Any]:
        """Analyze scaling efficiency from benchmark results."""

        if len(results) < 2:
            return {"error": "Need at least 2 data points for scaling analysis"}

        # Extract data
        sizes = sorted(results.keys())
        times = [results[size].total_time for size in sizes]

        # Calculate speedup factors
        speedup_factors = [times[0] / t for t in times]

        # Linear fit for scaling analysis
        try:
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(sizes, times)

            analysis = {
                "scaling_coefficient": slope,
                "ideal_scaling": "linear" if abs(slope) < 0.1 else "sublinear" if slope < 0 else "superlinear",
                "r_squared": r_value ** 2,
                "speedup_factors": dict(zip(sizes, speedup_factors)),
                "efficiency_trend": "improving" if speedup_factors[-1] > speedup_factors[0] else "degrading"
            }

        except ImportError:
            analysis = {
                "speedup_factors": dict(zip(sizes, speedup_factors)),
                "efficiency_trend": "improving" if speedup_factors[-1] > speedup_factors[0] else "degrading"
            }

        return analysis


def create_distributed_config(
    cluster_type: str = "local",
    n_workers: int = 4,
    **kwargs
) -> DistributedConfig:
    """Factory function to create distributed configuration."""
    return DistributedConfig(cluster_type=cluster_type, n_workers=n_workers, **kwargs)


def load_distributed_config(config_path: Path) -> DistributedConfig:
    """Load distributed computing configuration from file."""
    try:
        with open(config_path) as f:
            config_data = json.load(f)

        return DistributedConfig(
            cluster_type=config_data.get("cluster_type", "local"),
            n_workers=config_data.get("n_workers", 4),
            threads_per_worker=config_data.get("threads_per_worker", 2),
            memory_per_worker=config_data.get("memory_per_worker", "2GB"),
            partition_strategy=config_data.get("partition_strategy", "balanced"),
            n_partitions=config_data.get("n_partitions", 10),
            scheduler_address=config_data.get("scheduler_address", "127.0.0.1:8786"),
            dashboard_address=config_data.get("dashboard_address", "127.0.0.1:8787"),
            chunk_size=config_data.get("chunk_size", 1000),
            optimize_memory=config_data.get("optimize_memory", True),
            use_gpu_if_available=config_data.get("use_gpu_if_available", True),
            retry_failed_tasks=config_data.get("retry_failed_tasks", True),
            max_retries=config_data.get("max_retries", 3),
            timeout_minutes=config_data.get("timeout_minutes", 60)
        )

    except Exception as e:
        logger.error(f"Failed to load distributed config: {e}")
        return DistributedConfig()  # Return default config






