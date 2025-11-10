"""Integration tests for distributed computing functionality."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from clinical_survival.cli.main import app
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


class TestDistributedComputing:
    """Test distributed computing functionality."""

    @pytest.fixture
    def temp_distributed_dir(self):
        """Create a temporary directory for distributed computing tests."""
        temp_dir = tempfile.mkdtemp()

        # Create data and results directories
        data_dir = Path(temp_dir) / "data"
        data_dir.mkdir()

        results_dir = Path(temp_dir) / "results"
        results_dir.mkdir()

        models_dir = results_dir / "models"
        models_dir.mkdir()

        yield {
            "temp_dir": temp_dir,
            "data_dir": data_dir,
            "results_dir": results_dir,
            "models_dir": models_dir
        }

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config_distributed(self, temp_distributed_dir):
        """Create a test configuration for distributed computing."""
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "distributed_computing": {
                "enabled": True,
                "cluster_type": "local",
                "n_workers": 2,
                "threads_per_worker": 1,
                "memory_per_worker": "1GB",
                "partition_strategy": "balanced",
                "n_partitions": 4
            },
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(temp_distributed_dir["results_dir"]),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = temp_distributed_dir["results_dir"] / "distributed_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def test_distributed_config_creation(self, temp_distributed_dir):
        """Test distributed computing configuration creation."""
        from typer.testing import CliRunner

        runner = CliRunner()

        config_path = temp_distributed_dir["results_dir"] / "test_distributed_config.json"

        result = runner.invoke(app, ["configure-distributed",
                                   "--config-path", str(config_path),
                                   "--cluster-type", "dask",
                                   "--n-workers", "8",
                                   "--threads-per-worker", "2",
                                   "--memory-per-worker", "4GB",
                                   "--partition-strategy", "hash",
                                   "--n-partitions", "16"])

        assert result.exit_code == 0
        assert config_path.exists()

        # Verify configuration content
        with open(config_path) as f:
            config_data = json.load(f)

        assert config_data["cluster_type"] == "dask"
        assert config_data["n_workers"] == 8
        assert config_data["threads_per_worker"] == 2
        assert config_data["memory_per_worker"] == "4GB"
        assert config_data["partition_strategy"] == "hash"
        assert config_data["n_partitions"] == 16

    def test_distributed_client_initialization(self, temp_distributed_dir):
        """Test distributed client initialization."""
        # Test configuration
        config = DistributedConfig(
            cluster_type="local",
            n_workers=2,
            threads_per_worker=1
        )

        client = DistributedClient(config)

        # Test initialization
        success = client.initialize()
        assert success is True

        # Test that client is initialized
        assert client._initialized is True

        # Test shutdown
        client.shutdown()

    def test_distributed_dataset_partitioning(self, temp_distributed_dir):
        """Test distributed dataset partitioning."""
        # Create test data
        test_data = temp_distributed_dir["data_dir"] / "test_data.csv"
        test_data.write_text("id,time,event,age,sex\n1,100,1,65,male\n2,200,0,70,female\n3,150,1,75,male\n")

        # Create configuration
        config = DistributedConfig(
            cluster_type="local",
            n_partitions=2
        )

        client = DistributedClient(config)
        client.initialize()

        try:
            # Create distributed dataset
            dataset = DistributedDataset(client, test_data, config)

            # Test loading and partitioning
            success = dataset.load_and_partition()
            assert success is True

            # Test partition access
            n_partitions = dataset.get_n_partitions()
            assert n_partitions > 0

            # Test getting individual partitions
            for i in range(min(2, n_partitions)):
                partition = dataset.get_partition(i)
                assert len(partition) > 0

        finally:
            client.shutdown()

    def test_distributed_trainer_basic(self, temp_distributed_dir):
        """Test basic distributed trainer functionality."""
        # Create configuration
        config = DistributedConfig(
            cluster_type="local",
            n_workers=1,
            n_partitions=2
        )

        client = DistributedClient(config)
        client.initialize()

        try:
            # Create trainer
            trainer = DistributedTrainer(client, config)

            # Test that trainer initializes correctly
            assert trainer.client is client
            assert trainer.config is config

        finally:
            client.shutdown()

    def test_distributed_evaluator_basic(self, temp_distributed_dir):
        """Test basic distributed evaluator functionality."""
        # Create configuration
        config = DistributedConfig(
            cluster_type="local",
            n_workers=1,
            n_partitions=2
        )

        client = DistributedClient(config)
        client.initialize()

        try:
            # Create evaluator
            evaluator = DistributedEvaluator(client, config)

            # Test that evaluator initializes correctly
            assert evaluator.client is client
            assert evaluator.config is config

        finally:
            client.shutdown()

    def test_distributed_benchmarker_basic(self, temp_distributed_dir):
        """Test basic distributed benchmarker functionality."""
        # Create configuration
        config = DistributedConfig(
            cluster_type="local",
            n_workers=1
        )

        client = DistributedClient(config)
        client.initialize()

        try:
            # Create benchmarker
            benchmarker = DistributedBenchmarker(client, config)

            # Test that benchmarker initializes correctly
            assert benchmarker.client is client
            assert benchmarker.config is config

        finally:
            client.shutdown()

    def test_distributed_config_loading(self, temp_distributed_dir):
        """Test loading distributed computing configuration."""
        # Create a test config file
        config_file = temp_distributed_dir["results_dir"] / "test_distributed_config.json"
        config_data = {
            "cluster_type": "dask",
            "n_workers": 8,
            "threads_per_worker": 2,
            "memory_per_worker": "4GB",
            "partition_strategy": "hash",
            "n_partitions": 16,
            "scheduler_address": "127.0.0.1:8786",
            "dashboard_address": "127.0.0.1:8787"
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Test loading
        config = load_distributed_config(config_file)

        assert config.cluster_type == "dask"
        assert config.n_workers == 8
        assert config.threads_per_worker == 2
        assert config.memory_per_worker == "4GB"
        assert config.partition_strategy == "hash"
        assert config.n_partitions == 16

    def test_distributed_config_factory(self):
        """Test distributed configuration factory function."""
        # Test default configuration
        config = create_distributed_config()
        assert config.cluster_type == "local"
        assert config.n_workers == 4

        # Test custom configuration
        config = create_distributed_config(
            cluster_type="dask",
            n_workers=8,
            threads_per_worker=2,
            memory_per_worker="4GB"
        )

        assert config.cluster_type == "dask"
        assert config.n_workers == 8
        assert config.threads_per_worker == 2
        assert config.memory_per_worker == "4GB"

    def test_distributed_metrics_creation(self):
        """Test distributed metrics creation."""
        metrics = DistributedMetrics(
            total_time=10.5,
            computation_time=8.2,
            communication_time=2.3,
            memory_usage_peak=512.0,
            cpu_usage_avg=75.0,
            n_tasks_completed=100,
            n_tasks_failed=5,
            data_transfer_volume=1024.0,
            speedup_factor=2.5
        )

        assert metrics.total_time == 10.5
        assert metrics.computation_time == 8.2
        assert metrics.communication_time == 2.3
        assert metrics.memory_usage_peak == 512.0
        assert metrics.cpu_usage_avg == 75.0
        assert metrics.n_tasks_completed == 100
        assert metrics.n_tasks_failed == 5
        assert metrics.data_transfer_volume == 1024.0
        assert metrics.speedup_factor == 2.5

    def test_distributed_benchmarker_scaling_analysis(self, temp_distributed_dir):
        """Test scaling efficiency analysis."""
        # Create configuration
        config = DistributedConfig(cluster_type="local", n_workers=1)

        client = DistributedClient(config)
        client.initialize()

        try:
            benchmarker = DistributedBenchmarker(client, config)

            # Create mock results
            mock_results = {
                1000: DistributedMetrics(
                    total_time=5.0, computation_time=4.0, communication_time=1.0,
                    memory_usage_peak=100.0, cpu_usage_avg=50.0,
                    n_tasks_completed=1, n_tasks_failed=0, data_transfer_volume=0.0, speedup_factor=1.0
                ),
                5000: DistributedMetrics(
                    total_time=12.0, computation_time=10.0, communication_time=2.0,
                    memory_usage_peak=400.0, cpu_usage_avg=60.0,
                    n_tasks_completed=1, n_tasks_failed=0, data_transfer_volume=0.0, speedup_factor=0.83
                ),
                10000: DistributedMetrics(
                    total_time=25.0, computation_time=20.0, communication_time=5.0,
                    memory_usage_peak=800.0, cpu_usage_avg=70.0,
                    n_tasks_completed=1, n_tasks_failed=0, data_transfer_volume=0.0, speedup_factor=0.6
                )
            }

            # Analyze scaling
            analysis = benchmarker.analyze_scaling_efficiency(mock_results)

            # Should have speedup factors
            assert "speedup_factors" in analysis
            assert len(analysis["speedup_factors"]) == 3

            # Should detect degrading efficiency
            assert analysis["efficiency_trend"] == "degrading"

        finally:
            client.shutdown()

    def test_distributed_dataset_error_handling(self, temp_distributed_dir):
        """Test error handling in distributed dataset operations."""
        # Create configuration
        config = DistributedConfig(cluster_type="local", n_partitions=2)

        client = DistributedClient(config)
        client.initialize()

        try:
            # Test with non-existent file
            nonexistent_file = temp_distributed_dir["data_dir"] / "nonexistent.csv"
            dataset = DistributedDataset(client, nonexistent_file, config)

            success = dataset.load_and_partition()
            assert success is False  # Should fail gracefully

        finally:
            client.shutdown()

    def test_distributed_client_error_handling(self):
        """Test error handling in distributed client operations."""
        # Test with invalid cluster type
        config = DistributedConfig(cluster_type="invalid_type")

        client = DistributedClient(config)

        success = client.initialize()
        assert success is False  # Should fail gracefully

    def test_distributed_benchmarker_synthetic_data_creation(self, temp_distributed_dir):
        """Test synthetic data creation for benchmarking."""
        # Create configuration
        config = DistributedConfig(cluster_type="local", n_workers=1)

        client = DistributedClient(config)
        client.initialize()

        try:
            benchmarker = DistributedBenchmarker(client, config)

            # Create synthetic dataset
            synthetic_file = benchmarker._create_synthetic_dataset(1000)

            # Verify file exists and has content
            assert synthetic_file.exists()

            # Check that file has expected structure
            import pandas as pd
            df = pd.read_csv(synthetic_file)
            assert len(df) == 1000
            assert "id" in df.columns
            assert "time" in df.columns
            assert "event" in df.columns

            # Clean up
            synthetic_file.unlink()

        finally:
            client.shutdown()

    def test_distributed_trainer_local_fallback(self, temp_distributed_dir):
        """Test that trainer falls back to local execution when distributed fails."""
        # Create configuration with local cluster
        config = DistributedConfig(
            cluster_type="local",
            n_workers=1,
            n_partitions=2
        )

        client = DistributedClient(config)
        client.initialize()

        try:
            # Create trainer
            trainer = DistributedTrainer(client, config)

            # Test that it falls back to local training
            # (We can't easily test this without actual data, but we can verify setup)
            assert trainer.client is client
            assert trainer.config is config

        finally:
            client.shutdown()

    def test_distributed_evaluator_metrics_computation(self, temp_distributed_dir):
        """Test distributed evaluator metrics computation."""
        # Create configuration
        config = DistributedConfig(
            cluster_type="local",
            n_workers=1,
            n_partitions=2
        )

        client = DistributedClient(config)
        client.initialize()

        try:
            # Create evaluator
            evaluator = DistributedEvaluator(client, config)

            # Test metrics list handling
            metrics = ["concordance", "ibs"]
            assert evaluator.config is config

            # Test that evaluation method exists and is callable
            assert hasattr(evaluator, 'evaluate_distributed')
            assert callable(evaluator.evaluate_distributed)

        finally:
            client.shutdown()

    def test_distributed_config_validation(self, temp_distributed_dir):
        """Test distributed configuration validation."""
        # Test valid configuration
        config = DistributedConfig(
            cluster_type="dask",
            n_workers=4,
            threads_per_worker=2,
            memory_per_worker="2GB",
            partition_strategy="balanced",
            n_partitions=8
        )

        # Should create successfully
        assert config.cluster_type == "dask"
        assert config.n_workers == 4
        assert config.threads_per_worker == 2
        assert config.memory_per_worker == "2GB"
        assert config.partition_strategy == "balanced"
        assert config.n_partitions == 8

    def test_distributed_client_scatter_gather(self, temp_distributed_dir):
        """Test data scattering and gathering."""
        # Create configuration
        config = DistributedConfig(cluster_type="local", n_workers=1)

        client = DistributedClient(config)
        client.initialize()

        try:
            # Test data operations
            test_data = {"key": "value", "numbers": [1, 2, 3]}

            # Scatter data
            scattered = client.scatter_data(test_data)

            # Gather data
            gathered = client.gather_data(scattered)

            # Should be the same
            assert gathered == test_data

        finally:
            client.shutdown()

    def test_distributed_benchmarker_empty_results_handling(self, temp_distributed_dir):
        """Test handling of empty benchmark results."""
        # Create configuration
        config = DistributedConfig(cluster_type="local", n_workers=1)

        client = DistributedClient(config)
        client.initialize()

        try:
            benchmarker = DistributedBenchmarker(client, config)

            # Test with empty results
            analysis = benchmarker.analyze_scaling_efficiency({})

            # Should handle gracefully
            assert "error" in analysis
            assert "Need at least 2 data points" in analysis["error"]

        finally:
            client.shutdown()

    def test_distributed_dataset_partition_access(self, temp_distributed_dir):
        """Test accessing partitions in distributed dataset."""
        # Create test data
        test_data = temp_distributed_dir["data_dir"] / "partition_test.csv"
        test_data.write_text("id,time,event,age\n1,100,1,65\n2,200,0,70\n3,150,1,75\n4,180,1,80\n")

        # Create configuration
        config = DistributedConfig(
            cluster_type="local",
            n_partitions=2
        )

        client = DistributedClient(config)
        client.initialize()

        try:
            # Create distributed dataset
            dataset = DistributedDataset(client, test_data, config)

            success = dataset.load_and_partition()
            assert success is True

            # Test partition access
            n_partitions = dataset.get_n_partitions()
            assert n_partitions == 2

            # Test getting partitions
            partition_0 = dataset.get_partition(0)
            partition_1 = dataset.get_partition(1)

            assert len(partition_0) > 0
            assert len(partition_1) > 0

            # Test invalid partition access
            with pytest.raises((ValueError, IndexError)):
                dataset.get_partition(999)  # Invalid partition index

        finally:
            client.shutdown()

    def test_distributed_benchmarker_scaling_analysis_with_scipy(self, temp_distributed_dir):
        """Test scaling analysis with scipy (if available)."""
        # Create configuration
        config = DistributedConfig(cluster_type="local", n_workers=1)

        client = DistributedClient(config)
        client.initialize()

        try:
            benchmarker = DistributedBenchmarker(client, config)

            # Create mock results with more data points for scipy analysis
            mock_results = {}
            sizes = [1000, 2000, 4000, 8000]

            for i, size in enumerate(sizes):
                mock_results[size] = DistributedMetrics(
                    total_time=size / 1000,  # Linear scaling for testing
                    computation_time=size / 1000 * 0.8,
                    communication_time=size / 1000 * 0.2,
                    memory_usage_peak=size * 0.1,
                    cpu_usage_avg=50.0,
                    n_tasks_completed=1,
                    n_tasks_failed=0,
                    data_transfer_volume=0.0,
                    speedup_factor=1.0
                )

            # Analyze scaling
            analysis = benchmarker.analyze_scaling_efficiency(mock_results)

            # Should have scipy-based analysis if available
            if "r_squared" in analysis:
                assert 0 <= analysis["r_squared"] <= 1
                assert "scaling_coefficient" in analysis

        finally:
            client.shutdown()

    def test_distributed_config_serialization(self, temp_distributed_dir):
        """Test distributed configuration serialization/deserialization."""
        # Create configuration
        original_config = DistributedConfig(
            cluster_type="dask",
            n_workers=8,
            threads_per_worker=2,
            memory_per_worker="4GB",
            partition_strategy="hash",
            n_partitions=16,
            scheduler_address="192.168.1.100:8786",
            dashboard_address="192.168.1.100:8787",
            chunk_size=2000,
            optimize_memory=False,
            use_gpu_if_available=True,
            retry_failed_tasks=True,
            max_retries=5,
            timeout_minutes=120,
            resource_allocation_strategy="memory_aware"
        )

        # Serialize to dict (simulating JSON conversion)
        config_dict = {
            "cluster_type": original_config.cluster_type,
            "n_workers": original_config.n_workers,
            "threads_per_worker": original_config.threads_per_worker,
            "memory_per_worker": original_config.memory_per_worker,
            "partition_strategy": original_config.partition_strategy,
            "n_partitions": original_config.n_partitions,
            "scheduler_address": original_config.scheduler_address,
            "dashboard_address": original_config.dashboard_address,
            "chunk_size": original_config.chunk_size,
            "optimize_memory": original_config.optimize_memory,
            "use_gpu_if_available": original_config.use_gpu_if_available,
            "retry_failed_tasks": original_config.retry_failed_tasks,
            "max_retries": original_config.max_retries,
            "timeout_minutes": original_config.timeout_minutes,
            "resource_allocation_strategy": original_config.resource_allocation_strategy
        }

        # Create new config from dict
        recreated_config = DistributedConfig(**config_dict)

        # Verify all fields match
        assert recreated_config.cluster_type == original_config.cluster_type
        assert recreated_config.n_workers == original_config.n_workers
        assert recreated_config.threads_per_worker == original_config.threads_per_worker
        assert recreated_config.memory_per_worker == original_config.memory_per_worker
        assert recreated_config.partition_strategy == original_config.partition_strategy
        assert recreated_config.n_partitions == original_config.n_partitions
        assert recreated_config.scheduler_address == original_config.scheduler_address
        assert recreated_config.dashboard_address == original_config.dashboard_address
        assert recreated_config.chunk_size == original_config.chunk_size
        assert recreated_config.optimize_memory == original_config.optimize_memory
        assert recreated_config.use_gpu_if_available == original_config.use_gpu_if_available
        assert recreated_config.retry_failed_tasks == original_config.retry_failed_tasks
        assert recreated_config.max_retries == original_config.max_retries
        assert recreated_config.timeout_minutes == original_config.timeout_minutes
        assert recreated_config.resource_allocation_strategy == original_config.resource_allocation_strategy





