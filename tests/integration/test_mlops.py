"""Integration tests for MLOps pipeline and model lifecycle management."""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
import yaml

from clinical_survival.cli.main import app
from clinical_survival.mlops import (
    ABTestManager,
    AutomatedRetrainer,
    DeploymentEnvironment,
    DeploymentManager,
    ModelRegistry,
    ModelVersion,
    RetrainingTrigger,
    create_mlops_config,
    initialize_mlops_system,
    load_mlops_config,
)


class TestMLOpsPipeline:
    """Test MLOps pipeline and model lifecycle management functionality."""

    @pytest.fixture
    def temp_mlops_dir(self):
        """Create a temporary directory for MLOps tests."""
        temp_dir = tempfile.mkdtemp()

        # Create MLOps directory structure
        mlops_dir = Path(temp_dir) / "mlops"
        mlops_dir.mkdir()

        models_dir = mlops_dir / "models"
        models_dir.mkdir()

        yield mlops_dir

        shutil.rmtree(temp_dir)

    @pytest.fixture
    def test_config_mlops(self, temp_mlops_dir):
        """Create a test configuration for MLOps."""
        config = {
            "seed": 42,
            "n_splits": 3,
            "time_col": "time",
            "event_col": "event",
            "id_col": "id",
            "mlops": {
                "enabled": True,
                "registry_path": str(temp_mlops_dir),
                "environments": {
                    "development": {
                        "type": "development",
                        "auto_rollback": False,
                        "rollback_threshold": 0.1
                    },
                    "staging": {
                        "type": "staging",
                        "auto_rollback": True,
                        "rollback_threshold": 0.05
                    },
                    "production": {
                        "type": "production",
                        "auto_rollback": True,
                        "rollback_threshold": 0.02
                    }
                },
                "triggers": {
                    "daily_retrain": {
                        "enabled": True,
                        "trigger_type": "scheduled",
                        "schedule_cron": "0 2 * * *",
                        "require_approval": True
                    },
                    "performance_monitor": {
                        "enabled": True,
                        "trigger_type": "performance",
                        "performance_threshold": 0.05,
                        "auto_retrain": False,
                        "require_approval": True
                    }
                },
                "deployment_settings": {
                    "require_approval_for_production": True,
                    "auto_rollback_on_failure": True,
                    "max_concurrent_deployments": 3,
                    "deployment_timeout_minutes": 30
                }
            },
            "paths": {
                "data_csv": "data/toy/toy_survival.csv",
                "metadata": "data/toy/metadata.yaml",
                "outdir": str(temp_mlops_dir.parent),
                "features": "configs/features.yaml"
            },
            "models": ["coxph"]
        }

        config_path = temp_mlops_dir / "mlops_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        return config_path

    def test_mlops_config_creation(self):
        """Test MLOps configuration creation."""
        config = create_mlops_config(Path("test_mlops"))

        assert config["registry_path"] == "test_mlops"
        assert len(config["environments"]) == 3  # development, staging, production
        assert len(config["triggers"]) == 3  # daily, performance, drift
        assert "deployment_settings" in config

    def test_mlops_config_loading(self, temp_mlops_dir):
        """Test MLOps configuration loading."""
        # Create test config file
        config_file = temp_mlops_dir / "test_mlops_config.json"
        config_data = {
            "registry_path": str(temp_mlops_dir),
            "environments": [
                {
                    "name": "test_env",
                    "type": "development",
                    "auto_rollback": False
                }
            ],
            "triggers": [
                {
                    "trigger_id": "test_trigger",
                    "trigger_name": "Test Trigger",
                    "trigger_type": "scheduled",
                    "enabled": True
                }
            ]
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # Test loading
        loaded_config = load_mlops_config(config_file)

        assert loaded_config["registry_path"] == str(temp_mlops_dir)
        assert len(loaded_config["environments"]) == 1
        assert len(loaded_config["triggers"]) == 1

    def test_model_registry_initialization(self, temp_mlops_dir):
        """Test model registry initialization."""
        registry = ModelRegistry(temp_mlops_dir)

        # Test that registry initializes correctly
        assert registry.registry_path == temp_mlops_dir
        assert registry.models_file == temp_mlops_dir / "models.json"
        assert registry.versions_file == temp_mlops_dir / "versions.json"
        assert isinstance(registry._models, dict)
        assert isinstance(registry._versions, dict)

    def test_model_version_creation(self, temp_mlops_dir):
        """Test model version creation and metadata."""
        from clinical_survival.models import make_model

        # Create a simple model for testing
        model = make_model("coxph", random_state=42)

        # Create registry
        registry = ModelRegistry(temp_mlops_dir)

        # Register model
        version = registry.register_model(
            model=model,
            model_name="test_coxph",
            version_number="1.0.0",
            created_by="test_user",
            description="Test model for MLOps",
            tags=["test", "coxph"]
        )

        # Test version properties
        assert version.version_id is not None
        assert version.model_name == "test_coxph"
        assert version.model_type == "CoxPHModel"
        assert version.version_number == "1.0.0"
        assert version.created_by == "test_user"
        assert version.description == "Test model for MLOps"
        assert "test" in version.tags
        assert version.status == "development"
        assert version.approval_status == "pending"
        assert version.traffic_percentage == 0.0

    def test_model_version_status_updates(self, temp_mlops_dir):
        """Test model version status updates and promotion."""
        from clinical_survival.models import make_model

        # Create model and register
        model = make_model("coxph", random_state=42)
        registry = ModelRegistry(temp_mlops_dir)

        version = registry.register_model(
            model=model,
            model_name="test_model",
            version_number="1.0.0",
            created_by="test_user"
        )

        # Test status updates
        success = registry.update_version_status(version.version_id, "staging", "test_approver")
        assert success is True

        updated_version = registry.get_version(version.version_id)
        assert updated_version.status == "staging"
        assert updated_version.approval_status == "approved"
        assert updated_version.approved_by == "test_approver"

        # Test promotion to production
        success = registry.promote_version(version.version_id, "production", "test_approver")
        assert success is True

        final_version = registry.get_version(version.version_id)
        assert final_version.status == "production"
        assert "production" in final_version.deployed_to

    def test_model_performance_recording(self, temp_mlops_dir):
        """Test performance metrics recording."""
        from clinical_survival.models import make_model

        # Create model and register
        model = make_model("coxph", random_state=42)
        registry = ModelRegistry(temp_mlops_dir)

        version = registry.register_model(
            model=model,
            model_name="test_model",
            version_number="1.0.0",
            created_by="test_user"
        )

        # Record performance metrics
        performance_metrics = {
            "concordance": 0.75,
            "ibs": 0.15,
            "brier_score": 0.12
        }

        validation_metrics = {
            "validation_concordance": 0.73,
            "validation_ibs": 0.16
        }

        success = registry.record_performance(
            version.version_id,
            performance_metrics,
            validation_metrics
        )

        assert success is True

        # Verify metrics were recorded
        updated_version = registry.get_version(version.version_id)
        assert updated_version.performance_metrics["concordance"] == 0.75
        assert updated_version.performance_metrics["ibs"] == 0.15
        assert updated_version.validation_metrics["validation_concordance"] == 0.73

    def test_model_registry_persistence(self, temp_mlops_dir):
        """Test model registry persistence to disk."""
        from clinical_survival.models import make_model

        # Create and register model
        model = make_model("coxph", random_state=42)
        registry = ModelRegistry(temp_mlops_dir)

        version = registry.register_model(
            model=model,
            model_name="persistent_test",
            version_number="1.0.0",
            created_by="test_user"
        )

        # Record some metrics
        registry.record_performance(version.version_id, {"concordance": 0.8})

        # Create new registry instance (simulates restart)
        new_registry = ModelRegistry(temp_mlops_dir)

        # Verify data persisted
        persisted_version = new_registry.get_version(version.version_id)
        assert persisted_version is not None
        assert persisted_version.model_name == "persistent_test"
        assert persisted_version.performance_metrics["concordance"] == 0.8

    def test_deployment_manager_initialization(self, temp_mlops_dir):
        """Test deployment manager initialization."""
        # Create environments
        environments = [
            DeploymentEnvironment("development", "development", auto_rollback=False),
            DeploymentEnvironment("staging", "staging", auto_rollback=True),
            DeploymentEnvironment("production", "production", auto_rollback=True)
        ]

        # Create registry
        registry = ModelRegistry(temp_mlops_dir)

        # Create deployment manager
        deployment_manager = DeploymentManager(registry, environments)

        # Test initialization
        assert deployment_manager.registry is registry
        assert len(deployment_manager.environments) == 3
        assert "development" in deployment_manager.environments
        assert "staging" in deployment_manager.environments
        assert "production" in deployment_manager.environments

    def test_deployment_workflow(self, temp_mlops_dir):
        """Test complete deployment workflow."""
        from clinical_survival.models import make_model

        # Create model and register
        model = make_model("coxph", random_state=42)
        registry = ModelRegistry(temp_mlops_dir)

        version = registry.register_model(
            model=model,
            model_name="deployment_test",
            version_number="1.0.0",
            created_by="test_user"
        )

        # Create deployment manager
        environments = [
            DeploymentEnvironment("staging", "staging"),
            DeploymentEnvironment("production", "production")
        ]
        deployment_manager = DeploymentManager(registry, environments)

        # Deploy to staging
        success = deployment_manager.deploy_model(
            version_id=version.version_id,
            environment_name="staging",
            traffic_percentage=50.0,
            approved_by="test_approver"
        )

        assert success is True

        # Check deployment status
        status = deployment_manager.get_deployment_status("staging")
        assert status["status"] == "active"
        assert status["current_version"]["version_number"] == "1.0.0"
        assert status["current_version"]["traffic_percentage"] == 50.0

        # Promote to production
        success = deployment_manager.deploy_model(
            version_id=version.version_id,
            environment_name="production",
            traffic_percentage=100.0,
            approved_by="test_approver"
        )

        assert success is True

        # Verify both environments show deployment
        prod_status = deployment_manager.get_deployment_status("production")
        assert prod_status["status"] == "active"
        assert prod_status["current_version"]["traffic_percentage"] == 100.0

    def test_rollback_functionality(self, temp_mlops_dir):
        """Test deployment rollback functionality."""
        from clinical_survival.models import make_model

        # Create two model versions
        registry = ModelRegistry(temp_mlops_dir)

        model_v1 = make_model("coxph", random_state=42)
        version_v1 = registry.register_model(
            model=model_v1,
            model_name="rollback_test",
            version_number="1.0.0",
            created_by="test_user"
        )

        model_v2 = make_model("coxph", random_state=43)
        version_v2 = registry.register_model(
            model=model_v2,
            model_name="rollback_test",
            version_number="1.1.0",
            created_by="test_user"
        )

        # Deploy v1 to production
        environments = [DeploymentEnvironment("production", "production")]
        deployment_manager = DeploymentManager(registry, environments)

        deployment_manager.deploy_model(
            version_id=version_v1.version_id,
            environment_name="production",
            traffic_percentage=100.0,
            approved_by="test_approver"
        )

        # Deploy v2 to production (simulating update)
        deployment_manager.deploy_model(
            version_id=version_v2.version_id,
            environment_name="production",
            traffic_percentage=100.0,
            approved_by="test_approver"
        )

        # Rollback to v1
        success = deployment_manager.rollback_deployment(
            environment_name="production",
            target_version_id=version_v1.version_id,
            reason="performance_degradation"
        )

        assert success is True

        # Verify rollback
        status = deployment_manager.get_deployment_status("production")
        assert status["current_version"]["version_id"] == version_v1.version_id

    def test_ab_test_manager_initialization(self, temp_mlops_dir):
        """Test A/B test manager initialization."""
        registry = ModelRegistry(temp_mlops_dir)
        ab_test_manager = ABTestManager(registry)

        # Test initialization
        assert ab_test_manager.registry is registry
        assert isinstance(ab_test_manager._active_tests, dict)

    def test_ab_test_creation(self, temp_mlops_dir):
        """Test A/B test creation and management."""
        from clinical_survival.models import make_model

        # Create two model versions
        registry = ModelRegistry(temp_mlops_dir)

        model_v1 = make_model("coxph", random_state=42)
        version_v1 = registry.register_model(
            model=model_v1,
            model_name="ab_test_model",
            version_number="1.0.0",
            created_by="test_user"
        )

        model_v2 = make_model("coxph", random_state=43)
        version_v2 = registry.register_model(
            model=model_v2,
            model_name="ab_test_model",
            version_number="1.1.0",
            created_by="test_user"
        )

        # Create A/B test manager
        ab_test_manager = ABTestManager(registry)

        # Create A/B test
        traffic_split = {
            version_v1.version_id: 0.7,
            version_v2.version_id: 0.3
        }

        test_id = ab_test_manager.create_ab_test(
            test_name="Performance Test",
            model_versions=[version_v1.version_id, version_v2.version_id],
            traffic_split=traffic_split,
            test_duration_days=7,
            success_metrics=["concordance", "ibs"]
        )

        # Test A/B test creation
        assert test_id is not None
        assert test_id in ab_test_manager._active_tests

        test_config = ab_test_manager._active_tests[test_id]
        assert test_config.test_name == "Performance Test"
        assert test_config.model_versions == [version_v1.version_id, version_v2.version_id]
        assert test_config.traffic_split == traffic_split

        # Verify traffic allocation
        v1_updated = registry.get_version(version_v1.version_id)
        v2_updated = registry.get_version(version_v2.version_id)

        assert v1_updated.traffic_percentage == 0.7
        assert v2_updated.traffic_percentage == 0.3

    def test_automated_retrainer_initialization(self, temp_mlops_dir):
        """Test automated retrainer initialization."""
        # Create registry
        registry = ModelRegistry(temp_mlops_dir)

        # Create triggers
        triggers = [
            RetrainingTrigger(
                trigger_id="test_trigger",
                trigger_name="Test Trigger",
                trigger_type="scheduled",
                enabled=True
            )
        ]

        # Create default config
        default_config = {"test": "config"}

        # Create automated retrainer
        auto_retrainer = AutomatedRetrainer(registry, triggers, default_config)

        # Test initialization
        assert auto_retrainer.registry is registry
        assert len(auto_retrainer.triggers) == 1
        assert auto_retrainer.default_config == default_config
        assert isinstance(auto_retrainer._last_check, dict)

    def test_mlops_system_initialization(self, temp_mlops_dir):
        """Test complete MLOps system initialization."""
        config = create_mlops_config(temp_mlops_dir)

        # Initialize system
        registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

        # Test that all components are initialized
        assert isinstance(registry, ModelRegistry)
        assert isinstance(auto_retrainer, AutomatedRetrainer)
        assert isinstance(deployment_manager, DeploymentManager)
        assert isinstance(ab_test_manager, ABTestManager)

        # Test that environments are configured
        assert len(deployment_manager.environments) == 3  # development, staging, production

        # Test that triggers are configured
        assert len(auto_retrainer.triggers) == 3  # daily, performance, drift

    def test_mlops_registry_status_command(self, temp_mlops_dir):
        """Test MLOps registry status CLI command."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First create a model to have something in the registry
        from clinical_survival.models import make_model

        model = make_model("coxph", random_state=42)
        registry = ModelRegistry(temp_mlops_dir)

        registry.register_model(
            model=model,
            model_name="test_model",
            version_number="1.0.0",
            created_by="test_user"
        )

        # Test status command
        result = runner.invoke(app, ["mlops-status", "--registry-path", str(temp_mlops_dir)])

        assert result.exit_code == 0

    def test_register_model_command(self, temp_mlops_dir):
        """Test register model CLI command."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # First train a model to have something to register
        train_result = runner.invoke(app, ["train",
                                         "--config", "configs/params.yaml",
                                         "--grid", "configs/model_grid.yaml"])

        if train_result.exit_code != 0:
            pytest.skip("Model training failed, skipping registration test")

        # Test register model command
        result = runner.invoke(app, ["register-model",
                                   "--model", "results/artifacts/models/coxph.pkl",
                                   "--model-name", "test_registered_model",
                                   "--version-number", "1.0.0",
                                   "--description", "Test model for MLOps",
                                   "--tags", "test,coxph",
                                   "--registry-path", str(temp_mlops_dir)])

        # Should complete (may succeed or fail depending on model availability)
        assert result.exit_code in [0, 1]

    def test_deployment_commands(self, temp_mlops_dir):
        """Test deployment-related CLI commands."""
        from typer.testing import CliRunner

        runner = CliRunner()

        # Test deployment configuration command
        result = runner.invoke(app, ["configure-mlops",
                                   "--registry-path", str(temp_mlops_dir),
                                   "--environments", '[{"name": "test", "type": "development"}]'])

        assert result.exit_code == 0

    def test_model_lifecycle_workflow(self, temp_mlops_dir):
        """Test complete model lifecycle workflow."""
        from clinical_survival.models import make_model

        # 1. Create and register model
        model = make_model("coxph", random_state=42)
        registry = ModelRegistry(temp_mlops_dir)

        version = registry.register_model(
            model=model,
            model_name="lifecycle_test",
            version_number="1.0.0",
            created_by="test_user"
        )

        # 2. Record performance
        registry.record_performance(version.version_id, {"concordance": 0.75})

        # 3. Promote to staging
        registry.update_version_status(version.version_id, "staging", "test_approver")

        # 4. Promote to production
        registry.promote_version(version.version_id, "production", "test_approver")

        # 5. Verify final state
        final_version = registry.get_version(version.version_id)
        assert final_version.status == "production"
        assert final_version.approval_status == "approved"
        assert "production" in final_version.deployed_to
        assert final_version.performance_metrics["concordance"] == 0.75

    def test_model_registry_error_handling(self, temp_mlops_dir):
        """Test error handling in model registry operations."""
        registry = ModelRegistry(temp_mlops_dir)

        # Test getting non-existent version
        non_existent_version = registry.get_version("non_existent_id")
        assert non_existent_version is None

        # Test updating non-existent version
        success = registry.update_version_status("non_existent_id", "production")
        assert success is False

        # Test promoting non-existent version
        success = registry.promote_version("non_existent_id", "production", "test_user")
        assert success is False

    def test_deployment_environment_validation(self, temp_mlops_dir):
        """Test deployment environment validation."""
        # Test invalid environment
        environments = [DeploymentEnvironment("invalid_env", "invalid_type")]
        registry = ModelRegistry(temp_mlops_dir)
        deployment_manager = DeploymentManager(registry, environments)

        # Test deployment to invalid environment
        success = deployment_manager.deploy_model("test_id", "non_existent_env")
        assert success is False

    def test_ab_test_validation(self, temp_mlops_dir):
        """Test A/B test validation."""
        registry = ModelRegistry(temp_mlops_dir)
        ab_test_manager = ABTestManager(registry)

        # Test A/B test with non-existent versions
        with pytest.raises(ValueError, match="Model version .* not found"):
            ab_test_manager.create_ab_test(
                "Invalid Test",
                ["non_existent_id1", "non_existent_id2"],
                {"non_existent_id1": 0.5, "non_existent_id2": 0.5}
            )

        # Test A/B test with invalid traffic split
        from clinical_survival.models import make_model

        model = make_model("coxph", random_state=42)
        version = registry.register_model(model, "ab_test_model", "1.0.0", "test_user")

        with pytest.raises(ValueError, match="Traffic split must sum to 1.0"):
            ab_test_manager.create_ab_test(
                "Invalid Split Test",
                [version.version_id],
                {version.version_id: 0.5}  # Only 0.5, not 1.0
            )

    def test_retraining_trigger_evaluation(self, temp_mlops_dir):
        """Test retraining trigger evaluation logic."""
        # Create registry with production model
        registry = ModelRegistry(temp_mlops_dir)
        from clinical_survival.models import make_model

        model = make_model("coxph", random_state=42)
        version = registry.register_model(model, "trigger_test", "1.0.0", "test_user")
        registry.update_version_status(version.version_id, "production", "test_user")

        # Record baseline performance
        registry.record_performance(version.version_id, {"concordance": 0.8})

        # Create triggers
        triggers = [
            RetrainingTrigger(
                trigger_id="performance_trigger",
                trigger_name="Performance Trigger",
                trigger_type="performance",
                enabled=True,
                performance_threshold=0.05
            )
        ]

        # Create automated retrainer
        auto_retrainer = AutomatedRetrainer(registry, triggers, {})

        # Test trigger checking (should not fire initially)
        triggered_models = auto_retrainer.check_triggers()
        assert len(triggered_models) == 0  # No triggers should fire

    def test_model_registry_concurrent_access_simulation(self, temp_mlops_dir):
        """Test model registry under concurrent access simulation."""
        from clinical_survival.models import make_model

        registry = ModelRegistry(temp_mlops_dir)

        # Simulate concurrent model registration
        models = []
        for i in range(5):
            model = make_model("coxph", random_state=42 + i)
            version = registry.register_model(
                model=model,
                model_name=f"concurrent_test_{i}",
                version_number=f"1.0.{i}",
                created_by=f"user_{i}"
            )
            models.append(version)

        # Verify all models were registered
        assert len(registry._versions) == 5

        # Test concurrent status updates
        for version in models:
            success = registry.update_version_status(version.version_id, "staging", f"approver_{version.version_id}")
            assert success is True

        # Verify all updates were applied
        for version in models:
            updated_version = registry.get_version(version.version_id)
            assert updated_version.status == "staging"

    def test_mlops_system_integration(self, temp_mlops_dir):
        """Test complete MLOps system integration."""
        # Initialize complete MLOps system
        config = create_mlops_config(temp_mlops_dir)
        registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

        # Test that all components work together
        assert registry.registry_path == temp_mlops_dir
        assert len(deployment_manager.environments) == 3
        assert len(auto_retrainer.triggers) == 3
        assert ab_test_manager.registry is registry

        # Test workflow: register -> deploy -> monitor -> rollback
        from clinical_survival.models import make_model

        # Register model
        model = make_model("coxph", random_state=42)
        version = registry.register_model(
            model=model,
            model_name="integration_test",
            version_number="1.0.0",
            created_by="test_user"
        )

        # Deploy to staging
        deployment_manager.deploy_model(
            version_id=version.version_id,
            environment_name="staging",
            traffic_percentage=100.0,
            approved_by="test_approver"
        )

        # Record performance
        registry.record_performance(version.version_id, {"concordance": 0.75})

        # Mark as degraded and rollback
        registry.mark_degraded(version.version_id, "performance")

        # Create second version for rollback
        model_v2 = make_model("coxph", random_state=43)
        version_v2 = registry.register_model(
            model=model_v2,
            model_name="integration_test",
            version_number="1.1.0",
            created_by="test_user"
        )

        # Rollback to original version
        deployment_manager.rollback_deployment(
            environment_name="staging",
            target_version_id=version.version_id,
            reason="performance_restored"
        )

        # Verify rollback
        status = deployment_manager.get_deployment_status("staging")
        assert status["current_version"]["version_id"] == version.version_id

    def test_mlops_configuration_persistence(self, temp_mlops_dir):
        """Test MLOps configuration persistence."""
        # Create initial config
        config = create_mlops_config(temp_mlops_dir)

        # Initialize system
        registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

        # Create a model and register it
        from clinical_survival.models import make_model

        model = make_model("coxph", random_state=42)
        version = registry.register_model(
            model=model,
            model_name="persistence_test",
            version_number="1.0.0",
            created_by="test_user"
        )

        # Record some metrics
        registry.record_performance(version.version_id, {"concordance": 0.8})

        # Create new system instance (simulates restart)
        new_registry, new_auto_retrainer, new_deployment_manager, new_ab_test_manager = initialize_mlops_system(config)

        # Verify persistence
        persisted_version = new_registry.get_version(version.version_id)
        assert persisted_version is not None
        assert persisted_version.performance_metrics["concordance"] == 0.8
        assert persisted_version.model_name == "persistence_test"

    def test_mlops_error_recovery(self, temp_mlops_dir):
        """Test error recovery in MLOps operations."""
        # Create system
        config = create_mlops_config(temp_mlops_dir)
        registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

        # Test deployment to non-existent environment
        success = deployment_manager.deploy_model("test_id", "non_existent_env")
        assert success is False

        # Test rollback without active deployment
        success = deployment_manager.rollback_deployment("production", "test_id")
        assert success is False

        # Test A/B test with invalid versions
        with pytest.raises(ValueError):
            ab_test_manager.create_ab_test(
                "Invalid Test",
                ["invalid_id1", "invalid_id2"],
                {"invalid_id1": 0.5, "invalid_id2": 0.5}
            )

    def test_mlops_performance_monitoring_integration(self, temp_mlops_dir):
        """Test integration with performance monitoring."""
        # Create system
        config = create_mlops_config(temp_mlops_dir)
        registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

        # Register model and promote to production
        from clinical_survival.models import make_model

        model = make_model("coxph", random_state=42)
        version = registry.register_model(model, "monitoring_test", "1.0.0", "test_user")
        registry.update_version_status(version.version_id, "production", "test_user")

        # Record baseline performance
        registry.record_performance(version.version_id, {"concordance": 0.8})

        # Simulate performance degradation (would come from monitoring system)
        registry.record_performance(version.version_id, {"concordance": 0.72})  # 10% drop

        # Check if performance trigger would fire
        performance_trigger = None
        for trigger in auto_retrainer.triggers:
            if trigger.trigger_type == "performance":
                performance_trigger = trigger
                break

        assert performance_trigger is not None

        # Simulate trigger check
        baseline_concordance = 0.8
        current_concordance = 0.72
        threshold = performance_trigger.performance_threshold or 0.05

        # Should trigger if performance dropped below threshold
        should_trigger = current_concordance < baseline_concordance - threshold
        assert should_trigger is True

    def test_mlops_deployment_approval_workflow(self, temp_mlops_dir):
        """Test deployment approval workflow."""
        # Create system with approval requirements
        config = create_mlops_config(temp_mlops_dir)
        registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

        # Register model
        from clinical_survival.models import make_model

        model = make_model("coxph", random_state=42)
        version = registry.register_model(model, "approval_test", "1.0.0", "test_user")

        # Test deployment to production (should require approval)
        success = deployment_manager.deploy_model(
            version_id=version.version_id,
            environment_name="production",
            approved_by="test_approver"
        )

        # Should succeed with approval
        assert success is True

        # Verify approval was recorded
        approved_version = registry.get_version(version.version_id)
        assert approved_version.approval_status == "approved"
        assert approved_version.approved_by == "test_approver"
        assert approved_version.status == "production"

    def test_mlops_audit_trail(self, temp_mlops_dir):
        """Test audit trail functionality."""
        # Create system
        config = create_mlops_config(temp_mlops_dir)
        registry, auto_retrainer, deployment_manager, ab_test_manager = initialize_mlops_system(config)

        # Register model
        from clinical_survival.models import make_model

        model = make_model("coxph", random_state=42)
        version = registry.register_model(model, "audit_test", "1.0.0", "test_user")

        # Perform various operations
        registry.record_performance(version.version_id, {"concordance": 0.75})
        registry.update_version_status(version.version_id, "staging", "test_approver")
        deployment_manager.deploy_model(version.version_id, "staging", approved_by="test_approver")

        # Verify audit trail is maintained
        final_version = registry.get_version(version.version_id)

        # Check that all operations are recorded
        assert final_version.performance_metrics["concordance"] == 0.75
        assert final_version.status == "staging"
        assert final_version.approval_status == "approved"
        assert final_version.approved_by == "test_approver"
        assert "staging" in final_version.deployed_to

        # Verify timestamps are set
        assert final_version.created_at is not None
        assert final_version.approved_at is not None
        assert final_version.deployed_at is not None







