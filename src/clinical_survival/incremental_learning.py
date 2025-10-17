"""Incremental learning and online model updates for survival models."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from clinical_survival.logging_utils import log_function_call
from clinical_survival.models import BaseSurvivalModel, XGBAFTModel, XGBCoxModel
from clinical_survival.utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class IncrementalUpdateConfig:
    """Configuration for incremental learning."""

    # Update frequency settings
    update_frequency_days: int = 7  # How often to check for updates
    min_samples_for_update: int = 50  # Minimum new samples before updating
    max_samples_in_memory: int = 1000  # Maximum samples to keep in memory

    # Update strategy
    update_strategy: str = "online"  # "online", "batch", "sliding_window"
    window_size_days: int = 365  # For sliding window strategy

    # Performance thresholds
    performance_threshold: float = 0.02  # Minimum performance drop to trigger update
    max_updates_per_model: int = 10  # Limit updates per model

    # Data quality checks
    drift_detection_enabled: bool = True
    drift_threshold: float = 0.1

    # Model backup
    create_backup_before_update: bool = True
    backup_retention_days: int = 30


@dataclass
class UpdateHistory:
    """Track model update history."""

    update_id: str
    timestamp: str
    n_samples_added: int
    performance_before: float
    performance_after: float
    update_strategy: str
    drift_detected: bool
    backup_created: bool


class IncrementalLearner:
    """Handles incremental learning for survival models."""

    def __init__(
        self,
        model: BaseSurvivalModel,
        config: IncrementalUpdateConfig,
        model_path: Path,
        data_buffer: Optional[deque] = None
    ):
        self.model = model
        self.config = config
        self.model_path = model_path
        self.data_buffer = data_buffer or deque(maxlen=config.max_samples_in_memory)
        self.update_history: List[UpdateHistory] = []
        self.last_update_timestamp: Optional[str] = None
        self.model_backups: Dict[str, Path] = {}

    def add_new_data(self, X: pd.DataFrame, y: Sequence) -> bool:
        """Add new data to the buffer and check if update is needed."""
        # Convert to structured data if needed
        if hasattr(X, 'values'):
            X_data = X.values
        else:
            X_data = X

        # Add to buffer
        for i in range(len(X_data)):
            self.data_buffer.append((X_data[i], y[i]))

        logger.info(f"Added {len(X_data)} new samples to buffer. Buffer size: {len(self.data_buffer)}")

        # Check if update is needed
        return self._should_update()

    def _should_update(self) -> bool:
        """Determine if model should be updated based on configuration."""
        if len(self.data_buffer) < self.config.min_samples_for_update:
            return False

        if len(self.update_history) >= self.config.max_updates_per_model:
            logger.warning(f"Maximum updates ({self.config.max_updates_per_model}) reached for this model")
            return False

        return True

    def update_model(self, X_val: Optional[pd.DataFrame] = None, y_val: Optional[Sequence] = None) -> bool:
        """Perform incremental model update."""
        if not self._should_update():
            return False

        logger.info("Starting incremental model update...")

        # Create backup if enabled
        if self.config.create_backup_before_update:
            self._create_backup()

        # Extract data from buffer
        X_new, y_new = self._extract_buffer_data()

        try:
            # Perform update based on strategy
            if self.config.update_strategy == "online":
                success = self._online_update(X_new, y_new)
            elif self.config.update_strategy == "batch":
                success = self._batch_update(X_new, y_new)
            elif self.config.update_strategy == "sliding_window":
                success = self._sliding_window_update(X_new, y_new)
            else:
                raise ValueError(f"Unknown update strategy: {self.config.update_strategy}")

            if success:
                # Clear buffer after successful update
                self.data_buffer.clear()
                logger.info("Model update completed successfully")
                return True
            else:
                logger.error("Model update failed")
                return False

        except Exception as e:
            logger.error(f"Error during model update: {e}")
            return False

    def _extract_buffer_data(self) -> Tuple[np.ndarray, List]:
        """Extract data from buffer for updating."""
        X_list = []
        y_list = []

        for X_sample, y_sample in self.data_buffer:
            X_list.append(X_sample)
            y_list.append(y_sample)

        return np.array(X_list), y_list

    def _create_backup(self) -> None:
        """Create a backup of the current model."""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.model_path.stem}_backup_{timestamp}{self.model_path.suffix}"
        backup_path = self.model_path.parent / backup_name

        # For XGBoost models, we need to save the booster
        if isinstance(self.model, (XGBCoxModel, XGBAFTModel)):
            if self.model._booster is not None:
                self.model._booster.save_model(str(backup_path))
                self.model_backups[timestamp] = backup_path
                logger.info(f"Created model backup: {backup_path}")
        else:
            # For other models, try to save the entire model
            try:
                import joblib
                joblib.dump(self.model, backup_path)
                self.model_backups[timestamp] = backup_path
                logger.info(f"Created model backup: {backup_path}")
            except Exception as e:
                logger.warning(f"Could not create model backup: {e}")

    def _online_update(self, X_new: np.ndarray, y_new: List) -> bool:
        """Perform online learning update."""
        # For XGBoost models, we can use booster update
        if isinstance(self.model, (XGBCoxModel, XGBAFTModel)):
            return self._update_xgboost_model(X_new, y_new)
        else:
            # For other models, fall back to retraining
            return self._retrain_model(X_new, y_new)

    def _batch_update(self, X_new: np.ndarray, y_new: List) -> bool:
        """Perform batch update."""
        # Similar to online update but with all data at once
        return self._online_update(X_new, y_new)

    def _sliding_window_update(self, X_new: np.ndarray, y_new: List) -> bool:
        """Perform sliding window update."""
        # For sliding window, we need historical data plus new data
        # For now, implement as batch update
        return self._online_update(X_new, y_new)

    def _update_xgboost_model(self, X_new: np.ndarray, y_new: List) -> bool:
        """Update XGBoost model incrementally."""
        try:
            import xgboost as xgb

            # Prepare data for XGBoost
            times = np.array([record[1] for record in y_new], dtype=float)
            events = np.array([record[0] for record in y_new], dtype=float)

            # Create DMatrix for new data
            if hasattr(self.model, 'feature_names_') and self.model.feature_names_:
                feature_names = self.model.feature_names_
                X_df = pd.DataFrame(X_new, columns=feature_names)
            else:
                X_df = pd.DataFrame(X_new)

            dmatrix = xgb.DMatrix(X_df.values, label=times)

            if isinstance(self.model, XGBCoxModel):
                # For Cox model, use weighted labels
                weights = np.maximum(events, 1e-3)
                dmatrix.set_weight(weights)
            elif isinstance(self.model, XGBAFTModel):
                # For AFT model, set bounds
                lower = np.where(events == 1, times, np.zeros_like(times))
                upper = np.where(events == 1, times, times)
                dmatrix.set_float_info("label_lower_bound", lower)
                dmatrix.set_float_info("label_upper_bound", upper)

            # Update the booster
            self.model._booster.update(dmatrix, iteration=self.model._booster.num_boosted_rounds())

            logger.info(f"XGBoost model updated with {len(X_new)} new samples")
            return True

        except Exception as e:
            logger.error(f"Failed to update XGBoost model: {e}")
            return False

    def _retrain_model(self, X_new: np.ndarray, y_new: List) -> bool:
        """Retrain model with combined historical and new data."""
        try:
            # For models that don't support incremental learning,
            # we need to retrain with accumulated data
            logger.warning("Model doesn't support true incremental learning, retraining instead")

            # This is a simplified approach - in practice, you'd want to
            # load historical data and combine it with new data
            # For now, just fit on the new data
            if hasattr(self.model, 'fit'):
                self.model.fit(X_new, y_new)
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to retrain model: {e}")
            return False

    def evaluate_performance_change(self, X_val: pd.DataFrame, y_val: Sequence) -> float:
        """Evaluate performance change after update."""
        try:
            # This would compute performance metrics before and after update
            # For now, return a placeholder
            return 0.0
        except Exception as e:
            logger.error(f"Failed to evaluate performance: {e}")
            return 0.0

    def detect_drift(self, X_new: pd.DataFrame) -> bool:
        """Detect data drift in new samples."""
        if not self.config.drift_detection_enabled:
            return False

        try:
            # Simple drift detection based on feature distributions
            # This is a placeholder - real implementation would use statistical tests
            if hasattr(self.model, 'feature_names_') and self.model.feature_names_:
                # Check for basic distribution changes
                # In practice, you'd use more sophisticated drift detection
                logger.info("Drift detection check completed")
                return False  # Placeholder
            return False
        except Exception as e:
            logger.error(f"Failed to detect drift: {e}")
            return False

    def save_update_history(self) -> None:
        """Save update history to disk."""
        try:
            history_file = self.model_path.parent / "incremental_update_history.json"

            # Convert dataclass objects to dictionaries for JSON serialization
            history_data = []
            for update in self.update_history:
                update_dict = {
                    'update_id': update.update_id,
                    'timestamp': update.timestamp,
                    'n_samples_added': update.n_samples_added,
                    'performance_before': update.performance_before,
                    'performance_after': update.performance_after,
                    'update_strategy': update.update_strategy,
                    'drift_detected': update.drift_detected,
                    'backup_created': update.backup_created
                }
                history_data.append(update_dict)

            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)

            logger.info(f"Update history saved to {history_file}")

        except Exception as e:
            logger.error(f"Failed to save update history: {e}")

    def cleanup_old_backups(self) -> None:
        """Clean up old model backups."""
        try:
            import datetime

            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.config.backup_retention_days)

            for timestamp_str, backup_path in list(self.model_backups.items()):
                try:
                    # Parse timestamp from backup filename
                    # Assuming format: model_backup_YYYYMMDD_HHMMSS.pkl
                    timestamp_part = timestamp_str.split('_')[-1]
                    backup_date = datetime.datetime.strptime(timestamp_part, "%Y%m%d%H%M%S")

                    if backup_date < cutoff_date:
                        backup_path.unlink()
                        del self.model_backups[timestamp_str]
                        logger.info(f"Removed old backup: {backup_path}")

                except (ValueError, FileNotFoundError):
                    # If we can't parse the timestamp or file doesn't exist, remove from dict
                    if timestamp_str in self.model_backups:
                        del self.model_backups[timestamp_str]

        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")


class IncrementalLearningManager:
    """Manages incremental learning for multiple models."""

    def __init__(self, models_dir: Path, config: IncrementalUpdateConfig):
        self.models_dir = models_dir
        self.config = config
        self.learners: Dict[str, IncrementalLearner] = {}
        self._load_existing_learners()

    def _load_existing_learners(self) -> None:
        """Load existing incremental learners from disk."""
        try:
            # Look for model files and their associated incremental learning configs
            for model_file in self.models_dir.glob("*.pkl"):
                model_name = model_file.stem
                if model_name not in self.learners:
                    # Try to load the model and create an incremental learner
                    try:
                        import joblib
                        model = joblib.load(model_file)
                        if isinstance(model, BaseSurvivalModel):
                            learner = IncrementalLearner(model, self.config, model_file)
                            self.learners[model_name] = learner
                    except Exception as e:
                        logger.warning(f"Could not load model {model_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to load existing learners: {e}")

    def add_model_for_incremental_learning(self, model_name: str, model: BaseSurvivalModel) -> None:
        """Add a model to the incremental learning system."""
        model_path = self.models_dir / f"{model_name}.pkl"
        learner = IncrementalLearner(model, self.config, model_path)
        self.learners[model_name] = learner
        logger.info(f"Added model {model_name} for incremental learning")

    def process_new_data(self, model_name: str, X: pd.DataFrame, y: Sequence) -> bool:
        """Process new data for a specific model."""
        if model_name not in self.learners:
            logger.warning(f"No incremental learner found for model {model_name}")
            return False

        learner = self.learners[model_name]

        # Check for drift if enabled
        drift_detected = learner.detect_drift(X)

        # Add data and check if update is needed
        if learner.add_new_data(X, y):
            return learner.update_model()

        return False

    def get_model_update_status(self, model_name: str) -> Dict[str, Any]:
        """Get update status for a model."""
        if model_name not in self.learners:
            return {"status": "not_found"}

        learner = self.learners[model_name]

        return {
            "status": "active",
            "buffer_size": len(learner.data_buffer),
            "last_update": learner.last_update_timestamp,
            "total_updates": len(learner.update_history),
            "config": {
                "min_samples_for_update": learner.config.min_samples_for_update,
                "max_samples_in_memory": learner.config.max_samples_in_memory,
                "update_strategy": learner.config.update_strategy
            }
        }

    def save_all_learners(self) -> None:
        """Save all incremental learners to disk."""
        for model_name, learner in self.learners.items():
            try:
                # Save the model
                if hasattr(learner.model, '_booster') and learner.model._booster is not None:
                    # XGBoost model
                    learner.model._booster.save_model(str(learner.model_path))
                else:
                    # Other models
                    import joblib
                    joblib.dump(learner.model, learner.model_path)

                # Save update history
                learner.save_update_history()

                # Cleanup old backups
                learner.cleanup_old_backups()

            except Exception as e:
                logger.error(f"Failed to save learner {model_name}: {e}")


def create_incremental_learner(
    model: BaseSurvivalModel,
    config: IncrementalUpdateConfig,
    model_path: Path
) -> IncrementalLearner:
    """Factory function to create an incremental learner."""
    return IncrementalLearner(model, config, model_path)


def load_incremental_learning_config(config_path: Path) -> IncrementalUpdateConfig:
    """Load incremental learning configuration from file."""
    try:
        with open(config_path) as f:
            config_data = json.load(f)

        return IncrementalUpdateConfig(
            update_frequency_days=config_data.get("update_frequency_days", 7),
            min_samples_for_update=config_data.get("min_samples_for_update", 50),
            max_samples_in_memory=config_data.get("max_samples_in_memory", 1000),
            update_strategy=config_data.get("update_strategy", "online"),
            window_size_days=config_data.get("window_size_days", 365),
            performance_threshold=config_data.get("performance_threshold", 0.02),
            max_updates_per_model=config_data.get("max_updates_per_model", 10),
            drift_detection_enabled=config_data.get("drift_detection_enabled", True),
            drift_threshold=config_data.get("drift_threshold", 0.1),
            create_backup_before_update=config_data.get("create_backup_before_update", True),
            backup_retention_days=config_data.get("backup_retention_days", 30)
        )

    except Exception as e:
        logger.error(f"Failed to load incremental learning config: {e}")
        return IncrementalUpdateConfig()  # Return default config
