"""
Model plugin system with comprehensive type hints and Protocol support.

This module provides:
- BaseSurvivalModel: Abstract base class for survival model wrappers
- ModelRegistry: Registry for dynamically managing survival models
- Type definitions compatible with the types module
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd

# Type aliases for clarity
FeatureMatrix = Union[pd.DataFrame, np.ndarray]
RiskScores = np.ndarray
SurvivalProbabilities = np.ndarray
TimePoints = Iterable[float]


class BaseSurvivalModel(ABC):
    """
    Abstract Base Class for all survival model wrappers.
    
    All survival models in the pipeline must inherit from this class
    and implement the required methods.
    
    Attributes:
        name: Unique identifier for the model type (e.g., "coxph", "rsf")
        feature_names_: List of feature names after fitting (set by fit method)
        model: The underlying model object (set by subclass)
    
    Example:
        class MyModel(BaseSurvivalModel):
            name = "my_model"
            
            def __init__(self, **params):
                self.model = SomeEstimator(**params)
                
            def fit(self, X, y):
                self.feature_names_ = list(X.columns) if hasattr(X, 'columns') else None
                self.model.fit(X, y)
                return self
                
            def predict_risk(self, X):
                return self.model.predict(X)
                
            def predict_survival_function(self, X, times):
                return self.model.predict_survival_function(X, times)
    """

    name: str
    feature_names_: Optional[List[str]] = None
    model: Any = None

    @abstractmethod
    def fit(
        self, X: FeatureMatrix, y: Sequence[Any]
    ) -> "BaseSurvivalModel":
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix (DataFrame or ndarray)
            y: Survival target (structured array with event/time)
            
        Returns:
            self: The fitted model instance
        """
        raise NotImplementedError

    @abstractmethod
    def predict_risk(self, X: FeatureMatrix) -> RiskScores:
        """
        Predict risk scores for samples.
        
        Higher risk scores indicate higher probability of event.
        
        Args:
            X: Feature matrix (DataFrame or ndarray)
            
        Returns:
            1D array of risk scores, shape (n_samples,)
        """
        raise NotImplementedError

    @abstractmethod
    def predict_survival_function(
        self, X: FeatureMatrix, times: TimePoints
    ) -> SurvivalProbabilities:
        """
        Predict survival probabilities at given time points.
        
        Args:
            X: Feature matrix (DataFrame or ndarray)
            times: Time points at which to evaluate survival
            
        Returns:
            2D array of survival probabilities, shape (n_samples, n_times)
            Values are in [0, 1], representing P(T > t | X)
        """
        raise NotImplementedError

    def get_feature_names(self) -> Optional[List[str]]:
        """
        Get feature names from the fitted model.
        
        Returns:
            List of feature names, or None if not available
        """
        return self.feature_names_

    def __repr__(self) -> str:
        fitted = self.feature_names_ is not None
        return f"{self.__class__.__name__}(name='{self.name}', fitted={fitted})"


class ModelRegistry:
    """
    A registry for dynamically managing and retrieving survival models.
    
    The registry allows models to be registered by name and retrieved
    for instantiation. This enables a plugin architecture where new
    models can be added without modifying core code.
    
    Usage:
        # Register a model class
        @ModelRegistry.register
        class MyModel(BaseSurvivalModel):
            name = "my_model"
            ...
            
        # Or register manually
        ModelRegistry.register(MyModel)
        
        # Retrieve and instantiate
        model_class = ModelRegistry.get_model("my_model")
        model = model_class(**params)
        
        # List available models
        available = ModelRegistry.list_models()
    """

    _models: Dict[str, Type[BaseSurvivalModel]] = {}

    @classmethod
    def register(
        cls, model_class: Type[BaseSurvivalModel]
    ) -> Type[BaseSurvivalModel]:
        """
        Registers a model class with the registry.
        
        Can be used as a decorator or called directly.
        
        Args:
            model_class: A class that inherits from BaseSurvivalModel
            
        Returns:
            The registered model class (unchanged)
            
        Raises:
            ValueError: If model_class doesn't inherit from BaseSurvivalModel
            ValueError: If model_class doesn't have a 'name' attribute
        """
        if not issubclass(model_class, BaseSurvivalModel):
            raise ValueError(
                f"Registered class {model_class.__name__} must inherit from BaseSurvivalModel"
            )
        
        if not hasattr(model_class, "name") or not model_class.name:
            raise ValueError(
                f"Registered class {model_class.__name__} must have a 'name' attribute"
            )
            
        cls._models[model_class.name] = model_class
        return model_class

    @classmethod
    def get_model(cls, name: str) -> Type[BaseSurvivalModel]:
        """
        Retrieves a model class by its registered name.
        
        Args:
            name: The registered name of the model
            
        Returns:
            The model class
            
        Raises:
            KeyError: If no model with the given name is registered
        """
        if name not in cls._models:
            available = cls.list_models()
            raise KeyError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        """
        Lists the names of all registered models.
        
        Returns:
            List of registered model names
        """
        return list(cls._models.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a model is registered.
        
        Args:
            name: The model name to check
            
        Returns:
            True if registered, False otherwise
        """
        return name in cls._models

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered models.
        
        Primarily useful for testing.
        """
        cls._models.clear()
