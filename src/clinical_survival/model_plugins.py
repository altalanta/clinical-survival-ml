from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import list, Type, Dict

import numpy as np
import pandas as pd


class BaseSurvivalModel(ABC):
    """Abstract Base Class for all survival model wrappers."""

    name: str
    feature_names_: list[str] | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence) -> BaseSurvivalModel:
        raise NotImplementedError

    @abstractmethod
    def predict_risk(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_survival_function(
        self, X: pd.DataFrame | np.ndarray, times: Iterable[float]
    ) -> np.ndarray:
        raise NotImplementedError


class ModelRegistry:
    """A registry for dynamically managing and retrieving survival models."""

    _models: Dict[str, Type[BaseSurvivalModel]] = {}

    @classmethod
    def register(cls, model_class: Type[BaseSurvivalModel]) -> Type[BaseSurvivalModel]:
        """Registers a model class with the registry."""
        if not issubclass(model_class, BaseSurvivalModel):
            raise ValueError("Registered class must inherit from BaseSurvivalModel")
        cls._models[model_class.name] = model_class
        return model_class

    @classmethod
    def get_model(cls, name: str) -> Type[BaseSurvivalModel]:
        """Retrieves a model class by its registered name."""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """Lists the names of all registered models."""
        return list(cls._models.keys())
