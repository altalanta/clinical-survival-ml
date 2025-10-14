"""REST API server for clinical survival ML models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from clinical_survival.models import PipelineModel
from clinical_survival.utils import load_json, load_yaml


class PredictionRequest(BaseModel):
    """Request model for survival predictions."""

    features: dict[str, Any] = Field(..., description="Patient features for prediction")
    time_horizons: list[int] = Field(default=[90, 180, 365], description="Time horizons in days")
    model_name: str | None = Field(default=None, description="Specific model to use")


class PredictionResponse(BaseModel):
    """Response model for survival predictions."""

    risk_score: float = Field(..., description="Risk score (higher = worse prognosis)")
    survival_probabilities: dict[str, float] = Field(
        ..., description="Survival probabilities at each horizon"
    )
    model_used: str = Field(..., description="Model used for prediction")
    feature_importance: dict[str, float] | None = Field(
        default=None, description="Feature importance scores"
    )


class ModelInfo(BaseModel):
    """Model information response."""

    name: str
    description: str
    features: list[str]
    time_horizons: list[int]
    training_date: str | None
    performance_metrics: dict[str, Any] | None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_loaded: int
    version: str


class SurvivalAPIServer:
    """REST API server for clinical survival models."""

    def __init__(self, models_dir: str | Path, config_path: str | Path | None = None):
        self.models_dir = Path(models_dir)
        self.config_path = Path(config_path) if config_path else None
        self.models: dict[str, PipelineModel] = {}
        self.model_info: dict[str, dict[str, Any]] = {}
        self.app = FastAPI(
            title="Clinical Survival ML API",
            description="REST API for clinical survival analysis models",
            version="1.0.0",
        )
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Root endpoint with API information."""
            return """
            <html>
                <head><title>Clinical Survival ML API</title></head>
                <body>
                    <h1>Clinical Survival ML API</h1>
                    <p>REST API for clinical survival analysis models</p>
                    <h2>Available endpoints:</h2>
                    <ul>
                        <li><a href="/health">/health</a> - Health check</li>
                        <li><a href="/models">/models</a> - List available models</li>
                        <li><a href="/predict">/predict</a> - Make predictions (POST)</li>
                        <li><a href="/docs">/docs</a> - Interactive API documentation</li>
                    </ul>
                </body>
            </html>
            """

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy", models_loaded=len(self.models), version=self._get_version()
            )

        @self.app.get("/models", response_model=list[ModelInfo])
        async def list_models():
            """List available models."""
            return [
                ModelInfo(
                    name=name,
                    description=f"Clinical survival model: {name}",
                    features=info.get("features", []),
                    time_horizons=info.get("time_horizons", [90, 180, 365]),
                    training_date=info.get("training_date"),
                    performance_metrics=info.get("metrics"),
                )
                for name, info in self.model_info.items()
            ]

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Make survival predictions."""
            # Determine which model to use
            model_name = request.model_name
            if model_name is None:
                # Use best model if available
                best_model = self._get_best_model()
                if best_model:
                    model_name = best_model
                else:
                    # Use first available model
                    model_name = next(iter(self.models.keys()))

            if model_name not in self.models:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_name}' not found. Available models: {list(self.models.keys())}",
                )

            model = self.models[model_name]

            try:
                # Convert features to DataFrame
                features_df = pd.DataFrame([request.features])

                # Make predictions
                risk_score = float(model.predict_risk(features_df)[0])

                # Get survival probabilities
                survival_probs = model.predict_survival_function(
                    features_df, request.time_horizons
                )[0]

                survival_probabilities = {
                    f"day_{horizon}": float(prob)
                    for horizon, prob in zip(request.time_horizons, survival_probs, strict=False)
                }

                return PredictionResponse(
                    risk_score=risk_score,
                    survival_probabilities=survival_probabilities,
                    model_used=model_name,
                    feature_importance=self._get_feature_importance(model_name, features_df),
                )

            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Prediction failed: {e!s}"
                ) from e

    def load_models(self) -> None:
        """Load all available models from the models directory."""
        if not self.models_dir.exists():
            raise ValueError(f"Models directory not found: {self.models_dir}")

        # Load configuration if available
        if self.config_path and self.config_path.exists():
            config = load_yaml(self.config_path)
            self._load_config_metadata(config)

        # Find all model directories
        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            model_name = model_dir.name
            pipeline_path = model_dir / "pipeline.joblib"

            if pipeline_path.exists():
                try:
                    pipeline = joblib.load(pipeline_path)
                    model = PipelineModel(pipeline)
                    model.name = model_name
                    self.models[model_name] = model

                    # Load model metadata if available
                    self._load_model_metadata(model_dir, model_name)

                except Exception as e:
                    print(f"Warning: Failed to load model {model_name}: {e}")

        if not self.models:
            raise ValueError(f"No models found in {self.models_dir}")

    def _load_config_metadata(self, config: dict[str, Any]) -> None:
        """Load metadata from configuration file."""
        # Store basic config info that might be useful for API
        pass

    def _load_model_metadata(self, model_dir: Path, model_name: str) -> None:
        """Load model-specific metadata."""
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            try:
                metadata = load_json(metadata_file)
                self.model_info[model_name] = metadata
            except Exception:
                # If metadata loading fails, create basic info
                self.model_info[model_name] = {"name": model_name}

    def _get_best_model(self) -> str | None:
        """Get the name of the best performing model."""
        if not self.models:
            return None

        # Look for best model information in metadata
        best_model = None
        best_score = -1

        for model_name, info in self.model_info.items():
            if "metrics" in info and "concordance" in info["metrics"]:
                score = info["metrics"]["concordance"].get("estimate", 0)
                if score > best_score:
                    best_score = score
                    best_model = model_name

        return best_model

    def _get_feature_importance(
        self, model_name: str, features_df: pd.DataFrame
    ) -> dict[str, float] | None:
        """Get feature importance for the model if available."""
        # This would require SHAP or similar explainability integration
        # For now, return None
        return None

    def _get_version(self) -> str:
        """Get the API version."""
        try:
            from clinical_survival import __version__

            return __version__
        except ImportError:
            return "unknown"

    def run(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        """Run the API server."""
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)


def create_app(models_dir: str | Path, config_path: str | Path | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    server = SurvivalAPIServer(models_dir, config_path)
    server.load_models()
    return server.app


def run_server(
    models_dir: str | Path,
    config_path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """Run the survival ML API server."""
    server = SurvivalAPIServer(models_dir, config_path)
    server.load_models()
    server.run(host, port)
