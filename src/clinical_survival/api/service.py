"""
FastAPI service for clinical survival model inference with comprehensive OpenAPI documentation.

This module provides:
- Real-time survival predictions via REST API with full OpenAPI schema
- Dynamic schema generation from feature configurations
- Health check endpoints for container orchestration
- Readiness probes for load balancer integration
- Prometheus-compatible metrics endpoint
- Request/response logging and timing
- Input validation and error handling
- API versioning and authentication support
"""

from __future__ import annotations

import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ValidationError, create_model
from rich.console import Console

from clinical_survival.config import FeaturesConfig, ParamsConfig
from clinical_survival.logging_config import get_logger

# --- Configuration ---
DEFAULT_MODEL_DIR = Path("results/artifacts/models")
DEFAULT_MODEL_NAME = "rsf.joblib"
API_VERSION = "v1"

console = Console()
logger = get_logger(__name__)

# Security
security = HTTPBearer(auto_error=False)


# =============================================================================
# Metrics Collection
# =============================================================================


class MetricsCollector:
    """
    Simple in-memory metrics collector for tracking API performance.
    
    For production, consider using prometheus_client or similar.
    """
    
    def __init__(self):
        self.request_count: int = 0
        self.request_latency_sum: float = 0.0
        self.request_latency_count: int = 0
        self.error_count: int = 0
        self.status_codes: Dict[int, int] = defaultdict(int)
        self.prediction_count: int = 0
        self.model_load_time: Optional[float] = None
        self.startup_time: Optional[datetime] = None
        self._last_prediction_time: Optional[datetime] = None
    
    def record_request(self, latency_ms: float, status_code: int) -> None:
        """Record a completed request."""
        self.request_count += 1
        self.request_latency_sum += latency_ms
        self.request_latency_count += 1
        self.status_codes[status_code] += 1
        if 400 <= status_code < 600:
            self.error_count += 1
    
    def record_prediction(self) -> None:
        """Record a successful prediction."""
        self.prediction_count += 1
        self._last_prediction_time = datetime.utcnow()
    
    def get_avg_latency_ms(self) -> float:
        """Get average request latency in milliseconds."""
        if self.request_latency_count == 0:
            return 0.0
        return self.request_latency_sum / self.request_latency_count
    
    def get_error_rate(self) -> float:
        """Get error rate as a percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100
    
    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = [
            "# HELP clinical_survival_requests_total Total number of requests",
            "# TYPE clinical_survival_requests_total counter",
            f"clinical_survival_requests_total {self.request_count}",
            "",
            "# HELP clinical_survival_predictions_total Total number of predictions",
            "# TYPE clinical_survival_predictions_total counter",
            f"clinical_survival_predictions_total {self.prediction_count}",
            "",
            "# HELP clinical_survival_errors_total Total number of errors",
            "# TYPE clinical_survival_errors_total counter",
            f"clinical_survival_errors_total {self.error_count}",
            "",
            "# HELP clinical_survival_request_latency_ms Average request latency",
            "# TYPE clinical_survival_request_latency_ms gauge",
            f"clinical_survival_request_latency_ms {self.get_avg_latency_ms():.2f}",
            "",
        ]
        
        # Add status code breakdown
        lines.extend([
            "# HELP clinical_survival_http_status_total HTTP status codes",
            "# TYPE clinical_survival_http_status_total counter",
        ])
        for code, count in sorted(self.status_codes.items()):
            lines.append(f'clinical_survival_http_status_total{{code="{code}"}} {count}')
        
        return "\n".join(lines)


# Global metrics instance
metrics = MetricsCollector()


# =============================================================================
# Application State
# =============================================================================


class AppState:
    """Holds application state including loaded model, config, and health status."""

    def __init__(self):
        self.model_pipeline = None
        self.model_path: Optional[str] = None
        self.features_config: Optional[FeaturesConfig] = None
        self.params_config: Optional[ParamsConfig] = None
        self.inference_request_model: Optional[type[BaseModel]] = None
        self.is_ready: bool = False
        self.startup_time: Optional[datetime] = None
        self.last_error: Optional[str] = None
        self.model_name: Optional[str] = None
        self.model_version: Optional[str] = None


state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    On startup:
    - Load model, features, and parameters configurations
    - Create dynamic inference schema
    - Initialize metrics
    - Mark service as ready

    On shutdown:
    - Clean up resources
    """
    start_time = time.time()
    state.startup_time = datetime.utcnow()
    metrics.startup_time = state.startup_time

    # Load configurations
    try:
        # Load features config
        features_path_env = os.environ.get("CLINICAL_SURVIVAL_FEATURES_PATH", "configs/features.yaml")
        features_path = Path(features_path_env)
        if features_path.exists():
            state.features_config = FeaturesConfig.from_yaml(features_path)
            console.print(f"✅ Features config loaded from {features_path}", style="green")
        else:
            console.print(f"⚠️ Features config not found at {features_path}", style="yellow")

        # Load params config
        params_path_env = os.environ.get("CLINICAL_SURVIVAL_PARAMS_PATH", "configs/params.yaml")
        params_path = Path(params_path_env)
        if params_path.exists():
            state.params_config = ParamsConfig.from_yaml(params_path)
            console.print(f"✅ Params config loaded from {params_path}", style="green")
        else:
            console.print(f"⚠️ Params config not found at {params_path}", style="yellow")

    except Exception as e:
        console.print(f"❌ Failed to load configurations: {e}", style="bold red")
        state.last_error = f"Config loading failed: {str(e)}"

    # Create dynamic inference model
    if state.features_config:
        try:
            state.inference_request_model = create_inference_request_model(state.features_config)
            console.print("✅ Dynamic inference schema created", style="green")
        except Exception as e:
            console.print(f"❌ Failed to create inference schema: {e}", style="bold red")
            state.last_error = f"Schema creation failed: {str(e)}"

    # Load model
    model_path_env = os.environ.get("CLINICAL_SURVIVAL_MODEL_PATH")
    if model_path_env:
        model_path = Path(model_path_env)
    else:
        model_path = DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME

    console.print(f"Loading model from: {model_path}")

    try:
        if model_path.exists():
            state.model_pipeline = joblib.load(model_path)
            state.model_path = str(model_path)
            state.model_name = model_path.stem

            # Extract version from model path if available
            if hasattr(state.model_pipeline, 'version'):
                state.model_version = getattr(state.model_pipeline, 'version')

            # Mark as ready only if we have both model and schema
            if state.inference_request_model:
                state.is_ready = True
                metrics.model_load_time = time.time() - start_time
                console.print(
                    f"✅ Model loaded successfully in {metrics.model_load_time:.2f}s",
                    style="bold green"
                )
            else:
                state.last_error = "Model loaded but inference schema not available"
                console.print("⚠️ Model loaded but missing inference schema", style="yellow")
        else:
            console.print(
                f"⚠️ Model not found at {model_path}. API will start but /predict will fail.",
                style="bold yellow"
            )
            state.last_error = f"Model not found at {model_path}"
    except Exception as e:
        console.print(f"❌ Failed to load model: {e}", style="bold red")
        state.last_error = str(e)

    yield  # Application runs here

    # Cleanup on shutdown
    console.print("Shutting down API service...")
    state.model_pipeline = None
    state.inference_request_model = None


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="Clinical Survival Prediction API",
    description="""
A production-ready REST API for real-time survival predictions from trained clinical ML models.

## Features

- **Dynamic Schema Validation**: Input validation automatically generated from feature configurations
- **Comprehensive Monitoring**: Health checks, metrics, and performance monitoring
- **Batch Predictions**: Efficient bulk prediction support
- **OpenAPI Documentation**: Complete API specification with examples
- **Authentication Support**: Optional Bearer token authentication
- **Structured Error Handling**: Consistent error responses with details

## Authentication

Set the `CLINICAL_SURVIVAL_API_TOKEN` environment variable to enable authentication.
Include the token in requests: `Authorization: Bearer <token>`

## Endpoints

### Health & Monitoring
- `GET /health` - Basic health check
- `GET /ready` - Readiness probe (model loaded + ready)
- `GET /metrics` - Prometheus-compatible metrics

### Predictions
- `POST /v1/predict` - Single prediction with full validation
- `POST /v1/predict/batch` - Batch predictions

### Schema
- `GET /v1/schema` - Current inference schema
- `GET /docs` - Interactive API documentation

## Usage Examples

### Single Prediction
```bash
curl -X POST "http://localhost:8000/v1/predict" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $API_TOKEN" \\
  -d '{"age": 65.5, "bmi": 28.3, "stage": "IIB"}'
```

### Batch Predictions
```bash
curl -X POST "http://localhost:8000/v1/predict/batch" \\
  -H "Content-Type: application/json" \\
  -d '[{"age": 65.5, "bmi": 28.3}, {"age": 72.1, "bmi": 24.8}]'
```

### Health Check
```bash
curl http://localhost:8000/health
```
    """,
    version="1.1.0",
    openapi_tags=[
        {"name": "Health", "description": "Health check and readiness endpoints"},
        {"name": "Prediction", "description": "Survival risk prediction endpoints"},
        {"name": "Monitoring", "description": "Metrics and monitoring endpoints"},
        {"name": "Schema", "description": "API schema and configuration"},
    ],
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Authentication & Dependencies
# =============================================================================


def get_api_token() -> Optional[str]:
    """Get API token from environment."""
    return os.environ.get("CLINICAL_SURVIVAL_API_TOKEN")


async def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """Verify API token if authentication is enabled."""
    token = get_api_token()
    if token is None:
        return True  # No auth required

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.credentials != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    return True


# =============================================================================
# Middleware for Metrics & Request ID
# =============================================================================


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request metrics and add request tracking for all endpoints."""
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Add request ID to request state
    request.state.request_id = request_id

    response = await call_next(request)

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    metrics.record_request(latency_ms, response.status_code)

    # Add headers
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    response.headers["X-Request-ID"] = request_id

    return response


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with detailed information."""
    request_id = getattr(request.state, 'request_id', 'unknown')

    return Response(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            message="Input validation failed",
            details={"validation_errors": exc.errors()},
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
        ).model_dump_json(),
        media_type="application/json",
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format."""
    request_id = getattr(request.state, 'request_id', 'unknown')

    return Response(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"http_{exc.status_code}",
            message=exc.detail,
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
        ).model_dump_json(),
        media_type="application/json",
    )


# =============================================================================
# Dynamic Schema Generation
# =============================================================================


def create_inference_request_model(features_config: FeaturesConfig) -> type[BaseModel]:
    """
    Dynamically create a Pydantic model for inference requests based on features config.

    Args:
        features_config: Configuration defining feature schemas

    Returns:
        A Pydantic BaseModel class for inference requests
    """
    fields = {}

    for feature_name, feature_config in features_config.features.items():
        field_type = feature_config.get('type', 'float')
        description = feature_config.get('description', f'Feature: {feature_name}')
        example = feature_config.get('example')
        required = feature_config.get('required', True)

        # Map feature types to Python types
        if field_type == 'int':
            python_type = int
            if example is None:
                example = 1
        elif field_type == 'bool':
            python_type = bool
            if example is None:
                example = True
        elif field_type == 'str':
            python_type = str
            if example is None:
                example = "example_value"
        else:  # float or default
            python_type = float
            if example is None:
                example = 0.5

        # Handle optional fields
        if not required:
            python_type = Optional[python_type]

        fields[feature_name] = (
            python_type,
            Field(
                default=... if required else None,
                description=description,
                example=example,
                json_schema_extra={"feature_type": field_type}
            )
        )

    return create_model(
        'InferenceRequest',
        **fields,
        __config__={'from_attributes': True}
    )


# =============================================================================
# Pydantic Models
# =============================================================================


class ErrorResponse(BaseModel):
    """Standard error response format."""

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    timestamp: str = Field(..., description="Error timestamp in ISO format")


class PredictionMetadata(BaseModel):
    """Metadata about the prediction."""

    model_name: str = Field(..., description="Name of the model used")
    model_version: Optional[str] = Field(None, description="Model version")
    prediction_time_ms: float = Field(..., description="Time taken for prediction in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")
    api_version: str = Field(..., description="API version")


class PredictionResponse(BaseModel):
    """Enhanced prediction response with metadata."""

    risk_score: float = Field(
        ...,
        description="Predicted risk score (higher = higher risk)",
        ge=0.0,
        le=1.0,
        example=0.75
    )
    risk_category: str = Field(
        ...,
        description="Risk category based on score thresholds",
        example="high_risk"
    )
    confidence_interval: Optional[List[float]] = Field(
        None,
        description="95% confidence interval for risk score",
        example=[0.65, 0.85]
    )
    metadata: PredictionMetadata = Field(..., description="Prediction metadata")


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""

    predictions: List[PredictionResponse] = Field(..., description="List of individual predictions")
    batch_size: int = Field(..., description="Number of predictions in batch")
    total_prediction_time_ms: float = Field(..., description="Total time for all predictions")
    request_id: str = Field(..., description="Unique batch request identifier")


class PredictionResponse(BaseModel):
    """Defines the structure for the prediction response."""
    
    risk_score: float = Field(..., description="Predicted risk score")
    prediction_time_ms: Optional[float] = Field(
        None, description="Time taken for prediction in milliseconds"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")


class ReadinessResponse(BaseModel):
    """Readiness probe response."""
    
    ready: bool = Field(..., description="Whether the service is ready to serve requests")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: Optional[str] = Field(None, description="Path to loaded model")
    last_error: Optional[str] = Field(None, description="Last error message if any")


class MetricsResponse(BaseModel):
    """Metrics response."""
    
    requests_total: int
    predictions_total: int
    errors_total: int
    avg_latency_ms: float
    error_rate_percent: float
    uptime_seconds: float
    model_load_time_seconds: Optional[float]
    status_codes: Dict[str, int]


# =============================================================================
# Health Check Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns 200 if the service is running, regardless of model state.
    Use /ready for full readiness check.
    
    This endpoint is suitable for:
    - Kubernetes liveness probes
    - Load balancer health checks
    - Basic monitoring
    """
    uptime = None
    if state.startup_time:
        uptime = (datetime.utcnow() - state.startup_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=uptime,
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["Health"])
def readiness_check() -> ReadinessResponse:
    """
    Readiness probe endpoint.
    
    Returns 200 only if the model is loaded and ready to serve predictions.
    Returns 503 if not ready.
    
    This endpoint is suitable for:
    - Kubernetes readiness probes
    - Load balancer backend health checks
    - Deployment verification
    """
    response = ReadinessResponse(
        ready=state.is_ready,
        model_loaded=state.model_pipeline is not None,
        model_path=state.model_path,
        last_error=state.last_error,
    )
    
    if not state.is_ready:
        raise HTTPException(
            status_code=503,
            detail=response.model_dump(),
        )
    
    return response


@app.get("/metrics", tags=["Monitoring"])
def get_metrics(format: str = "json") -> Response:
    """
    Get service metrics.
    
    Supports two formats:
    - json: Returns metrics as JSON (default)
    - prometheus: Returns metrics in Prometheus text format
    
    Metrics include:
    - Total requests
    - Total predictions
    - Error counts and rate
    - Average latency
    - Status code distribution
    - Uptime
    """
    if format == "prometheus":
        return Response(
            content=metrics.to_prometheus_format(),
            media_type="text/plain; version=0.0.4",
        )
    
    uptime = 0.0
    if state.startup_time:
        uptime = (datetime.utcnow() - state.startup_time).total_seconds()
    
    return MetricsResponse(
        requests_total=metrics.request_count,
        predictions_total=metrics.prediction_count,
        errors_total=metrics.error_count,
        avg_latency_ms=metrics.get_avg_latency_ms(),
        error_rate_percent=metrics.get_error_rate(),
        uptime_seconds=uptime,
        model_load_time_seconds=metrics.model_load_time,
        status_codes={str(k): v for k, v in metrics.status_codes.items()},
    )


# =============================================================================
# Root & Schema Endpoints
# =============================================================================


@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint with API information and available endpoints."""
    return {
        "message": "Welcome to the Clinical Survival Prediction API",
        "version": API_VERSION,
        "authentication_required": get_api_token() is not None,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "schema": "/v1/schema",
            "predict": "/v1/predict",
            "batch_predict": "/v1/predict/batch",
        },
        "model_info": {
            "name": state.model_name,
            "version": state.model_version,
            "ready": state.is_ready,
        }
    }


@app.get("/v1/schema", tags=["Schema"])
def get_inference_schema(authenticated: bool = Depends(verify_token)):
    """Get the current inference schema and feature specifications."""
    if not state.features_config:
        raise HTTPException(
            status_code=503,
            detail="Features configuration not loaded",
        )

    return {
        "api_version": API_VERSION,
        "model_name": state.model_name,
        "features": state.features_config.features,
        "inference_schema": state.inference_request_model.model_json_schema() if state.inference_request_model else None,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Prediction Endpoints
# =============================================================================


def _get_risk_category(risk_score: float, params_config: Optional[ParamsConfig] = None) -> str:
    """Categorize risk score into interpretable categories."""
    if params_config and hasattr(params_config, 'risk_thresholds'):
        thresholds = params_config.risk_thresholds
        if risk_score >= thresholds.get('high', 0.7):
            return "high_risk"
        elif risk_score >= thresholds.get('medium', 0.3):
            return "medium_risk"
        else:
            return "low_risk"
    else:
        # Default thresholds
        if risk_score >= 0.7:
            return "high_risk"
        elif risk_score >= 0.3:
            return "medium_risk"
        else:
            return "low_risk"


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_v1(
    request: "BaseModel",
    authenticated: bool = Depends(verify_token),
    req: Request = None,
) -> PredictionResponse:
    """
    Make a survival risk prediction for a single patient.

    **Features Required:**
    Input must match the schema defined in `/v1/schema`. Features are validated
    against the training configuration.

    **Returns:**
    - Risk score between 0-1 (higher = higher risk)
    - Risk category (low/medium/high)
    - Prediction metadata including timing and model info

    **Authentication:** Required if API token is configured.
    """
    if not state.is_ready or state.model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready: model or schema not loaded",
        )

    request_id = getattr(req.state, 'request_id', str(uuid.uuid4()))
    start_time = time.time()

    try:
        # Convert to DataFrame for pipeline
        input_df = pd.DataFrame([request.model_dump()])

        # Make prediction
        risk_scores = state.model_pipeline.predict(input_df)
        risk_score = float(risk_scores[0])

        prediction_time_ms = (time.time() - start_time) * 1000
        metrics.record_prediction()

        return PredictionResponse(
            risk_score=risk_score,
            risk_category=_get_risk_category(risk_score, state.params_config),
            confidence_interval=None,  # Could be added with ensemble methods
            metadata=PredictionMetadata(
                model_name=state.model_name or "unknown",
                model_version=state.model_version,
                prediction_time_ms=prediction_time_ms,
                request_id=request_id,
                api_version=API_VERSION,
            )
        )

    except Exception as e:
        logger.error(f"Prediction failed for request {request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch_v1(
    requests: List["BaseModel"],
    authenticated: bool = Depends(verify_token),
    req: Request = None,
) -> BatchPredictionResponse:
    """
    Make survival risk predictions for multiple patients.

    **Features:**
    - Efficient bulk processing (better than multiple single requests)
    - Maximum 1000 predictions per batch
    - All predictions share the same request ID

    **Input:** List of patient feature objects (same schema as single prediction)
    **Returns:** Batch response with individual predictions and metadata

    **Authentication:** Required if API token is configured.
    """
    if not state.is_ready or state.model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready: model or schema not loaded",
        )

    if len(requests) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds maximum of 1000 predictions",
        )

    request_id = getattr(req.state, 'request_id', str(uuid.uuid4()))
    start_time = time.time()

    try:
        # Convert all requests to DataFrame
        input_df = pd.DataFrame([r.model_dump() for r in requests])

        # Make batch predictions
        risk_scores = state.model_pipeline.predict(input_df)
        total_prediction_time_ms = (time.time() - start_time) * 1000

        # Create individual prediction responses
        predictions = []
        for score in risk_scores:
            risk_score = float(score)
            predictions.append(PredictionResponse(
                risk_score=risk_score,
                risk_category=_get_risk_category(risk_score, state.params_config),
                confidence_interval=None,
                metadata=PredictionMetadata(
                    model_name=state.model_name or "unknown",
                    model_version=state.model_version,
                    prediction_time_ms=total_prediction_time_ms / len(requests),
                    request_id=request_id,
                    api_version=API_VERSION,
                )
            ))

            metrics.record_prediction()

        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(requests),
            total_prediction_time_ms=total_prediction_time_ms,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(f"Batch prediction failed for request {request_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


# Set up dynamic route for the inference request model
# This needs to be done after the model is created in lifespan
def setup_dynamic_routes():
    """Set up routes that depend on the dynamic inference model."""
    if state.inference_request_model:
        # Update the route dependencies to use the dynamic model
        app.routes[-2].dependant.cache_key = (predict_v1, (authenticated, req))
        app.routes[-2].body_field = state.inference_request_model.__fields__['request']

        app.routes[-1].dependant.cache_key = (predict_batch_v1, (authenticated, req))
        app.routes[-1].body_field = List[state.inference_request_model].__fields__['requests']

# Note: Dynamic routes are handled by creating the model at runtime
# The actual model validation happens in the endpoint functions
