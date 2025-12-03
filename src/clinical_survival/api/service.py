"""
FastAPI service for clinical survival model inference.

This module provides:
- Real-time survival predictions via REST API
- Health check endpoints for container orchestration
- Readiness probes for load balancer integration
- Prometheus-compatible metrics endpoint
- Request/response logging and timing
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.console import Console

# --- Configuration ---
DEFAULT_MODEL_DIR = Path("results/artifacts/models")
DEFAULT_MODEL_NAME = "rsf.joblib"

console = Console()


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
    """Holds application state including loaded model and health status."""
    
    def __init__(self):
        self.model_pipeline = None
        self.model_path: Optional[str] = None
        self.is_ready: bool = False
        self.startup_time: Optional[datetime] = None
        self.last_error: Optional[str] = None


state = AppState()


# =============================================================================
# Lifespan Management
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.
    
    On startup:
    - Load the model from disk or environment-specified path
    - Initialize metrics
    - Mark service as ready
    
    On shutdown:
    - Clean up resources
    """
    start_time = time.time()
    state.startup_time = datetime.utcnow()
    metrics.startup_time = state.startup_time
    
    # Check for environment variable override
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
            state.is_ready = True
            metrics.model_load_time = time.time() - start_time
            console.print(
                f"✅ Model loaded successfully in {metrics.model_load_time:.2f}s",
                style="bold green"
            )
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


# =============================================================================
# FastAPI Application
# =============================================================================


app = FastAPI(
    title="Clinical Survival Prediction API",
    description="""
An API to serve real-time survival predictions from trained clinical survival models.

## Endpoints

- **GET /health** - Basic health check (is the service running?)
- **GET /ready** - Readiness probe (is the model loaded and ready?)
- **GET /metrics** - Prometheus-compatible metrics
- **POST /predict** - Make survival predictions

## Usage

```bash
# Health check
curl http://localhost:8000/health

# Check if ready
curl http://localhost:8000/ready

# Make a prediction
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"feature_1": 0.5, "feature_2": 1, "feature_3": "A"}'
```
    """,
    version="1.1.0",
    lifespan=lifespan,
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
# Middleware for Metrics
# =============================================================================


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request metrics for all endpoints."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    metrics.record_request(latency_ms, response.status_code)
    
    # Add timing header
    response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
    
    return response


# =============================================================================
# Pydantic Models
# =============================================================================


class InferenceRequest(BaseModel):
    """Defines the structure for a single patient data instance."""
    
    # This schema should ideally be generated from the training features config
    feature_1: float = Field(..., example=0.5, description="Example continuous feature.")
    feature_2: int = Field(..., example=1, description="Example integer feature.")
    feature_3: str = Field(..., example="A", description="Example categorical feature.")

    class Config:
        from_attributes = True


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
# Prediction Endpoints
# =============================================================================


@app.get("/", tags=["Root"])
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Clinical Survival Prediction API.",
        "version": "1.1.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "ready_url": "/ready",
        "metrics_url": "/metrics",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(request: InferenceRequest) -> PredictionResponse:
    """
    Make a survival risk prediction for a patient.
    
    The input features should match the schema expected by the trained model.
    Returns a risk score where higher values indicate higher risk.
    """
    if state.model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure a trained model exists and the API is restarted.",
        )

    start_time = time.time()
    
    # Convert the Pydantic model to a pandas DataFrame for the pipeline
    input_df = pd.DataFrame([request.model_dump()])

    try:
        # The pipeline expects a DataFrame with specific column types/orders.
        risk_score = state.model_pipeline.predict(input_df)
        prediction = float(risk_score[0])
        
        prediction_time_ms = (time.time() - start_time) * 1000
        metrics.record_prediction()

        return PredictionResponse(
            risk_score=prediction,
            prediction_time_ms=prediction_time_ms,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(requests: List[InferenceRequest]) -> List[PredictionResponse]:
    """
    Make predictions for multiple patients at once.
    
    More efficient than calling /predict multiple times for bulk predictions.
    """
    if state.model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded.",
        )
    
    if len(requests) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds maximum of 1000",
        )
    
    start_time = time.time()
    
    # Convert all requests to a single DataFrame
    input_df = pd.DataFrame([r.model_dump() for r in requests])
    
    try:
        risk_scores = state.model_pipeline.predict(input_df)
        prediction_time_ms = (time.time() - start_time) * 1000
        
        # Record one prediction per sample
        for _ in range(len(requests)):
            metrics.record_prediction()
        
        return [
            PredictionResponse(
                risk_score=float(score),
                prediction_time_ms=prediction_time_ms / len(requests),
            )
            for score in risk_scores
        ]
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Batch prediction failed: {str(e)}",
        )
