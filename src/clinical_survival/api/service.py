from pathlib import Path
from typing import List

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rich.console import Console

# --- Configuration ---
DEFAULT_MODEL_DIR = Path("results/artifacts/models")
DEFAULT_MODEL_NAME = "rsf.joblib"  # Default model to load

# --- Application State ---
app = FastAPI(
    title="Clinical Survival Prediction API",
    description="An API to serve real-time survival predictions.",
    version="1.0.0",
)
console = Console()

# A global variable to hold the loaded model pipeline.
# This is populated during the 'startup' event.
model_pipeline = None


# --- Pydantic Models for Request Validation ---
class InferenceRequest(BaseModel):
    """Defines the structure for a single patient data instance."""

    # This schema should ideally be generated from the training features config
    # For now, we define a placeholder schema.
    feature_1: float = Field(..., example=0.5, description="Example continuous feature.")
    feature_2: int = Field(..., example=1, description="Example integer feature.")
    feature_3: str = Field(..., example="A", description="Example categorical feature.")

    class Config:
        # Pydantic v2 requires this for from_orm mode if you use it
        from_attributes = True


class PredictionResponse(BaseModel):
    """Defines the structure for the prediction response."""

    risk_score: float = Field(
        ..., example=0.85, description="The predicted risk score from the survival model."
    )


# --- API Events ---
@app.on_event("startup")
def load_model():
    """Load the model pipeline at application startup."""
    global model_pipeline

    # Check for a model path override from the environment variable (set by CLI)
    env_model_path = os.environ.get("CLINICAL_SURVIVAL_MODEL_PATH")

    if env_model_path:
        model_path = Path(env_model_path)
    else:
        model_path = DEFAULT_MODEL_DIR / DEFAULT_MODEL_NAME

    console.print(f"Attempting to load model from: {model_path}", style="cyan")
    if not model_path.exists():
        console.print(
            f"[bold red]Error: Model file not found at '{model_path}'.[/bold red]"
        )
        console.print(
            "Please run the training pipeline first (`clinical-ml training run`)."
        )
        # We don't raise an exception here to allow the API to start,
        # but the /predict endpoint will fail until a model is present.
        model_pipeline = None
    else:
        model_pipeline = joblib.load(model_path)
        console.print("✅ Model loaded successfully.", style="bold green")


# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint providing a welcome message."""
    return {"message": "Welcome to the Clinical Survival Prediction API."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: InferenceRequest):
    """
    Accepts patient data and returns a survival risk prediction.
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure a trained model exists and the API is restarted.",
        )

    # Convert the Pydantic model to a pandas DataFrame for the pipeline
    input_df = pd.DataFrame([request.model_dump()])

    try:
        # The pipeline expects a DataFrame with specific column types/orders.
        # The preprocessor inside the pipeline should handle this.
        risk_score = model_pipeline.predict(input_df)

        # Ensure the output is a standard float, not a numpy type
        prediction = float(risk_score[0])

        return PredictionResponse(risk_score=prediction)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Prediction failed: {str(e)}"
        )
