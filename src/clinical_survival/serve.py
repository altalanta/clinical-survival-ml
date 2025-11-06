from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import uvicorn
from typing import Dict, Any, List

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    time_horizons: List[int]

app = FastAPI(
    title="Clinical Survival ML API",
    description="API for making survival predictions.",
    version="0.1.0",
)

model = None

@app.on_event("startup")
def load_model():
    global model
    # This model URI will be passed in when the server is started.
    # It defaults to a placeholder.
    model_uri = "models:/placeholder/production"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # In a real application, you might want to prevent startup
        # if the model doesn't load.
        model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Clinical Survival ML API"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
    try:
        # The model is expected to handle the feature dictionary directly.
        # This might require a custom pyfunc model wrapper in MLflow.
        # For now, we convert to a DataFrame.
        feature_df = pd.DataFrame([request.features])
        
        # This is a placeholder for the actual prediction logic.
        # A real implementation would need to pass time_horizons to the model.
        predictions = model.predict(feature_df)
        
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

def run_server(model_uri: str, host: str = "127.0.0.1", port: int = 8000):
    """Starts the FastAPI server with the specified model."""
    global model
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        print(f"Successfully loaded model from {model_uri}")
    except Exception as e:
        print(f"FATAL: Could not load model from {model_uri}. Error: {e}")
        return
        
    uvicorn.run(app, host=host, port=port)

