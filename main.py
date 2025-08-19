from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load trained model, scaler, and metadata
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
metadata = joblib.load("model_metadata.pkl")
target_names = metadata["target_names"]

app = FastAPI(title="Iris ML Model API", description="API for Iris flower prediction")

# Define input schema
class PredictionInput(BaseModel):
    SepalLengthCm: float
    SepalWidthCm: float
    PetalLengthCm: float
    PetalWidthCm: float

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris ML Model API is running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        # Convert input to numpy array
        features = np.array([[input_data.SepalLengthCm,
                              input_data.SepalWidthCm,
                              input_data.PetalLengthCm,
                              input_data.PetalWidthCm]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction_idx = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled).max()

        # Map index back to class name
        prediction_label = target_names[prediction_idx]

        return PredictionOutput(prediction=prediction_label, confidence=prediction_proba)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "Logistic Regression",
        "problem_type": "Classification",
        "features": metadata["feature_names"],
        "target": "Species",
        "classes": target_names
    }
