from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Load model and metadata at startup
model = joblib.load("model.pkl")
metadata = joblib.load("model_metadata.pkl")

app = FastAPI(
    title="Iris Classification API",
    description="Predict iris species from 4 flower measurements",
    version="1.0"
)

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Output schema
class IrisOutput(BaseModel):
    prediction: str
    confidence: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "ML Model API is running 🚀"}

@app.post("/predict", response_model=IrisOutput)
def predict(input_data: IrisInput):
    try:
        # Convert input to array
        features = np.array([[input_data.sepal_length,
                              input_data.sepal_width,
                              input_data.petal_length,
                              input_data.petal_width]])
        
        # Prediction
        prediction_idx = model.predict(features)[0]
        prediction_proba = model.predict_proba(features).max()

        return IrisOutput(
            prediction=metadata["target_names"][prediction_idx],
            confidence=float(prediction_proba)
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "RandomForestClassifier",
        "problem_type": "classification",
        "features": metadata["feature_names"],
        "target_names": list(metadata["target_names"])
    }