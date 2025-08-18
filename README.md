# Iris Flower Classification API 🌸

## Problem
Predict iris species (setosa, versicolor, virginica) based on 4 flower measurements.

## Model
- **RandomForestClassifier** (scikit-learn)
- Trained on built-in Iris dataset
- Accuracy ~95% on test set

## API Endpoints
- `GET /` → Health check
- `POST /predict` → Predict species
- `GET /model-info` → Model metadata

## Example Usage
POST `/predict`
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```
Response:
```json
{
  "prediction": "setosa",
  "confidence": 0.98
}
```

## Run locally
```bash
uvicorn main:app --reload
```

Visit Swagger docs at: [http://localhost:8000/docs](http://localhost:8000/docs)
