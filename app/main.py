"""
FastAPI service for duplicate-question detection.

Run:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""
from fastapi import FastAPI, HTTPException

from app.schemas import QuestionPairRequest, PredictionResponse, HealthResponse
from src.predict import get_predictor

app = FastAPI(
    title="Quora Duplicate Question Detector",
    description="Predicts whether two questions carry the same intent.",
    version="1.0.0",
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        get_predictor()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="model not trained yet", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: QuestionPairRequest):
    try:
        predictor = get_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = predictor.predict(payload.question1, payload.question2)
    return PredictionResponse(**result)
