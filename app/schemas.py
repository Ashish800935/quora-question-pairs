"""Pydantic request/response models (matches the "Pydantic Structured
Output" skill already used in the RAG project)."""
from pydantic import BaseModel, Field


class QuestionPairRequest(BaseModel):
    question1: str = Field(..., min_length=1, examples=["How do I learn Python?"])
    question2: str = Field(..., min_length=1, examples=["What is the best way to learn Python?"])


class PredictionResponse(BaseModel):
    is_duplicate: bool
    confidence: float = Field(..., ge=0, le=1, description="Model's confidence in the predicted label")
    duplicate_probability: float = Field(..., ge=0, le=1, description="Raw P(is_duplicate=1)")
    model_used: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
