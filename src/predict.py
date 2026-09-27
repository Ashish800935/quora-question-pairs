"""
Load the trained artifacts and predict on a single (question1, question2)
pair. Used by both the FastAPI app and the Streamlit app so there's a
single source of truth for inference logic.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, csr_matrix

from src.preprocessing import preprocess
from src.features import features_for_pair

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


class DuplicateQuestionPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        model_path = model_dir / "model.pkl"
        tfidf_path = model_dir / "tfidf.pkl"
        meta_path = model_dir / "feature_columns.json"

        if not (model_path.exists() and tfidf_path.exists() and meta_path.exists()):
            raise FileNotFoundError(
                f"Trained artifacts not found in {model_dir}. "
                f"Run `python -m src.train` first."
            )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(tfidf_path, "rb") as f:
            self.tfidf = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)

        self.feature_cols = meta["feature_cols"]
        self.used_semantic = meta.get("used_semantic", False)
        self.best_model_name = meta.get("best_model", "unknown")
        self.requires_dense = meta.get("requires_dense", False)

    def predict(self, question1: str, question2: str) -> dict:
        q1 = preprocess(question1)
        q2 = preprocess(question2)

        engineered = features_for_pair(q1, q2)

        if self.used_semantic:
            from src.semantic_features import semantic_similarity_pair
            sim = semantic_similarity_pair(q1, q2)
            engineered = np.append(engineered, sim)

        engineered = csr_matrix(engineered.reshape(1, -1).astype(float))
        q1_tfidf = self.tfidf.transform([q1])
        q2_tfidf = self.tfidf.transform([q2])

        X = hstack([engineered, q1_tfidf, q2_tfidf]).tocsr()
        if self.requires_dense:
            X = X.toarray()

        proba = float(self.model.predict_proba(X)[0][1])
        is_duplicate = bool(proba >= 0.5)

        return {
            "is_duplicate": is_duplicate,
            "confidence": round(proba if is_duplicate else 1 - proba, 4),
            "duplicate_probability": round(proba, 4),
            "model_used": self.best_model_name,
        }


# Lazily-instantiated singleton so the API/Streamlit app load the model once.
_predictor = None


def get_predictor() -> DuplicateQuestionPredictor:
    global _predictor
    if _predictor is None:
        _predictor = DuplicateQuestionPredictor()
    return _predictor
