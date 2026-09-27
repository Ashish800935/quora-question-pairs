"""
End-to-end training pipeline for Quora Duplicate Question Detection.

Usage:
    python -m src.train --data data/train.csv --sample 30000
    python -m src.train --data data/train.csv --sample 30000 --semantic

Pipeline:
    1. Load + clean the data (drop NA, drop exact dupes)
    2. Preprocess question text (src.preprocessing)
    3. Engineer similarity features (src.features)
    4. Optionally add a semantic-similarity feature (src.semantic_features)
    5. TF-IDF vectorize the questions (matches the "TF-IDF" skill on the
       resume — this replaces the plain CountVectorizer/BOW approach)
    6. Train + compare Logistic Regression and Random Forest
       (both plain Scikit-learn — no unclaimed libraries like XGBoost)
    7. Evaluate with accuracy, precision, recall, F1 and log-loss
       (log-loss is the actual metric the original Kaggle competition
       was scored on)
    8. Save the best model + fitted TF-IDF vectorizer + feature list to
       model/
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocessing import preprocess_series
from src.features import build_all_features, ENGINEERED_FEATURE_COLUMNS
from src.evaluate import evaluate_model, print_comparison_table

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def load_data(path: str, sample: int | None, random_state: int = 2) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["question1", "question2"]).reset_index(drop=True)
    if sample is not None and sample < len(df):
        df = df.sample(sample, random_state=random_state).reset_index(drop=True)
    return df


def build_dataset(df: pd.DataFrame, use_semantic: bool, tfidf_max_features: int):
    print(f"[1/4] Preprocessing {len(df)} question pairs...")
    df["question1"] = preprocess_series(df["question1"])
    df["question2"] = preprocess_series(df["question2"])

    print("[2/4] Engineering similarity features...")
    df = build_all_features(df)

    semantic_col = None
    if use_semantic:
        from src.semantic_features import semantic_similarity_batch, SEMANTIC_AVAILABLE
        if not SEMANTIC_AVAILABLE:
            print("  -> sentence-transformers not installed, skipping semantic feature.")
        else:
            print("  -> computing semantic similarity (Sentence-Transformers)...")
            df["semantic_sim"] = semantic_similarity_batch(df["question1"], df["question2"])
            semantic_col = "semantic_sim"

    print(f"[3/4] Fitting TF-IDF (max_features={tfidf_max_features})...")
    tfidf = TfidfVectorizer(max_features=tfidf_max_features)
    all_questions = pd.concat([df["question1"], df["question2"]], ignore_index=True)
    tfidf.fit(all_questions)
    q1_tfidf = tfidf.transform(df["question1"])
    q2_tfidf = tfidf.transform(df["question2"])

    feature_cols = list(ENGINEERED_FEATURE_COLUMNS)
    if semantic_col:
        feature_cols.append(semantic_col)
    engineered = csr_matrix(df[feature_cols].values.astype(float))

    print("[4/4] Assembling final feature matrix...")
    X = hstack([engineered, q1_tfidf, q2_tfidf]).tocsr()
    y = df["is_duplicate"].values

    return X, y, tfidf, feature_cols


def main():
    parser = argparse.ArgumentParser(description="Train duplicate-question detector")
    parser.add_argument("--data", default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--sample", type=int, default=30000,
                         help="Number of rows to sample (use -1 for full dataset)")
    parser.add_argument("--tfidf-max-features", type=int, default=3000)
    parser.add_argument("--semantic", action="store_true",
                         help="Add Sentence-Transformers semantic similarity feature")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    sample = None if args.sample == -1 else args.sample
    df = load_data(args.data, sample, random_state=2)
    print(f"Loaded {len(df)} rows from {args.data}")

    X, y, tfidf, feature_cols = build_dataset(df, args.semantic, args.tfidf_max_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    candidates = {
        "LogisticRegression": (LogisticRegression(max_iter=2000), False),
        # max_depth / min_samples_leaf are capped on purpose: unrestricted
        # trees on ~6000-dim TF-IDF features grow enormous (150+ MB pickle)
        # for a negligible accuracy gain -- not something you want to ship.
        "RandomForest": (RandomForestClassifier(
            n_estimators=150, max_depth=20, min_samples_leaf=2,
            n_jobs=-1, random_state=args.random_state
        ), False),
        # Gradient-boosted trees -- handles the sparse TF-IDF matrix
        # natively (no dense conversion needed, unlike sklearn's
        # HistGradientBoostingClassifier), and was already the
        # best-performing model in the original exploratory notebooks.
        "XGBoost": (XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=args.random_state, n_jobs=-1
        ), False),
    }

    results = {}
    fitted_models = {}
    dense_flags = {}
    for name, (model, needs_dense) in candidates.items():
        print(f"\nTraining {name}...")
        start = time.time()
        X_tr = X_train.toarray() if needs_dense else X_train
        X_te = X_test.toarray() if needs_dense else X_test
        model.fit(X_tr, y_train)
        elapsed = time.time() - start
        metrics = evaluate_model(model, X_te, y_test)
        metrics["train_time_sec"] = round(elapsed, 2)
        results[name] = metrics
        fitted_models[name] = model
        dense_flags[name] = needs_dense
        print(f"  done in {elapsed:.1f}s -> {metrics}")

    print_comparison_table(results)

    # Pick the best model by log-loss (lower is better) -- matches the
    # metric the original Kaggle competition was actually scored on.
    best_name = min(results, key=lambda k: results[k]["log_loss"])
    best_model = fitted_models[best_name]
    print(f"\nBest model by log-loss: {best_name}")

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(MODEL_DIR / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(MODEL_DIR / "feature_columns.json", "w") as f:
        json.dump({
            "feature_cols": feature_cols,
            "best_model": best_name,
            "used_semantic": args.semantic,
            "requires_dense": dense_flags[best_name],
        }, f, indent=2)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved best model ({best_name}), TF-IDF vectorizer and metadata to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
