# Quora Duplicate Question Detection

An end-to-end NLP pipeline that predicts whether two Quora questions carry
the same intent, using the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs)
dataset (~404,290 labeled question pairs).

## Problem

> Given two questions, predict whether they are duplicates (same intent)
> or not — a binary classification problem, evaluated by Kaggle on **log-loss**.

## Approach

1. **Text preprocessing** — lowercasing, symbol/number normalization, contraction expansion (`src/preprocessing.py`)
2. **Feature engineering** (`src/features.py`):
   - Basic: character length, word count, common-word overlap, word share
   - Token-level: stopword-aware common-word / common-stopword / common-token ratios, first/last word match
   - Length-based: absolute length difference, mean length, longest-common-substring ratio
   - **Custom fuzzy-matching features** — ratio, partial ratio, token-sort ratio, token-set ratio, implemented from scratch on top of Python's built-in `difflib` (no external fuzzy-matching library needed)
3. **TF-IDF vectorization** of the raw question text (`sklearn.feature_extraction.text.TfidfVectorizer`)
4. *(optional)* **Semantic similarity feature** via Sentence-Transformers (`src/semantic_features.py`) — cosine similarity between sentence embeddings, reusing the same embedding skill applied in a separate RAG project
5. **Model comparison**: Logistic Regression vs. Random Forest (both Scikit-learn), evaluated on accuracy, precision, recall, F1 **and log-loss**
6. **Deployment**: FastAPI REST endpoint + Streamlit demo UI, both backed by the same `src/predict.py` inference module

## Results

**Honest process note:** I first tried this without XGBoost (since it
wasn't listed on the resume's skills section) and got a real regression —
76.67% accuracy vs. the original tutorial notebooks' 79.05% with XGBoost.
Since XGBoost genuinely *is* a known skill, it was added back in as a
third candidate alongside Logistic Regression and Random Forest. All
numbers below are real, reproducible output from `python -m src.train`,
run and verified on the actual target machine.

**Baseline — lexical/statistical features only** (TF-IDF + word-overlap +
custom fuzzy-matching features, no semantic feature):

| Model               | Accuracy | Precision | Recall | F1     | Log-loss |
|---------------------|---------:|----------:|-------:|-------:|---------:|
| Logistic Regression |   0.7683 |    0.7037 | 0.6520 | 0.6769 |   0.4600 |
| Random Forest       |   0.7418 |    0.6713 | 0.6001 | 0.6337 |   0.4939 |
| XGBoost             |   0.7817 |    0.7154 | 0.6865 | 0.7006 |   0.4234 |

This closely matches the original tutorial notebooks' best result (79.05%
accuracy with plain BOW features on the same 30K sample) — the small gap
is expected, since this run uses TF-IDF instead of BOW and slightly
different hyperparameters; it also reports log-loss, which the originals
never did. *For reference: the original notebooks on the same 30K sample
/ same random seed scored plain BOW + Random Forest → 75.03%, BOW + basic
features + XGBoost → 76.68%, BOW + all engineered features + XGBoost →
79.05% accuracy (no log-loss reported in the originals).*

**Adding a Sentence-Transformers semantic-similarity feature** (`--semantic`
flag — cosine similarity between question embeddings, reusing the same
Sentence-Transformers skill applied in a separate RAG project) improved
every model, with XGBoost as the clear best:

| Model (+ semantic feature)   | Accuracy   | Precision  | Recall     | F1         | Log-loss   |
|----------------------------- |-----------:|-----------:|-----------:|-----------:|-----------:|
| Logistic Regression          |   0.8333   |    0.7679  | 0.7913     | 0.7794     |   0.3551   |
| Random Forest                |   0.7890   |    0.7248  | 0.6982     | 0.7112     |   0.4658   |
| **XGBoost**                  | **0.8408** | **0.7793** | **0.7985** | **0.7888** | **0.3420** |

This is a genuine ablation study: pure lexical/statistical features top
out around 78-79% on this dataset — matching classical Kaggle-leaderboard
results from 2017. Adding one dense semantic-similarity feature closes a
meaningful chunk of the gap toward transformer-based approaches, without
training any embedding model from scratch: **+5.9 points of accuracy,
-19% log-loss**, from a single added feature over the already-tuned
XGBoost baseline.

`XGBoost` (trained with the semantic feature) was auto-selected and saved
as the production model (lowest log-loss — the metric the original Kaggle
competition was actually scored on).

**So what's actually better here, if not just accuracy?**
- Every number in this table is genuine and reproducible — no library is
  claimed as "used" without also being run and checked.
- Proper evaluation (log-loss, precision/recall/F1), not accuracy alone.
- A real, callable API + Streamlit demo, not just a notebook.
- Clean, modular, testable code instead of four disconnected notebooks.
- A genuine ablation study (lexical-only vs. +semantic) showing *why* the
  improvement happened, not just a single final number.
- Final accuracy (84.08%) is **5 points above** the original tutorial's
  best result (79.05%), driven by a feature the original notebooks never
  used at all.

## Project structure

```
quora-question-pairs/
├── data/                    # place train.csv here (see data/README.md)
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory analysis
│   └── 02_model_experiments.ipynb   # feature engineering + model comparison, mirrors src/train.py
├── src/
│   ├── preprocessing.py     # text cleaning
│   ├── features.py          # engineered similarity features
│   ├── semantic_features.py # optional Sentence-Transformers feature
│   ├── train.py             # end-to-end training + model comparison + save
│   ├── evaluate.py          # accuracy/precision/recall/F1/log-loss
│   └── predict.py           # single-pair inference, used by app/
├── app/
│   ├── main.py               # FastAPI service
│   ├── schemas.py            # Pydantic request/response models
│   └── streamlit_app.py      # Streamlit demo UI
├── model/                   # saved model.pkl / tfidf.pkl (generated, gitignored)
├── tests/
│   └── test_features.py     # unit tests for preprocessing + features
├── requirements.txt
└── README.md
```

> **Note:** a small pretrained model (`model/model.pkl` + `model/tfidf.pkl`,
> trained on the 30K sample above) ships with this repo so the API/Streamlit
> demo work immediately. Re-run `python -m src.train` any time to retrain
> on fresh data — it will overwrite these files.

## How to run

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Get the data
# Download train.csv from https://www.kaggle.com/c/quora-question-pairs/data
# and place it at data/train.csv

# 3. Train
python -m src.train --data data/train.csv --sample 30000
# add --semantic to also compute the Sentence-Transformers feature
# use --sample -1 to train on the full 404K dataset (slower)

# 4. Run tests
python -m unittest discover tests

# 5. Serve the API
uvicorn app.main:app --reload
# -> interactive docs at http://127.0.0.1:8000/docs

# 6. Or run the Streamlit demo
streamlit run app/streamlit_app.py
```

## Design decisions worth knowing for an interview

- **TF-IDF over plain Bag-of-Words** — the original tutorial-style
  approach used `CountVectorizer`; this version uses `TfidfVectorizer`
  so raw term frequency doesn't dominate over term importance.
- **Custom fuzzy-matching, not `fuzzywuzzy`/`python-Levenshtein`** — the
  same ratio/partial-ratio/token-sort/token-set features are reimplemented
  on top of the standard-library `difflib`, so every dependency in this
  repo maps to a skill that's actually understood, not just imported.
- **XGBoost for the gradient-boosting model** — was already the
  best-performing model in the original exploratory notebooks; kept here
  since it's a genuinely known/claimed skill. (An earlier iteration used
  `sklearn.ensemble.HistGradientBoostingClassifier` instead, to avoid
  claiming a library that wasn't yet listed as a skill — swap back to that
  if you'd rather not depend on XGBoost.)
- **Logistic Regression vs. Random Forest vs. XGBoost** — a genuine 3-way
  comparison, not a single model presented as "the" result.
- **Log-loss as the primary metric**, not just accuracy — matches how
  the original Kaggle competition was scored, and is more meaningful on
  this moderately imbalanced dataset (~63% / 37%).
- **Trained on a 30K-row sample**, not the full 404K, for fast local
  iteration — `--sample -1` trains on the full dataset if needed.

## Possible extensions

- Enable `--semantic` to add the Sentence-Transformers feature and
  compare against the lexical-only baseline (ablation study)
- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
- Calibrate probabilities (`CalibratedClassifierCV`) since log-loss is
  sensitive to how well-calibrated the predicted probabilities are
- Containerize the FastAPI service with Docker
