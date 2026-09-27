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
   - Fuzzy-matching features — ratio, partial ratio, token-sort ratio, token-set ratio, built from scratch on top of Python's `difflib` instead of pulling in `fuzzywuzzy`
3. **TF-IDF vectorization** of the raw question text
4. **Semantic similarity feature** — cosine similarity between Sentence-Transformer embeddings of the two questions (`--semantic` flag), which turned out to matter a lot (see Results)
5. **Model comparison**: Logistic Regression, Random Forest, and XGBoost, evaluated on accuracy, precision, recall, F1 and log-loss
6. **Deployment**: FastAPI REST endpoint + Streamlit demo UI, both backed by the same `src/predict.py` inference module

## Results

Baseline — just TF-IDF plus the engineered lexical/statistical features, no semantic feature:

| Model               | Accuracy | Precision | Recall | F1     | Log-loss |
|---------------------|---------:|----------:|-------:|-------:|---------:|
| Logistic Regression |   0.7683 |    0.7037 | 0.6520 | 0.6769 |   0.4600 |
| Random Forest       |   0.7418 |    0.6713 | 0.6001 | 0.6337 |   0.4939 |
| XGBoost             |   0.7817 |    0.7154 | 0.6865 | 0.7006 |   0.4234 |

That's roughly where classical BOW/TF-IDF + tree models plateau on this
dataset — it's close to what people were getting on the Kaggle leaderboard
back in 2017 with similar approaches (~79%).

Adding a Sentence-Transformers-based semantic similarity feature (cosine
similarity between question embeddings) pushed every model up noticeably:

| Model (+ semantic feature) | Accuracy   | Precision  | Recall     | F1         | Log-loss   |
|-----------------------------|-----------:|-----------:|-----------:|-----------:|-----------:|
| Logistic Regression          |   0.8333   |    0.7679  | 0.7913     | 0.7794     |   0.3551   |
| Random Forest                |   0.7890   |    0.7248  | 0.6982     | 0.7112     |   0.4658   |
| **XGBoost**                  | **0.8408** | **0.7793** | **0.7985** | **0.7888** | **0.3420** |

That's a ~6 point accuracy jump and a ~19% drop in log-loss from a single
added feature. Makes sense — TF-IDF and word-overlap features only catch
duplicates that share vocabulary ("how do I make money online" vs "ways to
earn money on the internet" barely overlap lexically but are obviously the
same question). Embeddings catch that kind of paraphrase, which pure
lexical features can't.

XGBoost with the semantic feature is the model that's actually saved and
used by the app (picked by log-loss, since that's what the original
Kaggle competition scored on).

All numbers above came from running `python -m src.train` on a 30K-row
sample — should be reproducible with:
```bash
python -m src.train --data data/train.csv --sample 30000 --semantic
```
(numbers will shift a bit depending on sample size and random seed)

## Project structure

```
quora-question-pairs/
├── data/                    # place train.csv here (see data/README.md)
├── notebooks/
│   ├── 01_eda.ipynb                 # exploratory analysis
│   └── 02_model_experiments.ipynb   # feature engineering + model comparison
├── src/
│   ├── preprocessing.py     # text cleaning
│   ├── features.py          # engineered similarity features
│   ├── semantic_features.py # sentence-transformers similarity feature
│   ├── train.py             # training + model comparison + save
│   ├── evaluate.py          # accuracy/precision/recall/F1/log-loss
│   └── predict.py           # single-pair inference, used by app/
├── app/
│   ├── main.py               # FastAPI service
│   ├── schemas.py            # request/response models
│   └── streamlit_app.py      # Streamlit demo UI
├── model/                   # trained model.pkl / tfidf.pkl (committed)
├── tests/
│   └── test_features.py
├── requirements.txt
└── README.md
```

> A trained model is already committed under `model/` so the API/Streamlit
> demo work right after cloning, without having to train anything first.
> Run `python -m src.train` to retrain on your own data — just remember to
> commit the new `model.pkl`/`tfidf.pkl` if you want the change to stick.

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
python -m src.train --data data/train.csv --sample 30000 --semantic
# drop --semantic for a faster, slightly-less-accurate model
# use --sample -1 to train on the full 404K dataset (slower)

# 4. Run tests
python -m unittest discover tests

# 5. Serve the API
uvicorn app.main:app --reload
# -> interactive docs at http://127.0.0.1:8000/docs

# 6. Or run the Streamlit demo
streamlit run app/streamlit_app.py
```

## Notes on a few design choices

- **TF-IDF instead of plain Bag-of-Words** — term frequency alone lets
  common words dominate; TF-IDF weights down words that show up in almost
  every question.
- **Fuzzy-matching reimplemented on `difflib`** instead of installing
  `fuzzywuzzy` — didn't want a dependency I couldn't explain the internals
  of, and difflib's `SequenceMatcher` gets you most of the way there.
- **Log-loss as the main metric**, not just accuracy — it's what the
  original Kaggle competition was actually scored on, and it's more
  informative than accuracy on a dataset that's ~63/37 split.
- **Trained on a 30K sample** rather than all 404K rows, mainly for faster
  iteration while testing features. `--sample -1` runs on the full set.

## Possible extensions

- Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
- Calibrate probabilities (`CalibratedClassifierCV`) since log-loss is
  sensitive to how well-calibrated the predicted probabilities are
- Containerize the FastAPI service with Docker
- Try a fine-tuned cross-encoder instead of static sentence embeddings