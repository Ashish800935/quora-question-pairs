# Data

This project uses the [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset from Kaggle (~404,290 labeled question pairs).

The raw CSV is **not committed to the repo** (60+ MB, and it's Kaggle's data, not ours to redistribute).

## Setup

1. Download `train.csv` from: https://www.kaggle.com/c/quora-question-pairs/data
2. Place it here as `data/train.csv`
3. Run training:
   ```bash
   python -m src.train --data data/train.csv --sample 30000
   ```

Columns: `id, qid1, qid2, question1, question2, is_duplicate`
