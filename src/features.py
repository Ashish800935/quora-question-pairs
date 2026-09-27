"""
Feature engineering for question-pair similarity.

Design choice: fuzzy-matching features are implemented from scratch on
top of Python's built-in `difflib` instead of pulling in `fuzzywuzzy` /
`python-Levenshtein`. This keeps every dependency traceable to a skill
already on the resume (core Python + DSA) instead of adding a library
nobody can speak to in an interview.
"""
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SAFE_DIV = 1e-4
STOP_WORDS = ENGLISH_STOP_WORDS


# ---------------------------------------------------------------------
# 1. Basic features
# ---------------------------------------------------------------------
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Character length, word count, common-word overlap, word share."""
    df = df.copy()
    df['q1_len'] = df['question1'].str.len()
    df['q2_len'] = df['question2'].str.len()
    df['q1_num_words'] = df['question1'].apply(lambda s: len(s.split()))
    df['q2_num_words'] = df['question2'].apply(lambda s: len(s.split()))

    def _common(row):
        w1 = set(row['question1'].split())
        w2 = set(row['question2'].split())
        return len(w1 & w2)

    def _total(row):
        w1 = set(row['question1'].split())
        w2 = set(row['question2'].split())
        return len(w1) + len(w2)

    df['word_common'] = df.apply(_common, axis=1)
    df['word_total'] = df.apply(_total, axis=1)
    df['word_share'] = round(df['word_common'] / (df['word_total'] + SAFE_DIV), 4)
    return df


# ---------------------------------------------------------------------
# 2. Token-based features (stopword-aware overlap)
# ---------------------------------------------------------------------
def _token_features(row) -> list:
    q1_tokens = row['question1'].split()
    q2_tokens = row['question2'].split()

    if not q1_tokens or not q2_tokens:
        return [0.0] * 8

    q1_words = {w for w in q1_tokens if w not in STOP_WORDS}
    q2_words = {w for w in q2_tokens if w not in STOP_WORDS}
    q1_stops = {w for w in q1_tokens if w in STOP_WORDS}
    q2_stops = {w for w in q2_tokens if w in STOP_WORDS}

    common_word_count = len(q1_words & q2_words)
    common_stop_count = len(q1_stops & q2_stops)
    common_token_count = len(set(q1_tokens) & set(q2_tokens))

    cwc_min = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    cwc_max = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    last_word_eq = float(q1_tokens[-1] == q2_tokens[-1])
    first_word_eq = float(q1_tokens[0] == q2_tokens[0])

    return [cwc_min, cwc_max, csc_min, csc_max, ctc_min, ctc_max,
            last_word_eq, first_word_eq]


def add_token_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feats = df.apply(_token_features, axis=1)
    cols = ['cwc_min', 'cwc_max', 'csc_min', 'csc_max',
            'ctc_min', 'ctc_max', 'last_word_eq', 'first_word_eq']
    for i, col in enumerate(cols):
        df[col] = feats.apply(lambda x: x[i])
    return df


# ---------------------------------------------------------------------
# 3. Length-based features
# ---------------------------------------------------------------------
def _longest_common_substring_ratio(q1: str, q2: str) -> float:
    matcher = SequenceMatcher(None, q1, q2)
    match = matcher.find_longest_match(0, len(q1), 0, len(q2))
    return match.size / (min(len(q1), len(q2)) + 1)


def _length_features(row) -> list:
    q1_tokens = row['question1'].split()
    q2_tokens = row['question2'].split()
    if not q1_tokens or not q2_tokens:
        return [0.0, 0.0, 0.0]

    abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
    mean_len = (len(q1_tokens) + len(q2_tokens)) / 2
    longest_substr_ratio = _longest_common_substring_ratio(
        row['question1'], row['question2']
    )
    return [abs_len_diff, mean_len, longest_substr_ratio]


def add_length_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feats = df.apply(_length_features, axis=1)
    df['abs_len_diff'] = feats.apply(lambda x: x[0])
    df['mean_len'] = feats.apply(lambda x: x[1])
    df['longest_substr_ratio'] = feats.apply(lambda x: x[2])
    return df


# ---------------------------------------------------------------------
# 4. Fuzzy-matching features (custom, difflib-based, no extra dependency)
# ---------------------------------------------------------------------
def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100


def _partial_ratio(a: str, b: str) -> float:
    """Best-matching-substring ratio: slide the shorter string over the
    longer one via difflib's matching blocks (approximates fuzzywuzzy's
    partial_ratio without needing the extra library)."""
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    matcher = SequenceMatcher(None, shorter, longer)
    blocks = matcher.get_matching_blocks()
    best = 0.0
    for block in blocks:
        start = max(0, block.b - block.a)
        end = min(len(longer), start + len(shorter))
        segment = longer[start:end]
        score = SequenceMatcher(None, shorter, segment).ratio() * 100
        best = max(best, score)
    return best


def _token_sort_ratio(a: str, b: str) -> float:
    a_sorted = ' '.join(sorted(a.split()))
    b_sorted = ' '.join(sorted(b.split()))
    return _ratio(a_sorted, b_sorted)


def _token_set_ratio(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    intersection = sorted(set_a & set_b)
    diff_ab = sorted(set_a - set_b)
    diff_ba = sorted(set_b - set_a)

    t0 = ' '.join(intersection)
    t1 = ' '.join(intersection + diff_ab)
    t2 = ' '.join(intersection + diff_ba)

    return max(_ratio(t0, t1), _ratio(t0, t2), _ratio(t1, t2))


def _fuzzy_features(row) -> list:
    q1, q2 = row['question1'], row['question2']
    return [
        _ratio(q1, q2),
        _partial_ratio(q1, q2),
        _token_sort_ratio(q1, q2),
        _token_set_ratio(q1, q2),
    ]


def add_fuzzy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feats = df.apply(_fuzzy_features, axis=1)
    cols = ['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio']
    for i, col in enumerate(cols):
        df[col] = feats.apply(lambda x: x[i])
    return df


# ---------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------
ENGINEERED_FEATURE_COLUMNS = [
    'q1_len', 'q2_len', 'q1_num_words', 'q2_num_words',
    'word_common', 'word_total', 'word_share',
    'cwc_min', 'cwc_max', 'csc_min', 'csc_max', 'ctc_min', 'ctc_max',
    'last_word_eq', 'first_word_eq',
    'abs_len_diff', 'mean_len', 'longest_substr_ratio',
    'fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio',
]


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full engineered-feature pipeline (question columns must
    already be preprocessed / cleaned)."""
    df = add_basic_features(df)
    df = add_token_features(df)
    df = add_length_features(df)
    df = add_fuzzy_features(df)
    return df


def features_for_pair(q1: str, q2: str) -> np.ndarray:
    """Build the engineered-feature vector for a single (q1, q2) pair —
    used at inference time in the API / Streamlit app."""
    row_df = pd.DataFrame([{'question1': q1, 'question2': q2}])
    row_df = build_all_features(row_df)
    return row_df[ENGINEERED_FEATURE_COLUMNS].values[0]
