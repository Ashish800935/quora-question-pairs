"""
Basic unit tests for the feature engineering + preprocessing modules.

Run:
    python -m unittest discover tests
"""
import unittest

from src.preprocessing import preprocess
from src.features import build_all_features, features_for_pair
import pandas as pd


class TestPreprocessing(unittest.TestCase):
    def test_lowercases_and_strips(self):
        self.assertEqual(preprocess("  What IS this?  "), "what is this")

    def test_expands_contractions(self):
        self.assertIn("cannot", preprocess("I can't do this"))

    def test_number_normalization(self):
        self.assertIn("1k", preprocess("I have 1,000 dollars"))


class TestFeatures(unittest.TestCase):
    def test_identical_questions_have_max_overlap(self):
        df = pd.DataFrame([{
            "question1": "what is the capital of india",
            "question2": "what is the capital of india",
        }])
        out = build_all_features(df)
        # word_total = len(q1_words) + len(q2_words) (not the union), so the
        # max possible word_share for identical questions is 0.5, not 1.0.
        self.assertAlmostEqual(out.loc[0, "word_share"], 0.5, places=2)
        self.assertGreater(out.loc[0, "fuzz_ratio"], 99)

    def test_unrelated_questions_have_low_overlap(self):
        df = pd.DataFrame([{
            "question1": "what is the capital of india",
            "question2": "how do i bake a chocolate cake",
        }])
        out = build_all_features(df)
        self.assertLess(out.loc[0, "word_share"], 0.3)

    def test_features_for_pair_shape(self):
        from src.features import ENGINEERED_FEATURE_COLUMNS
        vec = features_for_pair("hello world", "hello there world")
        self.assertEqual(len(vec), len(ENGINEERED_FEATURE_COLUMNS))


if __name__ == "__main__":
    unittest.main()
