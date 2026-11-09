import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sentiment_analysis import load_dataset, predict, preprocess_text, train_and_evaluate

POSITIVE = ["a wonderful and moving film", "brilliant acting and a great story", "i loved every minute, superb",
            "beautiful, funny and heartfelt", "excellent direction, great cast", "a delightful masterpiece"]
NEGATIVE = ["a boring and awful film", "terrible acting and a dull story", "i hated every minute, dreadful",
            "ugly, tedious and pointless", "poor direction, bad cast", "a horrible waste of time"]


def tiny_dataset() -> pd.DataFrame:
    rows = [(t, "positive") for t in POSITIVE * 5] + [(t, "negative") for t in NEGATIVE * 5]
    return pd.DataFrame(rows, columns=["review", "sentiment"])


class SentimentTests(unittest.TestCase):
    def test_preprocess_matches_original_cleaning(self):
        self.assertEqual(preprocess_text("Great!<br /><br />10/10 Loved it").split(), ["great", "loved", "it"])

    def test_trains_and_separates_obvious_reviews(self):
        model, metrics = train_and_evaluate(tiny_dataset())
        self.assertGreaterEqual(metrics["accuracy"], 0.9)
        (pos_label, _), (neg_label, _) = predict(model, ["a great and wonderful story", "an awful, boring mess"])
        self.assertEqual((pos_label, neg_label), ("positive", "negative"))

    def test_rejects_wrong_columns_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            bad_cols = Path(d) / "a.csv"
            pd.DataFrame({"text": ["x"], "label": ["positive"]}).to_csv(bad_cols, index=False)
            with self.assertRaisesRegex(ValueError, "missing columns"):
                load_dataset(bad_cols)
            bad_labels = Path(d) / "b.csv"
            pd.DataFrame({"review": ["x"], "sentiment": ["neutral"]}).to_csv(bad_labels, index=False)
            with self.assertRaisesRegex(ValueError, "neutral"):
                load_dataset(bad_labels)


if __name__ == "__main__":
    unittest.main()
