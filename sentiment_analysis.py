"""IMDB movie review sentiment classifier: TF-IDF features + logistic regression.

    python sentiment_analysis.py "IMDB Dataset.csv"
    python sentiment_analysis.py "IMDB Dataset.csv" --predict "A beautiful, moving film."
"""

import argparse
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = {"negative": 0, "positive": 1}


def preprocess_text(text: str) -> str:
    """Remove <br /> tags and anything that is not a letter, then lowercase."""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"review", "sentiment"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    df = df.dropna(subset=["review", "sentiment"])
    unknown = set(df["sentiment"]) - set(LABELS)
    if unknown:
        raise ValueError(f"unexpected sentiment labels: {sorted(unknown)}")
    return df


def build_model(max_features: int = 5000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, preprocessor=preprocess_text)),
        ("clf", LogisticRegression(max_iter=200)),
    ])


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df["review"]
    y = df["sentiment"].map(LABELS)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    model = build_model().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return model, metrics


def predict(model: Pipeline, texts: list[str]) -> list[tuple[str, float]]:
    """Return (label, probability of positive) for each text."""
    probs = model.predict_proba(texts)[:, 1]
    return [("positive" if p >= 0.5 else "negative", float(p)) for p in probs]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate an IMDB sentiment classifier.")
    parser.add_argument("csv", help="path to 'IMDB Dataset.csv' (columns: review, sentiment)")
    parser.add_argument("--predict", nargs="+", metavar="TEXT", help="classify your own reviews after training")
    args = parser.parse_args(argv)

    df = load_dataset(args.csv)
    print(f"Loaded {len(df):,} reviews ({(df['sentiment'] == 'positive').mean():.1%} positive)")
    model, m = train_and_evaluate(df)
    print(f"Accuracy:  {m['accuracy']:.3f}")
    print(f"Precision: {m['precision']:.3f}")
    print(f"Recall:    {m['recall']:.3f}")
    print(f"F1:        {m['f1']:.3f}")
    tn, fp, fn, tp = m["confusion_matrix"].ravel()
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    for text, (label, p) in zip(args.predict or [], predict(model, args.predict or [])):
        print(f"{label:8s} {p:.2f}  {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
