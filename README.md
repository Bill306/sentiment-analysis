# IMDB Sentiment Analysis

Classifies movie reviews as positive or negative with TF-IDF features and logistic regression. It reaches **89.5% accuracy** on 10,000 held-out reviews from the IMDB 50K dataset.

## Why

This was a first end-to-end NLP project: define the problem, pick evaluation metrics before modeling, then check whether a simple, explainable baseline is already good enough before reaching for deep learning. For binary sentiment on long reviews, it mostly is.

## Results

Test set: 20% of 50,000 reviews (random seed 42), balanced 50/50.

| Metric | Score |
|---|---|
| Accuracy | 0.895 |
| Precision | 0.887 |
| Recall | 0.906 |
| F1 | 0.897 |

Confusion matrix: 4,382 true negatives, 579 false positives, 472 false negatives, 4,567 true positives. Errors are fairly balanced, with slightly more false positives.

## How it works

1. **Clean:** remove `<br />` tags and non-letters, then lowercase.
2. **Features:** TF-IDF over the 5,000 most frequent words. It upweights words like "superb" and "awful" and downweights "the" and "and".
3. **Model:** logistic regression. It's fast on sparse text, hard to overfit at this size, and gives probabilities.
4. **Evaluate:** accuracy, precision, recall, F1 and the confusion matrix.

## Run it

1. Download `IMDB Dataset.csv` from [Kaggle: IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (it isn't included here because of its size).
2. Train and classify your own text:

```bash
pip install -r requirements.txt
python sentiment_analysis.py "IMDB Dataset.csv" --predict "A beautiful, moving film with superb acting."
```

Training takes about 7 seconds on a laptop.

## Limitations

A bag of words can't read sarcasm or idioms. "Two hours of my life I will never get back." is classified as **positive (0.73)** because it contains no clearly negative words. Word order and negation ("not good") are also lost. A model that reads context, such as a fine-tuned transformer, would be the next step.

## Tests

```bash
python -m unittest
```

The tests use a tiny synthetic dataset, so they run without downloading anything.
