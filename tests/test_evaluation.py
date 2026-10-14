"""Automated tests for evaluation and data quality."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.pipeline import build_classification_pipeline
from src.data_checks import (
    check_non_finite,
    check_class_counts,
    check_train_test_overlap,
)


def test_metric_ranges():
    """Metrics should be within valid ranges."""
    X, y = make_classification(n_samples=200, n_features=8, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0
    )
    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_split_integrity():
    """Train and test indices should not overlap."""
    X, y = make_classification(n_samples=150, random_state=1)
    _, _, _, _, idx_train, idx_test = train_test_split(
        X, y, test_size=0.25, random_state=1, return_indices=True
    )
    overlap = check_train_test_overlap(idx_train, idx_test)
    assert overlap == 0


def test_no_invalid_features():
    """Synthetic clean data should have zero non-finite values."""
    X, _ = make_classification(n_samples=100, n_features=5, random_state=2)
    assert check_non_finite(X) == 0
