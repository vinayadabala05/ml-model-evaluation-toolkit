"""Data quality checks for ML evaluation."""

import numpy as np
from typing import Tuple


def check_non_finite(X) -> int:
    """Return number of non-finite (NaN or Inf) values."""
    arr = np.asarray(X, dtype=float)
    return int(np.sum(~np.isfinite(arr)))


def check_class_counts(y) -> dict:
    """Return class distribution as a dictionary."""
    unique, counts = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def check_train_test_overlap(train_idx, test_idx) -> int:
    """Return number of overlapping indices between train and test."""
    overlap = set(train_idx).intersection(set(test_idx))
    return len(overlap)


def run_data_quality_checks(X, y, train_idx, test_idx) -> dict:
    """Run all data quality checks and return a summary dict."""
    return {
        "non_finite_count": check_non_finite(X),
        "class_counts": check_class_counts(y),
        "train_test_overlap": check_train_test_overlap(train_idx, test_idx),
        "n_samples": len(y),
        "n_features": X.shape[1] if hasattr(X, "shape") else None,
    }
