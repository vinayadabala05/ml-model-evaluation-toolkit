"""Main evaluation script for classification baseline."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.pipeline import build_classification_pipeline
from src.data_checks import run_data_quality_checks


def evaluate_classification():
    """Run stratified evaluation and print metrics + data quality report."""
    # Create a synthetic classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    # Stratified holdout split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=y,
        random_state=42,
        return_indices=True,
    )

    # Build and fit pipeline (scaler fits only on train)
    pipeline = build_classification_pipeline()
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Data quality checks
    quality = run_data_quality_checks(X, y, idx_train, idx_test)

    print("=== Classification Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== Data Quality Report ===")
    for k, v in quality.items():
        print(f"{k}: {v}")

    return metrics, quality


if __name__ == "__main__":
    evaluate_classification()
