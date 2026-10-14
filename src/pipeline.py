"""Preprocessing and model pipeline utilities."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def build_classification_pipeline():
    """
    Build a classification pipeline that scales features
    and fits a logistic regression model.
    Scaling is fitted only on training data when used with Pipeline.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    return pipeline
