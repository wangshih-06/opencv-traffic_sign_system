"""K-nearest-neighbours classifier training helper."""

from __future__ import annotations

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def train_knn(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_neighbors: int = 5,
    weights: str = "distance",
) -> tuple[KNeighborsClassifier, float]:
    """Fit KNN and return the classifier together with elapsed seconds."""
    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr)
    if X_tr.ndim != 2:
        raise ValueError(f"X_tr must be a 2D feature matrix; got shape={X_tr.shape}")
    if len(X_tr) != len(y_tr):
        raise ValueError("X_tr and y_tr must contain the same number of samples")
    if len(X_tr) == 0:
        raise ValueError("KNN training requires at least one sample")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if n_neighbors > len(X_tr):
        raise ValueError(
            f"n_neighbors ({n_neighbors}) cannot exceed training samples ({len(X_tr)})"
        )
    if weights not in {"uniform", "distance"} and not callable(weights):
        raise ValueError("weights must be 'uniform', 'distance', or a callable")

    clf = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1,
    )
    started = time.perf_counter()
    clf.fit(X_tr, y_tr)
    return clf, time.perf_counter() - started
