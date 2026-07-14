"""Random forest classifier training helper."""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def train_rf(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_estimators: int = 200,
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, float]:
    """Fit a random forest and return it together with elapsed seconds."""
    X_tr = np.asarray(X_tr)
    y_tr = np.asarray(y_tr)
    if X_tr.ndim != 2:
        raise ValueError(f"X_tr must be a 2D feature matrix; got shape={X_tr.shape}")
    if len(X_tr) != len(y_tr):
        raise ValueError("X_tr and y_tr must contain the same number of samples")
    if len(X_tr) == 0:
        raise ValueError("Random forest training requires at least one sample")
    if len(np.unique(y_tr)) < 2:
        raise ValueError("Random forest training requires at least two classes")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    if max_depth is not None and max_depth < 1:
        raise ValueError("max_depth must be None or at least 1")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    started = time.perf_counter()
    clf.fit(X_tr, y_tr)
    return clf, time.perf_counter() - started
