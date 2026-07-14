"""Metrics and reusable single-bundle evaluation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from traffic_sign_system.models.model_manager import load_bundle
from .confusion_matrix import plot_confusion_matrix
from .error_analysis import export_errors


def _label_ids(label_map: Mapping[int, str]) -> list[int]:
    """Return label ids in a deterministic numeric order."""
    try:
        return sorted(int(label_id) for label_id in label_map)
    except (TypeError, ValueError) as exc:
        raise ValueError("label_map keys must be integer class ids") from exc


def evaluate(
    classifier: Any,
    X: np.ndarray,
    y: np.ndarray,
    label_map: Mapping[int, str],
) -> dict[str, Any]:
    """Predict a scaled feature matrix and calculate standard classification metrics."""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D feature matrix; got shape={X.shape}")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")

    normalized_label_map = {int(key): str(value) for key, value in label_map.items()}
    labels = _label_ids(normalized_label_map)
    target_names = [normalized_label_map[label_id] for label_id in labels]
    y_pred = np.asarray(classifier.predict(X))
    report = classification_report(
        y,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    return {
        "acc": float(accuracy_score(y, y_pred)),
        "y_pred": y_pred,
        "report": report,
        "report_text": report_text,
        "labels": labels,
    }


def _to_builtin(value: Any) -> Any:
    """Convert NumPy scalar and array values to JSON compatible Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def run_evaluation(
    bundle_path: Path | str,
    X_te: np.ndarray,
    y_te: np.ndarray,
    out_dir: Path | str,
    top_n_errors: int = 50,
    sample_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate raw test features against a saved bundle and write all artifacts.

    Parameters
    ----------
    bundle_path:
        Path to the joblib bundle created by the training script.
    X_te:
        Unscaled feature matrix built using the bundle's feature configuration.
        This function applies the saved scaler internally.
    y_te:
        True class ids aligned with ``X_te``.
    out_dir:
        Output directory for metrics.json, confusion_matrix.png, and errors.csv.

    Returns
    -------
    dict
        Evaluation metrics, report, predictions, and generated artifact paths.
    """
    bundle = load_bundle(bundle_path)
    classifier = bundle["classifier"]
    scaler = bundle["scaler"]
    label_map = {int(key): str(value) for key, value in bundle["label_map"].items()}
    summary = bundle["summary"]

    X_te = np.asarray(X_te, dtype=np.float32)
    y_te = np.asarray(y_te)
    if X_te.ndim != 2:
        raise ValueError(f"X_te must be 2D; got shape={X_te.shape}")
    if len(X_te) != len(y_te):
        raise ValueError("X_te and y_te must contain the same number of samples")
    if len(X_te) == 0:
        raise ValueError("Cannot evaluate an empty test set")

    expected_dim = int(summary["feature_dim"])
    if X_te.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: bundle expects {expected_dim}, "
            f"received {X_te.shape[1]}"
        )
    scaler_dim = getattr(scaler, "n_features_in_", expected_dim)
    if int(scaler_dim) != expected_dim:
        raise ValueError(
            f"Invalid bundle: scaler dimension {scaler_dim} does not match "
            f"summary dimension {expected_dim}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_te_scaled = scaler.transform(X_te)
    result = evaluate(classifier, X_te_scaled, y_te, label_map)

    report = result["report"]
    metrics = {
        "acc": result["acc"],
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "train_seconds": float(summary["train_seconds"]),
        "n_test": int(len(y_te)),
        "model": summary["model"],
        "feature_mode": summary["feature_mode"],
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(_to_builtin(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    confusion_path = plot_confusion_matrix(
        y_te,
        result["y_pred"],
        label_map,
        out_dir / "confusion_matrix.png",
        normalize=True,
    )
    errors_path = export_errors(
        X_te_scaled,
        y_te,
        result["y_pred"],
        label_map,
        out_dir / "errors.csv",
        classifier=classifier,
        top_n=top_n_errors,
        sample_indices=sample_indices,
    )

    return {
        **result,
        # Expose the exact arrays used by evaluate() so downstream analyses can
        # reuse y_pred instead of invoking classifier.predict a second time.
        "y_true": y_te,
        "label_map": label_map,
        "sample_indices": sample_indices,
        "metrics": metrics,
        "metrics_path": metrics_path,
        "confusion_matrix_path": confusion_path,
        "errors_path": errors_path,
    }
