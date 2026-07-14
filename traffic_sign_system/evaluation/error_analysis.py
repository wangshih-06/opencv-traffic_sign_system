"""Misclassification exports, aggregate statistics, and visualizations."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ERROR_COLUMNS = (
    "index",
    "true_id",
    "true_name",
    "pred_id",
    "pred_name",
    "confidence",
)
TOP_CONFUSION_COLUMNS = (
    "true_id",
    "true_name",
    "pred_id",
    "pred_name",
    "count",
)
PER_CLASS_COLUMNS = (
    "class_id",
    "class_name",
    "support",
    "correct",
    "error_count",
    "recall",
)


def _normalize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of samples")
    return y_true, y_pred


def _normalize_label_map(label_map: Mapping[int, str]) -> dict[int, str]:
    try:
        return {int(key): str(value) for key, value in label_map.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("label_map keys must be integer class ids") from exc


def top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Mapping[int, str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Return the most frequent directed ``true -> predicted`` mistakes.

    The output columns are ``true_id``, ``true_name``, ``pred_id``,
    ``pred_name`` and ``count``. Correct predictions are excluded.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    y_true, y_pred = _normalize_predictions(y_true, y_pred)
    labels = _normalize_label_map(label_map)

    pairs = Counter(zip(y_true.tolist(), y_pred.tolist()))
    rows = [
        {
            "true_id": int(true_id),
            "true_name": labels.get(int(true_id), str(int(true_id))),
            "pred_id": int(pred_id),
            "pred_name": labels.get(int(pred_id), str(int(pred_id))),
            "count": int(count),
        }
        for (true_id, pred_id), count in pairs.items()
        if true_id != pred_id
    ]
    df = pd.DataFrame(rows, columns=TOP_CONFUSION_COLUMNS)
    if df.empty:
        return df
    return (
        df.sort_values(
            ["count", "true_id", "pred_id"],
            ascending=[False, True, True],
            kind="stable",
        )
        .head(top_k)
        .reset_index(drop=True)
    )


def plot_top_confusions(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Plot a horizontal bar chart of ``true_name -> pred_name`` counts."""
    missing = [column for column in TOP_CONFUSION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Top-confusion DataFrame is missing columns: {missing}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(4.0, 0.55 * max(len(df), 1) + 1.8)
    fig, ax = plt.subplots(figsize=(11, height))

    if df.empty:
        ax.text(
            0.5,
            0.5,
            "No misclassifications",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
        )
        ax.set_axis_off()
    else:
        ordered = df.iloc[::-1]
        pair_labels = [
            f"{row.true_name} ({row.true_id}) -> {row.pred_name} ({row.pred_id})"
            for row in ordered.itertuples(index=False)
        ]
        bars = ax.barh(pair_labels, ordered["count"].astype(int), color="#4C78A8")
        ax.bar_label(bars, padding=3, fmt="%d")
        ax.set_xlabel("Misclassified sample count")
        ax.set_ylabel("True class -> predicted class")
        ax.set_title("Top class confusions")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def errors_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Mapping[int, str],
) -> pd.DataFrame:
    """Return support, correct count, error count, and recall for each class."""
    y_true, y_pred = _normalize_predictions(y_true, y_pred)
    labels = _normalize_label_map(label_map)
    observed = set(y_true.astype(int).tolist()) | set(y_pred.astype(int).tolist())
    class_ids = sorted(set(labels) | observed)

    rows: list[dict[str, Any]] = []
    for class_id in class_ids:
        class_mask = y_true == class_id
        support = int(np.count_nonzero(class_mask))
        correct = int(np.count_nonzero(class_mask & (y_pred == class_id)))
        error_count = support - correct
        recall = correct / support if support else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": labels.get(class_id, str(class_id)),
                "support": support,
                "correct": correct,
                "error_count": error_count,
                "recall": float(recall),
            }
        )
    return pd.DataFrame(rows, columns=PER_CLASS_COLUMNS)


def plot_errors_per_class(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Plot per-class recall and error counts in two aligned panels."""
    missing = [column for column in PER_CLASS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Per-class DataFrame is missing columns: {missing}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    height = max(5.0, min(18.0, 0.34 * max(len(df), 1) + 2.0))
    fig, axes = plt.subplots(1, 2, figsize=(15, height), sharey=True)

    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No class data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
    else:
        ordered = df.sort_values(
            ["error_count", "support", "class_id"],
            ascending=[True, True, False],
            kind="stable",
        )
        class_labels = [
            f"{row.class_id}: {row.class_name}" for row in ordered.itertuples(index=False)
        ]
        recall_bars = axes[0].barh(class_labels, ordered["recall"], color="#59A14F")
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_xlabel("Recall")
        axes[0].set_title("Recall by true class")
        axes[0].grid(axis="x", linestyle="--", alpha=0.35)
        axes[0].bar_label(recall_bars, fmt="%.2f", padding=2, fontsize=7)

        error_bars = axes[1].barh(class_labels, ordered["error_count"], color="#E15759")
        axes[1].set_xlim(left=0)
        axes[1].set_xlabel("Error count")
        axes[1].set_title("Errors by true class")
        axes[1].grid(axis="x", linestyle="--", alpha=0.35)
        axes[1].bar_label(error_bars, fmt="%d", padding=2, fontsize=7)

    fig.suptitle("Per-class error distribution", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _predicted_confidences(classifier: Any, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return confidence for each predicted label, or NaN if probabilities are absent."""
    if not hasattr(classifier, "predict_proba"):
        return np.full(len(y_pred), np.nan, dtype=np.float64)

    probabilities = np.asarray(classifier.predict_proba(X))
    classes = np.asarray(getattr(classifier, "classes_", []))
    if probabilities.ndim != 2 or len(classes) != probabilities.shape[1]:
        return np.full(len(y_pred), np.nan, dtype=np.float64)

    class_to_column = {class_id: index for index, class_id in enumerate(classes)}
    result = np.full(len(y_pred), np.nan, dtype=np.float64)
    for index, predicted_id in enumerate(y_pred):
        column = class_to_column.get(predicted_id)
        if column is not None:
            result[index] = float(probabilities[index, column])
    return result


def export_errors(
    X_te: np.ndarray,
    y_te: np.ndarray,
    y_pred: np.ndarray,
    label_map: Mapping[int, str],
    out_csv: Path | str,
    top_n: int = 50,
    classifier: Any | None = None,
    sample_indices: np.ndarray | None = None,
) -> Path:
    """Write highest-confidence mistakes to CSV and return the destination path.

    ``X_te`` must be in the same scaled feature space expected by ``classifier``.
    When the classifier lacks ``predict_proba``, confidence is written as NaN.
    """
    X_te = np.asarray(X_te)
    y_te = np.asarray(y_te)
    y_pred = np.asarray(y_pred)
    normalized_label_map = _normalize_label_map(label_map)
    if len(X_te) != len(y_te) or len(y_te) != len(y_pred):
        raise ValueError("X_te, y_te, and y_pred must contain the same number of samples")
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    if sample_indices is None:
        sample_indices = np.arange(len(y_te))
    sample_indices = np.asarray(sample_indices)
    if len(sample_indices) != len(y_te):
        raise ValueError("sample_indices must align with y_te")

    confidences = (
        _predicted_confidences(classifier, X_te, y_pred)
        if classifier is not None
        else np.full(len(y_pred), np.nan, dtype=np.float64)
    )
    error_positions = np.flatnonzero(y_te != y_pred)
    ordered_positions = sorted(
        error_positions.tolist(),
        key=lambda position: (
            not np.isfinite(confidences[position]),
            -confidences[position] if np.isfinite(confidences[position]) else 0.0,
        ),
    )[:top_n]

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ERROR_COLUMNS)
        writer.writeheader()
        for position in ordered_positions:
            true_id = int(y_te[position])
            predicted_id = int(y_pred[position])
            confidence = confidences[position]
            writer.writerow(
                {
                    "index": int(sample_indices[position]),
                    "true_id": true_id,
                    "true_name": normalized_label_map.get(true_id, str(true_id)),
                    "pred_id": predicted_id,
                    "pred_name": normalized_label_map.get(predicted_id, str(predicted_id)),
                    "confidence": "NaN" if not np.isfinite(confidence) else f"{confidence:.8f}",
                }
            )
    return out_csv
