"""Confusion matrix visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: Mapping[int, str],
    out_path: Path | str,
    normalize: bool = True,
) -> Path:
    """Save a class-aligned confusion matrix PNG and return its path."""
    labels = sorted(int(label_id) for label_id in label_map)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must contain the same number of samples")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(
            matrix.astype(np.float32),
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float32),
            where=row_sums != 0,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    size = max(8.0, len(labels) * 0.30)
    fig, ax = plt.subplots(figsize=(size, max(6.0, len(labels) * 0.30)))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0.0)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Recall fraction" if normalize else "Sample count")

    tick_labels = [str(label_id) for label_id in labels]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Predicted class id")
    ax.set_ylabel("True class id")
    ax.set_title("Normalized confusion matrix" if normalize else "Confusion matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
