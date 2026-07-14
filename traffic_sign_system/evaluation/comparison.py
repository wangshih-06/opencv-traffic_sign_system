"""Train SVM/KNN/random-forest models on one shared feature split."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from traffic_sign_system.config.settings import VAL_SIZE
from traffic_sign_system.models.train_knn import train_knn
from traffic_sign_system.models.train_random_forest import train_rf
from traffic_sign_system.models.train_svm import train_svm

COMPARISON_COLUMNS = [
    "model",
    "feature_mode",
    "val_acc",
    "test_acc",
    "macro_f1",
    "weighted_f1",
    "train_seconds",
    "predict_seconds",
    "model_size_kb",
]


def _scalar(value: Any) -> Any:
    """Unwrap a scalar value stored by ``numpy.savez``."""
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _decode_json(value: Any, default: Any) -> Any:
    value = _scalar(value)
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, Mapping):
        return dict(value)
    return default


def load_feature_metadata(features_npz_path: Path | str) -> tuple[dict[int, str], str]:
    """Load the label map and feature mode embedded in a feature NPZ file."""
    path = Path(features_npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature file does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        raw_labels = _decode_json(data["label_map"] if "label_map" in data else None, {})
        feature_config = _decode_json(
            data["feature_config"] if "feature_config" in data else None, {}
        )

    labels = {int(key): str(value) for key, value in dict(raw_labels).items()}
    feature_mode = str(feature_config.get("mode", "unknown"))
    return labels, feature_mode


def _require_array(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(f"Feature file is missing required array; expected one of {names}")


def _load_shared_splits(
    features_npz_path: Path | str,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, dict[int, str]]:
    """Load test data and create one stratified train/validation split.

    If the NPZ already contains X_val/y_val, those arrays are used unchanged.
    Otherwise X_train/y_train are split once and then reused by every model.
    """
    path = Path(features_npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature file does not exist: {path}")

    with np.load(path, allow_pickle=False) as data:
        X_train_all = _require_array(data, "X_train", "X_tr")
        y_train_all = _require_array(data, "y_train", "y_tr")
        X_test = _require_array(data, "X_test", "X_te")
        y_test = _require_array(data, "y_test", "y_te")
        has_val = ("X_val" in data or "X_va" in data) and (
            "y_val" in data or "y_va" in data
        )
        X_val_saved = _require_array(data, "X_val", "X_va") if has_val else None
        y_val_saved = _require_array(data, "y_val", "y_va") if has_val else None
        raw_labels = _decode_json(data["label_map"] if "label_map" in data else None, {})
        feature_config = _decode_json(
            data["feature_config"] if "feature_config" in data else None, {}
        )

    X_train_all = np.asarray(X_train_all)
    y_train_all = np.asarray(y_train_all).reshape(-1)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).reshape(-1)

    if X_train_all.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must both be 2D feature matrices")
    if X_train_all.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test feature dimensions do not match")
    if len(X_train_all) != len(y_train_all) or len(X_test) != len(y_test):
        raise ValueError("Each feature matrix must have the same length as its labels")
    if len(X_train_all) == 0 or len(X_test) == 0:
        raise ValueError("Train and test sets must not be empty")
    if not np.isfinite(X_train_all).all() or not np.isfinite(X_test).all():
        raise ValueError("Feature matrices contain NaN or infinite values")

    if X_val_saved is not None and y_val_saved is not None:
        X_train = X_train_all
        y_train = y_train_all
        X_val = np.asarray(X_val_saved)
        y_val = np.asarray(y_val_saved).reshape(-1)
        if X_val.ndim != 2 or X_val.shape[1] != X_train.shape[1]:
            raise ValueError("Saved validation features have an incompatible shape")
        if len(X_val) != len(y_val) or len(X_val) == 0:
            raise ValueError("Saved validation features/labels are empty or misaligned")
        if not np.isfinite(X_val).all():
            raise ValueError("Validation feature matrix contains NaN or infinite values")
    else:
        # This is the only split operation in the comparison pipeline. All three
        # classifiers consume these exact arrays, guaranteeing a fair comparison.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_all,
            y_train_all,
            test_size=VAL_SIZE,
            random_state=random_state,
            stratify=y_train_all,
        )

    if len(np.unique(y_train)) < 2:
        raise ValueError("Comparison training data must contain at least two classes")

    embedded_labels = {int(key): str(value) for key, value in dict(raw_labels).items()}
    feature_mode = str(feature_config.get("mode", "unknown"))
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_mode, embedded_labels


def _serialized_size_kb(model: Any) -> float:
    """Measure a fitted estimator by serializing it without a RAM-sized copy."""
    # KNN stores the complete training matrix, so serializing to BytesIO can
    # temporarily duplicate hundreds of MB. A temporary directory keeps peak
    # memory bounded and also handles joblib versions that emit sidecar files.
    with tempfile.TemporaryDirectory(prefix="traffic_sign_model_size_") as tmp_dir:
        model_path = Path(tmp_dir) / "model.joblib"
        joblib.dump(model, model_path, compress=0)
        size_bytes = sum(path.stat().st_size for path in Path(tmp_dir).rglob("*") if path.is_file())
    return size_bytes / 1024.0


def _normalized_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> np.ndarray:
    matrix = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)


def plot_comparison(df: pd.DataFrame, out_path: Path | str) -> Path:
    """Save grouped validation/test accuracy and macro-F1 bars."""
    missing = [column for column in COMPARISON_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Comparison DataFrame is missing columns: {missing}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    models = df["model"].astype(str).str.upper().tolist()
    x = np.arange(len(models), dtype=np.float64)
    width = 0.24

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - width, df["val_acc"], width, label="Validation accuracy")
    ax.bar(x, df["test_acc"], width, label="Test accuracy")
    ax.bar(x + width, df["macro_f1"], width, label="Test macro F1")
    ax.set_xticks(x, models)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SVM / KNN / Random Forest comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_confusion_matrices(
    predictions: Mapping[str, np.ndarray],
    y_true: np.ndarray,
    label_map: Mapping[int, str],
    out_path: Path | str,
) -> Path:
    """Save normalized confusion matrices for all compared models in one image."""
    observed = set(np.asarray(y_true).astype(int).tolist())
    for y_pred in predictions.values():
        observed.update(np.asarray(y_pred).astype(int).tolist())
    labels = sorted(set(int(key) for key in label_map) | observed)
    if not labels:
        raise ValueError("Cannot plot a confusion matrix without class labels")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_models = len(predictions)
    panel_size = max(5.0, min(10.0, len(labels) * 0.18))
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(panel_size * n_models, panel_size),
        squeeze=False,
        constrained_layout=True,
    )

    image = None
    tick_positions = np.arange(len(labels))
    tick_font_size = 6 if len(labels) > 20 else 8
    for ax, (model_name, y_pred) in zip(axes[0], predictions.items()):
        matrix = _normalized_confusion(y_true, y_pred, labels)
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(model_name.upper())
        ax.set_xlabel("Predicted class id")
        ax.set_ylabel("True class id")
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=tick_font_size)
        ax.set_yticklabels(labels, fontsize=tick_font_size)

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.78)
        colorbar.set_label("Recall fraction")
    fig.suptitle("Normalized test confusion matrices", fontsize=14)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_comparison(
    features_npz_path: Path | str,
    out_dir: Path,
    label_map: Mapping[int, str] | None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run SVM/KNN/RF on one shared split and write comparison artifacts.

    The returned columns are ``model``, ``feature_mode``, ``val_acc``,
    ``test_acc``, ``macro_f1``, ``weighted_f1``, ``train_seconds``,
    ``predict_seconds`` and ``model_size_kb``. F1 values are measured on the
    test set. ``predict_seconds`` measures prediction on the test set only.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_mode,
        embedded_labels,
    ) = _load_shared_splits(features_npz_path, random_state=random_state)

    effective_label_map = (
        {int(key): str(value) for key, value in label_map.items()}
        if label_map is not None
        else embedded_labels
    )
    all_classes = sorted(
        set(np.concatenate((y_train, y_val, y_test)).astype(int).tolist())
    )
    for class_id in all_classes:
        effective_label_map.setdefault(class_id, str(class_id))

    builders: list[tuple[str, Callable[[], tuple[Any, float]]]] = [
        (
            "svm",
            lambda: train_svm(
                X_train,
                y_train,
                random_state=random_state,
            ),
        ),
        ("knn", lambda: train_knn(X_train, y_train)),
        (
            "rf",
            lambda: train_rf(
                X_train,
                y_train,
                random_state=random_state,
            ),
        ),
    ]

    rows: list[dict[str, Any]] = []
    test_predictions: dict[str, np.ndarray] = {}
    for model_name, build_clf in builders:
        classifier, train_seconds = build_clf()

        y_val_pred = classifier.predict(X_val)
        predict_started = time.perf_counter()
        y_test_pred = classifier.predict(X_test)
        predict_seconds = time.perf_counter() - predict_started
        test_predictions[model_name] = np.asarray(y_test_pred)

        rows.append(
            {
                "model": model_name,
                "feature_mode": feature_mode,
                "val_acc": float(accuracy_score(y_val, y_val_pred)),
                "test_acc": float(accuracy_score(y_test, y_test_pred)),
                "macro_f1": float(
                    f1_score(y_test, y_test_pred, average="macro", zero_division=0)
                ),
                "weighted_f1": float(
                    f1_score(y_test, y_test_pred, average="weighted", zero_division=0)
                ),
                "train_seconds": float(train_seconds),
                "predict_seconds": float(predict_seconds),
                "model_size_kb": float(_serialized_size_kb(classifier)),
            }
        )

    df = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    df.to_csv(out_dir / "comparison.csv", index=False, encoding="utf-8")
    plot_comparison(df, out_dir / "comparison.png")
    plot_confusion_matrices(
        test_predictions,
        y_test,
        effective_label_map,
        out_dir / "confusion_matrices.png",
    )
    return df
