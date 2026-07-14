"""Evaluate one saved model bundle against a labeled image test set."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from traffic_sign_system.config.settings import DATASET_DIR, MODEL_ARTIFACTS_DIR
from traffic_sign_system.data_processing.data_loader import load_train_data
from traffic_sign_system.evaluation.evaluator import run_evaluation
from traffic_sign_system.features.feature_fusion import FeatureBuilder
from traffic_sign_system.models.model_manager import load_bundle

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".ppm"}


def _read_image(path: Path) -> np.ndarray | None:
    """Read images through NumPy bytes so non-ASCII paths also work on Windows."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def _contains_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in directory.iterdir()
    )


def _resolve_data_dir(data_dir: Path | None) -> Path:
    """Find the test image directory for common GTSRB layouts."""
    if data_dir is not None:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Test data directory does not exist: {data_dir}")
        return data_dir

    candidates = (
        DATASET_DIR / "test" / "Test",
        DATASET_DIR / "test",
        DATASET_DIR / "train" / "Test",
    )
    for candidate in candidates:
        if _contains_images(candidate):
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find a test image directory. Searched: {searched}")


def _resolve_labels_csv(data_dir: Path, labels_csv: Path | None) -> Path | None:
    if labels_csv is not None:
        labels_csv = Path(labels_csv)
        if not labels_csv.exists():
            raise FileNotFoundError(f"Label CSV does not exist: {labels_csv}")
        return labels_csv

    candidates = (
        data_dir / "GT-final_test.csv",
        data_dir / "Test.csv",
        data_dir.parent / "GT-final_test.csv",
        data_dir.parent / "Test.csv",
        DATASET_DIR / "Test.csv",
        DATASET_DIR / "train" / "Test.csv",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _find_image_path(data_dir: Path, csv_path_value: str) -> Path | None:
    relative = Path(csv_path_value.replace("\\", "/"))
    candidates = (
        data_dir / relative,
        data_dir.parent / relative,
        data_dir / relative.name,
        data_dir.parent / relative.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _crop_roi(image: np.ndarray, row: dict[str, str]) -> np.ndarray:
    """Crop ROI fields when supplied by a GTSRB ground-truth CSV."""
    try:
        x1 = int(row["Roi.X1"])
        y1 = int(row["Roi.Y1"])
        x2 = int(row["Roi.X2"])
        y2 = int(row["Roi.Y2"])
    except (KeyError, TypeError, ValueError):
        return image

    height, width = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    return image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else image


def _load_csv_test_data(data_dir: Path, labels_csv: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Load comma- or semicolon-delimited GTSRB ground-truth CSV files."""
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        header = stream.readline()
        delimiter = ";" if header.count(";") > header.count(",") else ","
        stream.seek(0)
        rows = list(csv.DictReader(stream, delimiter=delimiter))

    images: list[np.ndarray] = []
    labels: list[int] = []
    sample_indices: list[int] = []
    skipped = 0
    for row_index, row in enumerate(rows):
        path_value = (row.get("Filename") or row.get("Path") or "").strip()
        class_value = (row.get("ClassId") or row.get("class_id") or "").strip()
        if not path_value or not class_value:
            skipped += 1
            continue
        try:
            class_id = int(class_value)
        except ValueError:
            skipped += 1
            continue

        image_path = _find_image_path(data_dir, path_value)
        image = _read_image(image_path) if image_path is not None else None
        if image is None:
            skipped += 1
            continue
        images.append(_crop_roi(image, row))
        labels.append(class_id)
        sample_indices.append(row_index)

    logger.info(
        "Loaded %d test samples from %s; skipped %d rows.",
        len(images), labels_csv, skipped,
    )
    return images, np.asarray(labels, dtype=np.int32), np.asarray(sample_indices, dtype=np.int64)


def load_evaluation_data(
    data_dir: Path | None,
    labels_csv: Path | None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Load either a CSV-labeled GTSRB test set or class-directory test data."""
    resolved_data_dir = _resolve_data_dir(data_dir)
    resolved_labels_csv = _resolve_labels_csv(resolved_data_dir, labels_csv)
    if resolved_labels_csv is not None:
        return _load_csv_test_data(resolved_data_dir, resolved_labels_csv)

    images, labels, _, bad_log = load_train_data(resolved_data_dir)
    if bad_log:
        logger.warning("Skipped %d unreadable class-directory test images.", len(bad_log))
    return images, np.asarray(labels, dtype=np.int32), np.arange(len(images), dtype=np.int64)


def evaluate_bundle(
    bundle_path: Path | str,
    data_dir: Path | None = None,
    labels_csv: Path | None = None,
    out_dir: Path | str | None = None,
    top_n_errors: int = 50,
) -> dict:
    """Build raw test features from bundle config and run reusable evaluation."""
    bundle = load_bundle(bundle_path)
    feature_config = dict(bundle["feature_config"])
    expected_keys = {"mode", "img_size", "h_bins", "s_bins"}
    unsupported_keys = set(feature_config) - expected_keys
    if unsupported_keys:
        raise ValueError(
            "Unsupported feature_config keys for FeatureBuilder: "
            f"{sorted(unsupported_keys)}"
        )

    images, y_test, sample_indices = load_evaluation_data(data_dir, labels_csv)
    if not images:
        raise RuntimeError("No labeled test images were loaded")

    builder = FeatureBuilder(**feature_config)
    logger.info("Extracting test features with %s", builder)
    X_test = builder.extract_batch(images)
    destination = Path(out_dir) if out_dir is not None else MODEL_ARTIFACTS_DIR / "eval"
    return run_evaluation(
        bundle_path,
        X_test,
        y_test,
        destination,
        top_n_errors=top_n_errors,
        sample_indices=sample_indices,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate one saved traffic-sign model bundle")
    parser.add_argument("--bundle", type=Path, required=True, help="Path to a .joblib model bundle")
    parser.add_argument("--data", type=Path, default=None, help="Directory containing test images")
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional test ground-truth CSV")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for evaluation artifacts")
    parser.add_argument("--top-n-errors", type=int, default=50, help="Maximum errors written to errors.csv")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    result = evaluate_bundle(
        args.bundle,
        data_dir=args.data,
        labels_csv=args.labels_csv,
        out_dir=args.out,
        top_n_errors=args.top_n_errors,
    )
    metrics = result["metrics"]
    print(result["report_text"])
    print(f"accuracy: {metrics['acc']:.4f}")
    print(f"macro_f1: {metrics['macro_f1']:.4f}")
    print(f"weighted_f1: {metrics['weighted_f1']:.4f}")
    print(f"metrics: {result['metrics_path']}")
    print(f"confusion_matrix: {result['confusion_matrix_path']}")
    print(f"errors: {result['errors_path']}")


if __name__ == "__main__":
    main()
