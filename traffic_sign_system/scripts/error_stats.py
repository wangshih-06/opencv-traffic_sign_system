"""Run bundle evaluation once and export aggregate error statistics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from traffic_sign_system.config.settings import MODEL_ARTIFACTS_DIR, ROOT_DIR
from traffic_sign_system.evaluation.error_analysis import (
    errors_per_class,
    plot_errors_per_class,
    plot_top_confusions,
    top_confusions,
)
from traffic_sign_system.scripts.evaluate import evaluate_bundle


def _resolve_existing_path(value: Path, description: str, extra_roots: tuple[Path, ...]) -> Path:
    candidates = [value]
    if not value.is_absolute():
        candidates.extend(root / value for root in extra_roots)
        candidates.append(MODEL_ARTIFACTS_DIR / value.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"{description} was not found. Searched: {searched}")


def _resolve_bundle(value: Path) -> Path:
    path = _resolve_existing_path(value, "Model bundle", (ROOT_DIR,))
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle is not a file: {path}")
    return path


def _resolve_optional_input(value: Path | None, description: str) -> Path | None:
    if value is None:
        return None
    return _resolve_existing_path(value, description, (ROOT_DIR,))


def run_error_stats(
    bundle_path: Path | str,
    data_dir: Path | str | None = None,
    labels_csv: Path | str | None = None,
    out_dir: Path | str | None = None,
    top_k: int = 10,
    top_n_errors: int = 50,
):
    """Evaluate a bundle once, then derive all aggregate error artifacts."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if top_n_errors < 1:
        raise ValueError("top_n_errors must be at least 1")

    bundle_path = Path(bundle_path)
    destination = Path(out_dir) if out_dir is not None else bundle_path.parent / "error_stats"
    destination.mkdir(parents=True, exist_ok=True)

    evaluation = evaluate_bundle(
        bundle_path,
        data_dir=Path(data_dir) if data_dir is not None else None,
        labels_csv=Path(labels_csv) if labels_csv is not None else None,
        out_dir=destination,
        top_n_errors=top_n_errors,
    )

    # evaluate_bundle/run_evaluation already called classifier.predict. Reuse
    # that exact y_pred array for every statistic below; do not predict again.
    y_true = evaluation["y_true"]
    y_pred = evaluation["y_pred"]
    label_map = evaluation["label_map"]

    confusion_df = top_confusions(y_true, y_pred, label_map, top_k=top_k)
    per_class_df = errors_per_class(y_true, y_pred, label_map)

    confusion_csv = destination / "top_confusions.csv"
    confusion_plot = destination / "top_confusions.png"
    per_class_csv = destination / "errors_per_class.csv"
    per_class_plot = destination / "errors_per_class.png"

    confusion_df.to_csv(confusion_csv, index=False, encoding="utf-8")
    per_class_df.to_csv(per_class_csv, index=False, encoding="utf-8")
    plot_top_confusions(confusion_df, confusion_plot)
    plot_errors_per_class(per_class_df, per_class_plot)

    return {
        "evaluation": evaluation,
        "top_confusions": confusion_df,
        "errors_per_class": per_class_df,
        "top_confusions_csv": confusion_csv,
        "top_confusions_plot": confusion_plot,
        "errors_per_class_csv": per_class_csv,
        "errors_per_class_plot": per_class_plot,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a model once and summarize its most frequent mistakes."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Model bundle path or filename under models/artifacts.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Labeled test image directory; common GTSRB locations are auto-detected.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV containing test filenames and ClassId labels.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <bundle directory>/error_stats).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of confusion pairs to keep.")
    parser.add_argument(
        "--top-n-errors",
        type=int,
        default=50,
        help="Maximum individual mistakes written by the reused evaluator.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    bundle_path = _resolve_bundle(args.bundle)
    data_dir = _resolve_optional_input(args.data, "Test data directory")
    labels_csv = _resolve_optional_input(args.labels_csv, "Label CSV")
    out_dir = args.out.resolve() if args.out is not None else None

    result = run_error_stats(
        bundle_path,
        data_dir=data_dir,
        labels_csv=labels_csv,
        out_dir=out_dir,
        top_k=args.top_k,
        top_n_errors=args.top_n_errors,
    )

    print("Top confusion pairs:")
    top_df = result["top_confusions"]
    print(top_df.to_string(index=False) if not top_df.empty else "No misclassifications.")
    print(f"top_confusions_csv: {result['top_confusions_csv'].resolve()}")
    print(f"top_confusions_plot: {result['top_confusions_plot'].resolve()}")
    print(f"errors_per_class_csv: {result['errors_per_class_csv'].resolve()}")
    print(f"errors_per_class_plot: {result['errors_per_class_plot'].resolve()}")
    print(f"confusion_matrix: {result['evaluation']['confusion_matrix_path'].resolve()}")


if __name__ == "__main__":
    main()
