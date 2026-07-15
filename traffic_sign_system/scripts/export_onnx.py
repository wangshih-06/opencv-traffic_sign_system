"""Export joblib model bundles to ONNX for faster inference.

Usage::

    # Export a single bundle
    python -m traffic_sign_system.scripts.export_onnx svm_hog+hsv.joblib

    # Export all bundles in models/artifacts/
    python -m traffic_sign_system.scripts.export_onnx --all

    # Custom output directory and opset
    python -m traffic_sign_system.scripts.export_onnx --all --out models/onnx --opset 17

The exporter is **lazy** about its dependencies: skl2onnx and onnx are
imported only when an actual export is requested. If they are missing,
the script exits with a clear error message.

Output: ``<bundle>.onnx`` written next to the input bundle. For
:class:`~traffic_sign_system.models.train_ensemble.EnsembleClassifier`
bundles, additional per-sub-model files ``<bundle>__<name>.onnx`` are
also written.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m traffic_sign_system.scripts.export_onnx",
        description="Export joblib model bundles to ONNX for faster inference.",
    )
    parser.add_argument(
        "bundle",
        nargs="?",
        help="Path or name of a .joblib bundle to export. Use --all to export "
        "every bundle in models/artifacts/.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every .joblib bundle in models/artifacts/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (defaults to the same directory as the input bundle).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a tabular report of input size, ONNX size and elapsed time.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose (DEBUG) logging.",
    )
    return parser


def _default_artifacts_dir() -> Path:
    # traffic_sign_system/scripts/export_onnx.py -> ../../models/artifacts
    here = Path(__file__).resolve().parent
    return here.parent / "models" / "artifacts"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.bundle and not args.all:
        parser.error("Provide a bundle path or pass --all.")
    if args.bundle and args.all:
        parser.error("--all and a positional bundle path are mutually exclusive.")

    artifacts = _default_artifacts_dir()
    if args.all:
        if not artifacts.is_dir():
            logger.error("Artifacts directory not found: %s", artifacts)
            return 2
        bundles = sorted(artifacts.glob("*.joblib"))
        if not bundles:
            logger.error("No .joblib bundles found in %s", artifacts)
            return 2
    else:
        bundle_path = Path(args.bundle)
        if not bundle_path.is_absolute():
            candidate_artifacts = artifacts / bundle_path.name
            candidate_cwd = Path.cwd() / bundle_path
            if candidate_artifacts.is_file():
                bundle_path = candidate_artifacts
            elif candidate_cwd.is_file():
                bundle_path = candidate_cwd
            else:
                logger.error("Bundle not found: %s", args.bundle)
                return 2
        bundles = [bundle_path]

    try:
        from traffic_sign_system.models.onnx_exporter import export_bundle_to_onnx
    except ImportError as exc:
        logger.error("Cannot import onnx_exporter: %s", exc)
        return 3

    rows: list[tuple[str, int, int, float]] = []
    overall_started = time.perf_counter()
    for bundle_path in bundles:
        out_dir = args.out if args.out else bundle_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / bundle_path.with_suffix(".onnx").name
        size_in = bundle_path.stat().st_size
        started = time.perf_counter()
        try:
            written = export_bundle_to_onnx(bundle_path, out_path, target_opset=args.opset)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to export %s: %s", bundle_path.name, exc)
            continue
        elapsed = time.perf_counter() - started
        size_out = written.stat().st_size
        rows.append((bundle_path.name, size_in, size_out, elapsed))
        logger.info(
            "Wrote %s (%.1f MB -> %.1f MB) in %.2fs",
            written.name,
            size_in / 1024 / 1024,
            size_out / 1024 / 1024,
            elapsed,
        )

    total_elapsed = time.perf_counter() - overall_started

    if args.report and rows:
        name_w = max(len(r[0]) for r in rows)
        print()
        print(f"{'bundle'.ljust(name_w)}  {'joblib (MB)':>12}  {'onnx (MB)':>10}  {'time (s)':>9}")
        print("-" * (name_w + 40))
        for name, size_in, size_out, elapsed in rows:
            print(
                f"{name.ljust(name_w)}  "
                f"{size_in / 1024 / 1024:>12.2f}  "
                f"{size_out / 1024 / 1024:>10.2f}  "
                f"{elapsed:>9.2f}"
            )
        print("-" * (name_w + 40))
        print(f"{'(total)'.ljust(name_w)}  {' ':>12}  {' ':>10}  {total_elapsed:>9.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())