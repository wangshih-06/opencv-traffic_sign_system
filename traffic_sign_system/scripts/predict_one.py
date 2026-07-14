"""Predict one traffic-sign image with a saved model bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from traffic_sign_system.config.settings import MODEL_ARTIFACTS_DIR, ROOT_DIR
from traffic_sign_system.recognition.predictor import Predictor


NO_PROBABILITY_MESSAGE = (
    "本模型不支持概率输出"
    "（仅 SVM/KNN/RF 中部分支持）"
)


def _resolve_bundle(value: Path) -> Path:
    candidates = [value]
    if not value.is_absolute():
        candidates.extend((ROOT_DIR / value, MODEL_ARTIFACTS_DIR / value.name))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Model bundle was not found. Searched: {searched}")


def _resolve_image(value: Path) -> Path:
    candidates = [value]
    if not value.is_absolute():
        candidates.append(ROOT_DIR / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Image was not found. Searched: {searched}")


def _read_bgr(path: Path) -> np.ndarray:
    """Read a color image through bytes to support non-ASCII Windows paths."""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise OSError(f"Could not read image bytes: {path}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict one BGR image with a traffic-sign model bundle."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Bundle path or filename under models/artifacts.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image path; relative paths may start from traffic_sign_system/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    bundle_path = _resolve_bundle(args.bundle)
    image_path = _resolve_image(args.image)

    predictor = Predictor(bundle_path)
    result = predictor.predict(_read_bgr(image_path))

    print(f"class_id: {result['class_id']}")
    print(f"class_name: {result['class_name']}")
    confidence = result["confidence"]
    if confidence is None:
        print("confidence: None")
        print(NO_PROBABILITY_MESSAGE)
    else:
        print(f"confidence: {confidence:.6f}")


if __name__ == "__main__":
    main()
