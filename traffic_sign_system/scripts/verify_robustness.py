"""Verify adaptive preprocessing on the bundled low-light sample.

Usage:
    python -m traffic_sign_system.scripts.verify_robustness MODEL.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from traffic_sign_system.recognition.predictor import Predictor
from traffic_sign_system.recognition.scene_aware import SceneAnalyzer


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="验证低光照自适应增强")
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "assets" / "low_light_confidence_test.png",
    )
    args = parser.parse_args()

    image = read_image(args.image)
    analyzer = SceneAnalyzer()
    analyzer.analyze(image)  # OpenCV cold-start warm-up
    analysis = analyzer.analyze(image)
    predictor = Predictor(args.model, use_cache=False)
    if predictor.preprocessor is None:
        raise SystemExit("当前 HSV-only 模型没有 HOG 预处理器，无法执行此项对比。")

    predictor.preprocessor.set_adaptive(False)
    baseline = predictor.predict(image)
    predictor.preprocessor.set_adaptive(True)
    predictor.preprocessor.set_runtime_params(**analyzer.recommend_params(analysis))
    adaptive = predictor.predict(image)

    before = baseline.get("confidence")
    after = adaptive.get("confidence")
    improvement = None if before is None or after is None else float(after) - float(before)
    print(f"场景分析: {analysis}")
    print(f"关闭自适应: {baseline}")
    print(f"开启自适应: {adaptive}")
    if improvement is not None:
        print(f"置信度绝对提升: {improvement:.2%}（目标 > 5%）")
    print(f"有效预处理参数: {predictor.preprocessor.last_effective_params}")


if __name__ == "__main__":
    main()
