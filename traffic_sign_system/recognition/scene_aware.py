"""Fast scene-quality analysis for adaptive traffic-sign recognition."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np


class SceneAnalyzer:
    """分析当前图像的亮度、对比度、模糊和噪声退化。"""

    def __init__(
        self,
        *,
        low_light_threshold: float = 80.0,
        fog_contrast_threshold: float = 40.0,
        blur_threshold: float = 50.0,
        noise_threshold: float = 0.16,
        analysis_width: int = 320,
    ) -> None:
        self.low_light_threshold = float(low_light_threshold)
        self.fog_contrast_threshold = float(fog_contrast_threshold)
        self.blur_threshold = float(blur_threshold)
        self.noise_threshold = float(noise_threshold)
        self.analysis_width = max(64, int(analysis_width))
        self.last_analysis_seconds = 0.0

    def analyze(self, img_bgr: np.ndarray) -> dict[str, Any]:
        """返回场景质量指标及退化类型。

        为满足实时性，长边大于 ``analysis_width`` 的输入会先等比例缩小；
        输出指标仍足以用于阈值式自适应控制。
        """
        started = time.perf_counter()
        image = self._validate_image(img_bgr)
        sample = self._downsample(image)
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        brightness = float(gray.mean())
        contrast = float(gray.std())
        blur_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        # 高频残差能量 / 总灰度能量。噪声越强，该比例通常越高。
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        residual = gray.astype(np.float32) - smooth.astype(np.float32)
        high_frequency_energy = float(np.mean(residual * residual))
        total_energy = float(np.var(gray.astype(np.float32))) + 1e-6
        noise_score = float(high_frequency_energy / total_energy)

        degradations: list[str] = []
        if brightness < self.low_light_threshold:
            degradations.append("low_light")
        if contrast < self.fog_contrast_threshold:
            degradations.append("fog")
        if blur_score < self.blur_threshold:
            degradations.append("blur")
        if noise_score > self.noise_threshold:
            degradations.append("noise")

        self.last_analysis_seconds = time.perf_counter() - started
        return {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": blur_score,
            "noise_score": noise_score,
            "degradations": degradations,
            "analysis_seconds": self.last_analysis_seconds,
        }

    def recommend_params(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """根据分析结果推荐预处理参数。"""
        degradations = set(analysis.get("degradations", ()))
        params: dict[str, Any] = {}
        if "low_light" in degradations:
            params["clahe"] = True
            params["clahe_clip"] = 4.0
        if "fog" in degradations:
            params["clahe"] = True
            # 同时低光时保留更强的 4.0，而不是被雾天参数覆盖。
            params["clahe_clip"] = max(float(params.get("clahe_clip", 0.0)), 3.0)
            params["normalize"] = "minmax"
        if "blur" in degradations:
            params["gaussian_ksize"] = 0
            params["sharpen"] = True
        if "noise" in degradations and "blur" not in degradations:
            params["gaussian_ksize"] = 3
        return params

    def _downsample(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= self.analysis_width:
            return image
        scale = self.analysis_width / float(longest)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _validate_image(img_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(img_bgr, np.ndarray):
            raise TypeError("img_bgr 必须是 numpy.ndarray")
        if img_bgr.size == 0 or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("img_bgr 必须是非空 BGR 三通道图像")
        if img_bgr.dtype == np.uint8:
            return img_bgr
        return np.clip(img_bgr, 0, 255).astype(np.uint8)
