"""图像预处理：尺寸统一、去噪、CLAHE、自适应增强与归一化。"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Preprocessor:
    """可配置预处理链，并可按当前图像质量自适应调整参数。"""

    def __init__(
        self,
        img_size: int = 64,
        to_gray: bool = True,
        gaussian_ksize: int = 3,
        clahe: bool = True,
        clahe_clip: float = 2.0,
        clahe_grid: int = 8,
        normalize: str = "divide255",
        adaptive: bool = False,
    ) -> None:
        if img_size < 1:
            raise ValueError("img_size 必须 >= 1。")
        self._validate_gaussian_ksize(gaussian_ksize)
        if normalize not in ("divide255", "minmax"):
            raise ValueError("normalize 只能是 'divide255' 或 'minmax'。")
        if clahe_clip <= 0:
            raise ValueError("clahe_clip 必须 > 0。")
        if clahe_grid < 1:
            raise ValueError("clahe_grid 必须 >= 1。")

        self.img_size = int(img_size)
        self.to_gray = bool(to_gray)
        self.gaussian_ksize = int(gaussian_ksize)
        self.clahe_enabled = bool(clahe)
        self.clahe_clip = float(clahe_clip)
        self.clahe_grid = int(clahe_grid)
        self.normalize = normalize
        self.adaptive = bool(adaptive)
        self._runtime_params: dict[str, Any] = {}
        self.last_analysis: dict[str, float | list[str]] = {}
        self.last_effective_params: dict[str, Any] = {}

        self._clahe = self._make_clahe(self.clahe_clip) if self.clahe_enabled else None
        self.config = {
            "img_size": self.img_size,
            "to_gray": self.to_gray,
            "gaussian_ksize": self.gaussian_ksize,
            "clahe": self.clahe_enabled,
            "clahe_clip": self.clahe_clip,
            "clahe_grid": self.clahe_grid,
            "normalize": self.normalize,
            "adaptive": self.adaptive,
        }

    def set_adaptive(self, enabled: bool) -> None:
        """运行时开启/关闭自适应增强。"""
        self.adaptive = bool(enabled)
        self.config["adaptive"] = self.adaptive
        if not self.adaptive:
            self._runtime_params.clear()
            self._clahe = self._make_clahe(self.clahe_clip) if self.clahe_enabled else None

    def set_runtime_params(self, **params: Any) -> None:
        """应用 :class:`SceneAnalyzer` 推荐的本帧参数。

        支持 ``clahe``、``clahe_clip``、``gaussian_ksize``、``normalize``、
        ``sharpen``。这些参数只影响运行时，不会改写训练时基础配置。
        """
        allowed = {"clahe", "clahe_clip", "gaussian_ksize", "normalize", "sharpen"}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f"未知运行时预处理参数: {sorted(unknown)}")
        if "gaussian_ksize" in params:
            self._validate_gaussian_ksize(int(params["gaussian_ksize"]))
        if "normalize" in params and params["normalize"] not in ("divide255", "minmax"):
            raise ValueError("normalize 只能是 'divide255' 或 'minmax'。")
        self._runtime_params = dict(params)

    def clear_runtime_params(self) -> None:
        self._runtime_params.clear()

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """执行预处理，返回 ``uint8`` 灰度图或 BGR 图。"""
        img_bgr = self._validate_image(img_bgr)
        # 分类器最终只使用固定尺寸；先缩小能显著降低自适应分析和增强开销。
        img = cv2.resize(
            img_bgr,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_AREA if max(img_bgr.shape[:2]) > self.img_size else cv2.INTER_LINEAR,
        )

        effective = {
            "clahe": self.clahe_enabled,
            "clahe_clip": self.clahe_clip,
            "gaussian_ksize": self.gaussian_ksize,
            "normalize": self.normalize,
            "sharpen": False,
        }
        if self.adaptive:
            effective.update(self._runtime_params)
            img, adaptive_params = self._adaptive_enhance(img)
            effective.update(adaptive_params)

        if self.to_gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gaussian_ksize = int(effective["gaussian_ksize"])
        if gaussian_ksize != 0:
            img = cv2.GaussianBlur(img, (gaussian_ksize, gaussian_ksize), sigmaX=0)

        if bool(effective["clahe"]):
            clip = float(effective["clahe_clip"])
            self._clahe = self._make_clahe(clip)
            if self.to_gray:
                img = self._clahe.apply(img)
            else:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                l_channel = self._clahe.apply(l_channel)
                img = cv2.cvtColor(
                    cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
                )
        else:
            self._clahe = None

        img = self._normalize(img, method=str(effective["normalize"]))
        self.last_effective_params = dict(effective)
        return img

    def _adaptive_enhance(self, img_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """根据图像质量自动调整增强参数并返回本帧有效参数。"""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        contrast = float(gray.std())
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        degradations: list[str] = []
        params: dict[str, Any] = {}

        if brightness < 80.0:
            degradations.append("low_light")
            params["clahe"] = True
            params["clahe_clip"] = 4.0

        if contrast < 40.0:
            degradations.append("fog")
            params["clahe"] = True
            params["clahe_clip"] = max(float(params.get("clahe_clip", 0.0)), 3.0)
            params["normalize"] = "minmax"
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            fog_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_channel = fog_clahe.apply(l_channel)
            img_bgr = cv2.cvtColor(
                cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR
            )

        if lap_var < 50.0:
            degradations.append("blur")
            params["gaussian_ksize"] = 0
            params["sharpen"] = True

        # SceneAnalyzer 的外部建议优先级与内置分析一致；任一方判为模糊即锐化。
        if bool(self._runtime_params.get("sharpen", False)):
            params["sharpen"] = True
            params["gaussian_ksize"] = 0
        if bool(params.get("sharpen", False)):
            kernel = np.array(
                [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32
            )
            img_bgr = cv2.filter2D(img_bgr, -1, kernel)

        self.last_analysis = {
            "brightness": brightness,
            "contrast": contrast,
            "blur_score": lap_var,
            "degradations": degradations,
        }
        return img_bgr, params

    def _make_clahe(self, clip_limit: float):
        return cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(self.clahe_grid, self.clahe_grid),
        )

    def _normalize(self, img: np.ndarray, *, method: str | None = None) -> np.ndarray:
        """归一化并确保返回 uint8、值在 [0, 255]。"""
        method = self.normalize if method is None else method
        if method == "divide255":
            img_f = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
        else:
            img_f = img.astype(np.float32)
            v_min, v_max = float(img_f.min()), float(img_f.max())
            if v_max - v_min > 1e-8:
                img_f = (img_f - v_min) / (v_max - v_min)
            img_f = np.clip(img_f, 0.0, 1.0)
        return (img_f * 255).astype(np.uint8)

    @staticmethod
    def _validate_gaussian_ksize(value: int) -> None:
        if value != 0 and (value % 2 == 0 or value < 1):
            raise ValueError("gaussian_ksize 必须是 0 或正奇数。")

    @staticmethod
    def _validate_image(img_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(img_bgr, np.ndarray):
            raise TypeError("img_bgr 必须是 numpy.ndarray")
        if img_bgr.size == 0 or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError("img_bgr 必须是非空 BGR 三通道图像")
        if img_bgr.dtype == np.uint8:
            return img_bgr
        return np.clip(img_bgr, 0, 255).astype(np.uint8)
