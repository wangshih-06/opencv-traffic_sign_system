"""特征融合：统一 HOG / HSV / HOG+HSV 三种特征模式。

FeatureBuilder 根据模式选择性地组合 HOG 和 HSV 提取器，
输出固定维度的一维特征向量，便于下游 SVM / 逻辑回归使用。
"""

import logging

import numpy as np

from ..data_processing.preprocessing import Preprocessor
from .hog_extractor import HOGExtractor
from .color_extractor import HSVColorHistogram

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """统一特征提取器，支持 hog / hsv / hog+hsv 三种模式。

    参数
    ----
    mode : str
        特征模式，取值 {"hog", "hsv", "hog+hsv"}。
    img_size : int
        统一 resize 的目标尺寸（正方形），默认 64。
    h_bins : int
        HSV 直方图 H 通道分箱数，默认 8。
    s_bins : int
        HSV 直方图 S 通道分箱数，默认 8。
    """

    VALID_MODES = ("hog", "hsv", "hog+hsv")

    def __init__(
        self,
        mode: str = "hog",
        img_size: int = 64,
        h_bins: int = 8,
        s_bins: int = 8,
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode 必须是 {self.VALID_MODES} 之一，实际为 '{mode}'"
            )
        self.mode = mode
        self.img_size = img_size

        # 按需创建提取器
        self.hog: HOGExtractor | None = None
        self.hsv: HSVColorHistogram | None = None
        self.prep_gray: Preprocessor | None = None

        if mode != "hsv":
            self.hog = HOGExtractor(win_size=(img_size, img_size))
            self.prep_gray = Preprocessor(to_gray=True, img_size=img_size)

        if mode != "hog":
            self.hsv = HSVColorHistogram(h_bins=h_bins, s_bins=s_bins)

        self.config = dict(
            mode=mode, img_size=img_size, h_bins=h_bins, s_bins=s_bins
        )

    # ──────────────────────────────────────────────────────────────────
    # 单张提取
    # ──────────────────────────────────────────────────────────────────
    def extract_one(self, img_bgr: np.ndarray) -> np.ndarray:
        """从单张 BGR 图像中提取特征，返回一维 float32 向量。

        参数
        ----
        img_bgr : np.ndarray
            BGR 彩色图，shape (H, W, 3)。

        返回
        ----
        np.ndarray
            一维 float32 特征向量：
            - hog   → shape (1764,)
            - hsv   → shape (64,)（默认 8×8）
            - hog+hsv → shape (1764+64,) = (1828,)
        """
        if self.mode == "hog":
            gray = self.prep_gray(img_bgr)
            return self.hog.extract(gray)

        if self.mode == "hsv":
            return self.hsv.extract(img_bgr)

        # hog+hsv：拼接
        gray = self.prep_gray(img_bgr)
        hog_feat = self.hog.extract(gray)
        hsv_feat = self.hsv.extract(img_bgr)
        return np.concatenate([hog_feat, hsv_feat])

    # ──────────────────────────────────────────────────────────────────
    # 批量提取
    # ──────────────────────────────────────────────────────────────────
    def extract_batch(self, imgs_bgr: list[np.ndarray]) -> np.ndarray:
        """批量提取特征，返回 (N, D) 矩阵。

        参数
        ----
        imgs_bgr : list[np.ndarray]
            BGR 彩色图列表，每张 shape (H, W, 3)。

        返回
        ----
        np.ndarray
            形状 (N, D)，dtype float32。
        """
        if not imgs_bgr:
            return np.zeros((0, self.feature_dim()), dtype=np.float32)
        return np.stack([self.extract_one(im) for im in imgs_bgr], axis=0)

    # ──────────────────────────────────────────────────────────────────
    # 特征维度
    # ──────────────────────────────────────────────────────────────────
    def feature_dim(self) -> int:
        """返回当前模式下的特征维度 D。"""
        dim = 0
        if self.hog is not None:
            dim += self.hog.feature_dim()
        if self.hsv is not None:
            dim += self.hsv.h_bins * self.hsv.s_bins
        return dim

    def __repr__(self) -> str:
        return (
            f"FeatureBuilder(mode={self.mode!r}, img_size={self.img_size}, "
            f"feature_dim={self.feature_dim()})"
        )
