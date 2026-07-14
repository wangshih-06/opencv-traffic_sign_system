"""HSV 颜色直方图特征提取器 — 基于 H/S 通道的二维直方图"""

import cv2
import numpy as np


class HSVColorHistogram:
    """计算 H/S 通道的二维直方图，固定维度（默认 8×8=64）。

    输入 BGR 图像，先转 HSV，再对 H∈[0,180) 和 S∈[0,256) 做二维分箱，
    展平后做 L1 归一化（sum=1），便于与 HOG 等特征拼接后统一标准化。
    """

    def __init__(self, h_bins: int = 8, s_bins: int = 8):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.config = dict(h_bins=h_bins, s_bins=s_bins)

    def extract(self, img_bgr: np.ndarray) -> np.ndarray:
        """返回 (h_bins*s_bins,) 的一维 L1 归一化直方图"""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # H 在 [0,180), S 在 [0,256)；按配置分箱
        hist = cv2.calcHist([hsv], channels=[0, 1], mask=None,
                            histSize=[self.h_bins, self.s_bins],
                            ranges=[0, 180, 0, 256])
        hist = hist.reshape(-1).astype(np.float32)
        hist /= (hist.sum() + 1e-6)
        return hist

    def extract_batch(self, imgs_bgr: list[np.ndarray]) -> np.ndarray:
        """批量提取，返回 (N, h_bins*s_bins)"""
        return np.stack([self.extract(im) for im in imgs_bgr], axis=0)
