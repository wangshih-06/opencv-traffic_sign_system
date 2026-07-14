"""HOG（方向梯度直方图）特征提取器。

基于 cv2.HOGDescriptor，对单张 64×64 uint8 灰度图输出固定维度的
一维 float32 特征向量；并提供 batch 入口。
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HOGExtractor:
    """HOG 特征提取器，固定窗口尺寸（默认 64×64），用于 GTSRB 训练/推理一致性。"""

    def __init__(
        self,
        win_size: tuple[int, int] = (64, 64),
        block_size: tuple[int, int] = (16, 16),
        block_stride: tuple[int, int] = (8, 8),
        cell_size: tuple[int, int] = (8, 8),
        nbins: int = 9,
    ) -> None:
        # OpenCV 的 HOGDescriptor 接受 (width, height)，与 NumPy (h, w) 相反
        self.win_size: tuple[int, int] = tuple(win_size)
        self.block_size: tuple[int, int] = tuple(block_size)
        self.block_stride: tuple[int, int] = tuple(block_stride)
        self.cell_size: tuple[int, int] = tuple(cell_size)
        self.nbins: int = int(nbins)

        # 构造底层 HOGDescriptor（OpenCV 接受 (W, H)）
        self.hog = cv2.HOGDescriptor(
            self.win_size,
            self.block_size,
            self.block_stride,
            self.cell_size,
            self.nbins,
        )

        # 配置字典：用于持久化与一致性校验
        self.config: dict = {
            "win_size": self.win_size,
            "block_size": self.block_size,
            "block_stride": self.block_stride,
            "cell_size": self.cell_size,
            "nbins": self.nbins,
        }

        # 维度缓存（首次调用 feature_dim() 后填充）
        self._dim: int | None = None

    # ──────────────────────────────────────────────────────────────────
    # 单张
    # ──────────────────────────────────────────────────────────────────
    def extract(self, img_gray_u8: np.ndarray) -> np.ndarray:
        """从单张 uint8 灰度图中提取 HOG 特征，返回一维 float32 向量。

        参数
        ----
        img_gray_u8 : np.ndarray
            单通道灰度图，dtype=uint8，shape=(H, W)。

        返回
        ----
        np.ndarray
            一维 float32 特征向量，shape=(D,)。

        异常
        ----
        ValueError
            - 输入维度异常（非 2D、含 NaN/Inf）
            - 输入尺寸与 win_size 不一致
            - OpenCV `compute()` 返回 None
        """
        # ── 校验输入 ────────────────────────────────────────────────
        if not isinstance(img_gray_u8, np.ndarray):
            raise ValueError(f"输入必须是 np.ndarray，实际为 {type(img_gray_u8).__name__}")
        if img_gray_u8.ndim != 2:
            raise ValueError(
                f"HOG 要求单通道灰度图（ndim=2），实际 ndim={img_gray_u8.ndim}"
            )
        h, w = img_gray_u8.shape
        if (h, w) != tuple(self.win_size[::-1]):  # win_size=(W,H)
            raise ValueError(
                f"HOG 窗口固定为 {self.win_size}（宽×高），"
                f"输入尺寸为 ({w}, {h})，不一致。"
            )
        if not np.isfinite(img_gray_u8.astype(np.float32)).all():
            raise ValueError("输入图像包含 NaN/Inf。")

        # ── 计算 HOG 特征 ──────────────────────────────────────────
        feat = self.hog.compute(img_gray_u8)
        if feat is None:
            raise ValueError(
                f"HOG compute 失败，image shape={img_gray_u8.shape}, "
                f"win_size={self.win_size}"
            )

        return feat.reshape(-1).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────
    # 批量
    # ──────────────────────────────────────────────────────────────────
    def extract_batch(self, imgs_gray: list[np.ndarray]) -> np.ndarray:
        """批量提取 HOG 特征，返回 (N, D) 的 np.ndarray。

        参数
        ----
        imgs_gray : list[np.ndarray]
            灰度图列表，每张 dtype=uint8、shape=(win_h, win_w)。

        返回
        ----
        np.ndarray
            形状为 (N, D)，dtype=float32。
        """
        if not imgs_gray:
            return np.zeros((0, 0), dtype=np.float32)

        feats = [self.extract(im) for im in imgs_gray]
        return np.stack(feats, axis=0)

    # ──────────────────────────────────────────────────────────────────
    # 特征维度
    # ──────────────────────────────────────────────────────────────────
    def feature_dim(self) -> int:
        """返回 HOG 特征维度 D；首次调用后缓存到 self._dim。"""
        if self._dim is None:
            dummy = np.zeros(self.win_size[::-1], dtype=np.uint8)  # (h, w)
            self._dim = self.extract(dummy).shape[0]
            logger.info(f"HOG 特征维度（懒计算）: D = {self._dim}")
        return self._dim

    def __repr__(self) -> str:
        return (
            f"HOGExtractor(win_size={self.win_size}, "
            f"block_size={self.block_size}, block_stride={self.block_stride}, "
            f"cell_size={self.cell_size}, nbins={self.nbins})"
        )
