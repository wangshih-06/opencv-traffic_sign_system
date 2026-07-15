"""HOG（方向梯度直方图）特征提取器。

纯 NumPy 实现，兼容 OpenCV 4.x / 5.x（cv2.HOGDescriptor 在 5.0 被移除）。
对单张 64×64 uint8 灰度图输出固定维度的一维 float32 特征向量。
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _compute_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用 [-1,0,1] 核计算 x/y 梯度、幅值和方向（0~180°，无符号）。"""
    gx = np.empty_like(img, dtype=np.float32)
    gy = np.empty_like(img, dtype=np.float32)
    gx[:, 1:-1] = (img[:, 2:].astype(np.float32) - img[:, :-2].astype(np.float32)) * 0.5
    gx[:, 0] = img[:, 1].astype(np.float32) - img[:, 0].astype(np.float32)
    gx[:, -1] = img[:, -1].astype(np.float32) - img[:, -2].astype(np.float32)
    gy[1:-1, :] = (img[2:, :].astype(np.float32) - img[:-2, :].astype(np.float32)) * 0.5
    gy[0, :] = img[1, :].astype(np.float32) - img[0, :].astype(np.float32)
    gy[-1, :] = img[-1, :].astype(np.float32) - img[-2, :].astype(np.float32)

    mag = np.sqrt(gx**2 + gy**2)
    orient = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    return gx, gy, mag, orient


def _cell_histograms(
    mag: np.ndarray, orient: np.ndarray,
    cells_per_row: int, cells_per_col: int,
    cell_h: int, cell_w: int, nbins: int,
) -> np.ndarray:
    """为每个 cell 计算方向梯度直方图，返回 (cells_per_col, cells_per_row, nbins)。"""
    bin_width = 180.0 / nbins
    hists = np.zeros((cells_per_col, cells_per_row, nbins), dtype=np.float32)

    for ci in range(cells_per_col):
        y0, y1 = ci * cell_h, (ci + 1) * cell_h
        for cj in range(cells_per_row):
            x0, x1 = cj * cell_w, (cj + 1) * cell_w
            cell_mag = mag[y0:y1, x0:x1]
            cell_orient = orient[y0:y1, x0:x1]

            for b in range(nbins):
                low = b * bin_width
                high = (b + 1) * bin_width
                mask = (cell_orient >= low) & (cell_orient < high)
                hists[ci, cj, b] = cell_mag[mask].sum()

    return hists


class HOGExtractor:
    """HOG 特征提取器，基于纯 NumPy 实现，与 OpenCV HOGDescriptor 产出维度一致。"""

    def __init__(
        self,
        win_size: tuple[int, int] = (64, 64),
        block_size: tuple[int, int] = (16, 16),
        block_stride: tuple[int, int] = (8, 8),
        cell_size: tuple[int, int] = (8, 8),
        nbins: int = 9,
    ) -> None:
        self.win_size: tuple[int, int] = tuple(win_size)       # (W, H)
        self.block_size: tuple[int, int] = tuple(block_size)    # (W, H)
        self.block_stride: tuple[int, int] = tuple(block_stride)# (W, H)
        self.cell_size: tuple[int, int] = tuple(cell_size)      # (W, H)
        self.nbins: int = int(nbins)

        self.config: dict = {
            "win_size": self.win_size,
            "block_size": self.block_size,
            "block_stride": self.block_stride,
            "cell_size": self.cell_size,
            "nbins": self.nbins,
        }

        # 预计算网格参数
        self._cells_per_row = self.win_size[0] // self.cell_size[0]
        self._cells_per_col = self.win_size[1] // self.cell_size[1]
        self._blocks_per_row = (self._cells_per_row - self.block_size[0] // self.cell_size[0]) // (self.block_stride[0] // self.cell_size[0]) + 1
        self._blocks_per_col = (self._cells_per_col - self.block_size[1] // self.cell_size[1]) // (self.block_stride[1] // self.cell_size[1]) + 1
        self._cells_per_block_x = self.block_size[0] // self.cell_size[0]
        self._cells_per_block_y = self.block_size[1] // self.cell_size[1]

        self._dim: int | None = None

    # ── 单张 ────────────────────────────────────────────────────────
    def extract(self, img_gray_u8: np.ndarray) -> np.ndarray:
        """从单张 uint8 灰度图中提取 HOG 特征，返回一维 float32 向量。"""
        if not isinstance(img_gray_u8, np.ndarray):
            raise ValueError(f"输入必须是 np.ndarray，实际为 {type(img_gray_u8).__name__}")
        if img_gray_u8.ndim != 2:
            raise ValueError(f"HOG 要求单通道灰度图（ndim=2），实际 ndim={img_gray_u8.ndim}")
        h, w = img_gray_u8.shape
        if (w, h) != self.win_size:
            raise ValueError(f"HOG 窗口固定为 {self.win_size}（宽×高），输入尺寸为 ({w}, {h})，不一致。")

        img = img_gray_u8.astype(np.float32)
        if not np.isfinite(img).all():
            raise ValueError("输入图像包含 NaN/Inf。")

        _, _, mag, orient = _compute_gradients(img)

        # 逐 cell 直方图
        cell_hists = _cell_histograms(
            mag, orient,
            self._cells_per_row, self._cells_per_col,
            self.cell_size[1], self.cell_size[0], self.nbins,
        )

        # 逐 block 拼接 + L2 归一化
        block_cx = self._cells_per_block_x
        block_cy = self._cells_per_block_y
        stride_cx = self.block_stride[0] // self.cell_size[0]
        stride_cy = self.block_stride[1] // self.cell_size[1]

        blocks = []
        eps = 1e-6
        for bi in range(self._blocks_per_col):
            for bj in range(self._blocks_per_row):
                ci, cj = bi * stride_cy, bj * stride_cx
                block_feat = cell_hists[ci:ci+block_cy, cj:cj+block_cx, :].ravel()
                # L2-Hys (L2 norm + clamp)
                norm = np.sqrt((block_feat**2).sum()) + eps
                block_feat = np.minimum(block_feat / norm, 0.2)
                norm2 = np.sqrt((block_feat**2).sum()) + eps
                block_feat /= norm2
                blocks.append(block_feat)

        feat = np.concatenate(blocks).astype(np.float32)

        if self._dim is None:
            self._dim = feat.shape[0]
            logger.info(f"HOG 特征维度（懒计算）: D = {self._dim}")

        return feat

    # ── 批量 ────────────────────────────────────────────────────────
    def extract_batch(self, imgs_gray: list[np.ndarray]) -> np.ndarray:
        if not imgs_gray:
            return np.zeros((0, 0), dtype=np.float32)
        feats = [self.extract(im) for im in imgs_gray]
        return np.stack(feats, axis=0)

    # ── 维度 ────────────────────────────────────────────────────────
    def feature_dim(self) -> int:
        if self._dim is None:
            dummy = np.zeros((self.win_size[1], self.win_size[0]), dtype=np.uint8)
            self._dim = int(self.extract(dummy).shape[0])
        return self._dim

    def __repr__(self) -> str:
        return (
            f"HOGExtractor(win_size={self.win_size}, "
            f"block_size={self.block_size}, block_stride={self.block_stride}, "
            f"cell_size={self.cell_size}, nbins={self.nbins})"
        )
