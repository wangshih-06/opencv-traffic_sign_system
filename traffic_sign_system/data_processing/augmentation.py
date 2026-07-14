"""数据增强函数（不包含水平翻转，以保留交通标志语义）。

增强池
----
- 几何：``random_affine``、``random_perspective``
- 光度：``random_brightness_contrast``
- 噪声：``gaussian_noise``
- 模糊：``gaussian_blur``、``motion_blur``
- 遮挡：``cutout``

组合策略
----
- ``apply_random`` —— 默认池 + 加权采样
- ``apply_strong`` —— 范围更大的组合（旋转 ±15°、亮度 ±0.3、噪声 σ=10）
- ``apply_random`` 与 ``apply_strong`` 都从 ``AUGMENT_POOL`` 中按权重随机抽样。
"""

import random

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 基础增强操作
# ─────────────────────────────────────────────────────────────────────────────

def random_affine(
    img: np.ndarray,
    max_angle: float = 10.0,
    max_shift: float = 0.1,
    max_scale: float = 0.1,
) -> np.ndarray:
    """
    随机仿射变换：旋转 + 平移 + 缩放。

    参数
    ----
    img : np.ndarray
        输入图像，任意 dtype 和通道数。
    max_angle : float, 默认 10.0
        最大旋转角度（度）。
    max_shift : float, 默认 0.1
        最大平移量（相对于图像尺寸的比例）。
    max_scale : float, 默认 0.1
        最大缩放比例（±）。

    返回
    ----
    np.ndarray
        变换后的新图像（与输入 shape/dtype 相同）。
    """
    h, w = img.shape[:2]

    # 随机参数
    angle = random.uniform(-max_angle, max_angle)
    tx = random.uniform(-max_shift, max_shift) * w
    ty = random.uniform(-max_shift, max_shift) * h
    scale = 1.0 + random.uniform(-max_scale, max_scale)

    # 旋转 + 平移矩阵
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    if img.ndim == 2:
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    else:
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def random_perspective(img: np.ndarray, max_offset: float = 0.15) -> np.ndarray:
    """
    随机透视变换：模拟不同视角下的标志牌。

    参数
    ----
    img : np.ndarray
        输入图像，shape (H, W, C) 或 (H, W)。
    max_offset : float, 默认 0.15
        角点最大偏移比例（相对图像短边）。

    返回
    ----
    np.ndarray
        透视变换后的图像。
    """
    h, w = img.shape[:2]
    side = min(h, w)
    offset = int(max_offset * side)

    # 原始四角
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # 随机偏移四角
    def jitter(pt):
        return [
            float(np.clip(pt[0] + random.randint(-offset, offset), 0, w - 1)),
            float(np.clip(pt[1] + random.randint(-offset, offset), 0, h - 1)),
        ]

    dst = np.float32([jitter(pt) for pt in src])
    M = cv2.getPerspectiveTransform(src, dst)
    if img.ndim == 2:
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def random_brightness_contrast(
    img: np.ndarray,
    brightness: float = 0.2,
    contrast: float = 0.2,
) -> np.ndarray:
    """
    随机亮度与对比度调整。

    参数
    ----
    img : np.ndarray
        输入图像，dtype uint8。
    brightness : float, 默认 0.2
        亮度变化范围（相对于 255 的比例）。
    contrast : float, 默认 0.2
        对比度变化范围。

    返回
    ----
    np.ndarray
        调整后的图像，dtype uint8。
    """
    # brightness: 整体明暗偏移
    alpha = 1.0 + random.uniform(-contrast, contrast)   # 对比度因子
    beta  = random.uniform(-brightness, brightness) * 255  # 亮度偏移

    img_f = img.astype(np.float32)
    img_f = img_f * alpha + beta
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)


def gaussian_noise(img: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """
    添加高斯噪声。

    参数
    ----
    img : np.ndarray
        输入图像，dtype uint8。
    sigma : float, 默认 5.0
        高斯噪声标准差。

    返回
    ----
    np.ndarray
        加噪后的图像，dtype uint8。
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    img_f = img.astype(np.float32) + noise
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)


def gaussian_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    高斯模糊（模拟运动模糊或轻微失焦）。

    参数
    ----
    img : np.ndarray
        输入图像。
    ksize : int, 默认 3
        模糊核大小，必须为奇数。

    返回
    ----
    np.ndarray
        模糊后的图像。
    """
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)


def motion_blur(img: np.ndarray, kernel_size: int = 5, angle: float = 0.0) -> np.ndarray:
    """
    运动模糊：通过沿指定方向累计产生线性模糊，模拟行车抖动。

    参数
    ----
    img : np.ndarray
        输入图像。
    kernel_size : int, 默认 5
        模糊核长度（像素）。
    angle : float, 默认 0.0
        模糊方向（度），0 表示水平。

    返回
    ----
    np.ndarray
        运动模糊后的图像。
    """
    if kernel_size < 3:
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 构造方向性 kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    cos_val = np.cos(np.deg2rad(angle))
    sin_val = np.sin(np.deg2rad(angle))
    for i in range(kernel_size):
        offset = i - center
        x = int(round(center + offset * cos_val))
        y = int(round(center + offset * sin_val))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    if kernel.sum() == 0:
        kernel[center, center] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def cutout(img: np.ndarray, n_holes: int = 1, size: int = 8) -> np.ndarray:
    """
    Cutout 随机遮挡：模拟部分被遮挡的标志牌。

    参数
    ----
    img : np.ndarray
        输入图像。
    n_holes : int, 默认 1
        遮挡块的数量。
    size : int, 默认 8
        单个遮挡块的边长（像素）。

    返回
    ----
    np.ndarray
        被遮挡后的图像。
    """
    out = img.copy()
    h, w = out.shape[:2]
    size = max(1, int(size))
    for _ in range(max(1, int(n_holes))):
        # 保证 hole 完整落在图像内
        if w <= size or h <= size:
            continue
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        if out.ndim == 2:
            out[y:y + size, x:x + size] = 0
        else:
            out[y:y + size, x:x + size, :] = 0
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 增强池 + 加权采样
# ─────────────────────────────────────────────────────────────────────────────

# 每个条目 (name, fn, weight)。权重决定 ``apply_random`` 采样概率。
AUGMENT_POOL: list[tuple[str, callable, float]] = [
    ("random_affine", lambda im: random_affine(im, max_angle=8.0), 1.0),
    ("perspective", lambda im: random_perspective(im, max_offset=0.12), 0.6),
    ("brightness_contrast", lambda im: random_brightness_contrast(im, 0.15, 0.15), 1.0),
    ("gaussian_noise", lambda im: gaussian_noise(im, sigma=4.0), 0.8),
    ("gaussian_blur", lambda im: gaussian_blur(im, 3), 0.4),
    ("motion_blur", lambda im: motion_blur(im, kernel_size=5, angle=random.uniform(-45, 45)), 0.3),
    ("cutout", lambda im: cutout(im, n_holes=1, size=6), 0.3),
]


def _sample_augmentations(img: np.ndarray, weights: list[float], k_range: tuple[int, int]) -> np.ndarray:
    """按权重从 ``AUGMENT_POOL`` 中采样 k 个增强并顺序应用。"""
    if not AUGMENT_POOL:
        return img
    weights = list(weights)
    if len(weights) != len(AUGMENT_POOL):
        # 自动回退到池权重
        weights = [w for _, _, w in AUGMENT_POOL]
    names = [name for name, _, _ in AUGMENT_POOL]
    fns = [fn for _, fn, _ in AUGMENT_POOL]
    k = random.randint(k_range[0], k_range[1])
    chosen_idx = random.choices(range(len(AUGMENT_POOL)), weights=weights, k=k)
    for idx in chosen_idx:
        try:
            img = fns[idx](img)
        except Exception:  # noqa: BLE001
            # 单个增强失败时跳过，不影响后续流程
            continue
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 公开组合策略
# ─────────────────────────────────────────────────────────────────────────────

def apply_random(img: np.ndarray, p: float = 0.5, weights: list[float] | None = None) -> np.ndarray:
    """
    按概率 p 从增强池中按权重随机采样并应用。

    参数
    ----
    img : np.ndarray
        输入图像。
    p : float, 默认 0.5
        是否触发任何增强的概率。
    weights : list[float] | None
        自定义池权重；为 None 时使用 ``AUGMENT_POOL`` 的默认权重。

    返回
    ----
    np.ndarray
        增强后的图像。
    """
    if random.random() >= p:
        return img
    pool_weights = weights if weights is not None else [w for _, _, w in AUGMENT_POOL]
    return _sample_augmentations(img, pool_weights, k_range=(1, 2))


def apply_strong(img: np.ndarray, p: float = 0.7) -> np.ndarray:
    """
    强增强：旋转 ±15°、亮度 ±0.3、噪声 σ=10，并叠加一个
    透视/cutout/motion blur 之一；小样本类别上 recall 提升显著。

    参数
    ----
    img : np.ndarray
        输入图像。
    p : float, 默认 0.7
        触发增强的概率。

    返回
    ----
    np.ndarray
        增强后的图像。
    """
    if random.random() >= p:
        return img

    img = random_affine(img, max_angle=15.0, max_shift=0.12, max_scale=0.15)
    img = random_brightness_contrast(img, brightness=0.3, contrast=0.3)
    img = gaussian_noise(img, sigma=10.0)

    # 附加 1 项几何或遮挡
    extra = random.choice([
        lambda x: random_perspective(x, max_offset=0.15),
        lambda x: cutout(x, n_holes=1, size=8),
        lambda x: motion_blur(x, kernel_size=5, angle=random.uniform(-30, 30)),
    ])
    try:
        img = extra(img)
    except Exception:  # noqa: BLE001
        pass
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 用法示例（训练时 & 测试时）
# ─────────────────────────────────────────────────────────────────────────────
"""
## 训练时
from data_processing.preprocessing import Preprocessor
from data_processing.augmentation import apply_random, apply_strong

preprocessor = Preprocessor()          # 默认参数：64×64, 灰度, CLAHE, divide255

for img_bgr, label in zip(images, labels):
    img = preprocessor(img_bgr)        # 先预处理
    img = apply_random(img, p=0.5)     # 默认 0~2 个增强
    # 小样本类别用 apply_strong 进一步扩张
    if class_counts[label] < 600:
        img = apply_strong(img, p=0.7)
    # → 送入 HOG 特征提取

## 测试时
preprocessor = Preprocessor()          # 与训练完全一致

for img_bgr, label in zip(test_images, test_labels):
    img = preprocessor(img_bgr)        # 只预处理，不增强
    # → 送入 HOG 特征提取
"""