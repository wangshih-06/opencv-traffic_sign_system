"""GTSRB 数据集加载器：递归扫描 train/<class_id>/ 下的图像，跳过损坏文件；
同时支持 GTSRB 测试集（CSV + ROI）加载和分层划分。"""

import csv
import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# 支持的图片扩展名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}


def load_train_data(
    root: Path,
    labels_csv: Path | None = None,
) -> tuple[
    list[np.ndarray],           # images
    list[int],                 # labels（ClassId）
    list[int],                 # class_ids（同 labels，保留别名）
    list[dict],                # bad_log
]:
    """
    递归扫描 root/<class_id>/*.{jpg,jpeg,png,ppm,bmp}，读取图像。

    参数
    ----
    root : Path
        数据集根目录（包含 train、Test 等子目录的父目录）。
        GTSRB 格式下即 dataset/train/。
    labels_csv : Path | None, 可选
        标签 CSV 路径（当前未使用，保留接口）。

    返回
    ----
    (images, labels, class_ids, bad_log)
        images    : 成功读取的图像列表（彩色 BGR，np.ndarray）
        labels    : 每张图对应的 ClassId 列表
        class_ids : 与 labels 完全相同，保留别名方便调用方
        bad_log   : 读取失败的记录列表，每条为 dict，含
                    path(str), class_id(int), reason(str)

    注意
    ----
    - 使用 cv2.imdecode + np.fromfile 兼容中文路径
    - 空目录或无有效图像时返回空列表，不崩溃
    """
    root = Path(root)
    bad_log: list[dict] = []
    images: list[np.ndarray] = []
    labels: list[int] = []
    class_ids: list[int] = []

    # 先检查 root 下是否有子目录（按 class_id 组织）
    class_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not class_dirs:
        logger.warning(
            f"在 {root} 下未找到任何子目录。"
            "请确认数据集结构为 root/<class_id>/*.png ，"
            "例如 dataset/train/0/00000.png"
        )
        return [], [], [], []

    for class_dir in sorted(class_dirs, key=lambda d: int(d.name) if d.name.isdigit() else -1):
        class_id_str = class_dir.name
        if not class_id_str.isdigit():
            logger.debug(f"跳过非类别目录: {class_dir}")
            continue
        class_id = int(class_id_str)

        # 收集该目录下的所有图片文件
        img_files: list[Path] = []
        for ext in IMAGE_EXTS:
            img_files.extend(class_dir.glob(f"*{ext}"))
            img_files.extend(class_dir.glob(f"*{ext.upper()}"))

        if not img_files:
            logger.debug(f"类别 {class_id} 目录为空: {class_dir}")
            continue

        # Windows file systems are often case-insensitive, so *.png and *.PNG
        # can resolve to the same files. Deduplicate them before loading.
        unique_img_files: list[Path] = []
        seen_paths: set[str] = set()
        for img_path in img_files:
            path_key = str(img_path).casefold()
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_img_files.append(img_path)

        for img_path in sorted(unique_img_files):
            img = _imread(str(img_path))
            if img is None:
                bad_log.append({
                    "path": str(img_path),
                    "class_id": class_id,
                    "reason": "cv2.imdecode 返回 None（文件损坏或格式不支持）",
                })
                logger.warning(f"读取失败 [{class_id}] {img_path.name}")
                continue
            if img.ndim == 0:
                bad_log.append({
                    "path": str(img_path),
                    "class_id": class_id,
                    "reason": "图像 shape 为空数组",
                })
                logger.warning(f"空图像 [{class_id}] {img_path.name}")
                continue

            images.append(img)
            labels.append(class_id)
            class_ids.append(class_id)

    logger.info(
        f"扫描完成：成功 {len(images)} 张，失败 {len(bad_log)} 张，"
        f"类别数 {len(set(labels))}。"
    )
    return images, labels, class_ids, bad_log


def _imread(path: str) -> np.ndarray | None:
    """
    兼容中文路径的图像读取。

    参数
    ----
    path : str
        图像文件路径。

    返回
    ----
    np.ndarray | None
        成功返回 BGR 彩色图像，失败返回 None。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.getLogger(__name__).warning(f"图像读取异常: {path} → {e}")
        return None


def load_test_data(
    root: Path,
    csv_path: Path,
    img_size: int = 64,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    """加载 GTSRB 测试集（CSV + ROI 格式）。

    GTSRB 测试集结构为 Test/*.png（文件名即索引，如 00000.png），
    配合 Test/GT-final_test.csv（含 ClassId 和 ROI 坐标）。
    读取 CSV 得到 ClassId，再用 Roi.X1/Y1/X2/Y2 裁剪后 resize 到 img_size。

    参数
    ----
    root : Path
        测试集图片目录（如 dataset/test/Test/）。
    csv_path : Path
        GT-final_test.csv 路径。
    img_size : int, 默认 64
        裁剪后 resize 的目标尺寸。

    返回
    ----
    (images, labels, class_ids)
        images    : 成功读取的图像列表（BGR，resize 到 img_size×img_size）
        labels    : 每张图对应的 ClassId
        class_ids : 与 labels 完全相同，保留别名
    """
    root = Path(root)
    csv_path = Path(csv_path)

    if not csv_path.exists():
        logger.error(f"测试集 CSV 不存在: {csv_path}")
        return [], [], []

    images: list[np.ndarray] = []
    labels: list[int] = []
    class_ids: list[int] = []
    bad_count = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            filename = row.get("Filename", "").strip()
            if not filename:
                continue

            img_path = root / filename
            img = _imread(str(img_path))
            if img is None:
                bad_count += 1
                continue

            # ROI 裁剪
            try:
                x1 = int(row["Roi.X1"])
                y1 = int(row["Roi.Y1"])
                x2 = int(row["Roi.X2"])
                y2 = int(row["Roi.Y2"])
                # 安全边界检查
                h, w = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    img = img[y1:y2, x1:x2]
            except (KeyError, ValueError) as e:
                logger.debug(f"ROI 解析失败 {filename}: {e}，使用整图")

            # resize
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            class_id = int(row.get("ClassId", -1))

            images.append(img)
            labels.append(class_id)
            class_ids.append(class_id)

    logger.info(
        f"测试集加载完成：成功 {len(images)} 张，失败 {bad_count} 张，"
        f"类别数 {len(set(labels))}。"
    )
    return images, labels, class_ids


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float | None = None,
    random_state: int = 42,
) -> tuple:
    """按 stratify 切分数据集，并返回 StandardScaler 拟合后的结果。

    先 train/test 切分；若 val_size 非空，则从 train 中再切出 val
    （val_size 比例相对于原始 train 计算）。

    参数
    ----
    X : np.ndarray
        特征矩阵，shape (N, D)。
    y : np.ndarray
        标签向量，shape (N,)。
    test_size : float, 默认 0.2
        测试集比例。
    val_size : float | None, 默认 None
        验证集比例（相对于 train 的比例）。None 表示不切验证集。
    random_state : int, 默认 42
        随机种子。

    返回
    ----
    若 val_size 为 None：
        (scaler, X_tr_s, X_te_s, y_tr, y_te)
    若 val_size 非 None：
        (scaler, X_tr_s, X_val_s, X_te_s, y_tr, y_val, y_te)

    scaler 仅在 X_tr 上 fit，然后 transform 所有子集。
    """
    # 先切 train / test
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if val_size is not None:
        # 从 train 中切出 val：val_size 是相对 train 的比例
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=random_state
        )

    # StandardScaler 仅在 train 上 fit
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    if val_size is not None:
        X_val_s = scaler.transform(X_val)
        return scaler, X_tr_s, X_val_s, X_te_s, y_tr, y_val, y_te

    return scaler, X_tr_s, X_te_s, y_tr, y_te
