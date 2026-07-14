"""特征构建脚本：从原始图像提取 HOG/HSV/HOG+HSV 特征并保存。

用法（项目根目录下）：
    python -m traffic_sign_system.scripts.build_features --mode hog
    python -m traffic_sign_system.scripts.build_features --mode hsv
    python -m traffic_sign_system.scripts.build_features --mode hog+hsv

输出：
    models/artifacts/features_<mode>.npz
    含 X_train, y_train, X_test, y_test, scaler, label_map, feature_config
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ── 让上层 traffic_sign_system 包可被导入 ──────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config.settings import (
    DATASET_DIR,
    IMG_SIZE,
    MODEL_ARTIFACTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from data_processing.data_loader import load_train_data, load_test_data, stratified_split
from features.feature_fusion import FeatureBuilder

# ── 日志配置 ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_features(mode: str) -> None:
    """主流程：加载数据 → 提取特征 → 分割 → 缩放 → 保存。"""

    # ── 1. 路径 ────────────────────────────────────────────────────
    train_dir = DATASET_DIR / "train" / "Train"
    test_dir = DATASET_DIR / "test" / "Test"
    test_csv = test_dir / "GT-final_test.csv"

    output_dir = MODEL_ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. 加载数据 ────────────────────────────────────────────────
    logger.info(f"加载训练集: {train_dir}")
    images, labels, _, bad_log = load_train_data(train_dir)
    if not images:
        logger.error("训练集为空，退出。")
        return

    # 构建标签映射（class_id → class_id，这里用整数索引）
    unique_ids = sorted(set(labels))
    label_map = {i: i for i in unique_ids}

    y_train_all = np.array(labels, dtype=np.int32)

    # ── 3. 加载测试集 ──────────────────────────────────────────────
    test_images: list[np.ndarray] = []
    y_test: np.ndarray = np.array([], dtype=np.int32)

    if test_dir.exists() and test_csv.exists():
        logger.info(f"加载测试集: {test_dir} + {test_csv}")
        test_images, test_labels, _ = load_test_data(
            test_dir, test_csv, img_size=IMG_SIZE
        )
        if test_images:
            y_test = np.array(test_labels, dtype=np.int32)
    else:
        logger.warning(
            f"测试集路径不存在，将使用 train_test_split 切分。"
            f"  test_dir={test_dir}, csv={test_csv}"
        )

    # ── 4. 创建 FeatureBuilder ─────────────────────────────────────
    builder = FeatureBuilder(mode=mode, img_size=IMG_SIZE)
    logger.info(f"FeatureBuilder: {builder}")

    # ── 5. 提取训练特征 ────────────────────────────────────────────
    t0 = time.time()
    logger.info(f"提取训练集特征 (N={len(images)}, mode={mode}) …")
    X_train_all = builder.extract_batch(images)
    logger.info(
        f"训练特征提取完成: shape={X_train_all.shape}, "
        f"耗时 {time.time() - t0:.1f}s"
    )

    # ── 6. 提取测试特征 ────────────────────────────────────────────
    if test_images:
        t0 = time.time()
        logger.info(f"提取测试集特征 (N={len(test_images)}, mode={mode}) …")
        X_test = builder.extract_batch(test_images)
        logger.info(
            f"测试特征提取完成: shape={X_test.shape}, "
            f"耗时 {time.time() - t0:.1f}s"
        )
        # 标准化（仅在 train 上 fit）
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_all)
        X_test = scaler.transform(X_test)
    else:
        # 无独立测试集 → 从 train 中 stratify 切分
        logger.info("使用 stratified_split 从训练集切分 80/20 …")
        result = stratified_split(
            X_train_all, y_train_all,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        scaler, X_train, X_test, y_train, y_test = result

    # ── 7. 保存 ────────────────────────────────────────────────────
    out_path = output_dir / f"features_{mode}.npz"
    # StandardScaler 无法直接存入 npz，序列化为 pickle bytes
    import pickle

    scaler_bytes = pickle.dumps(scaler)

    np.savez(
        out_path,
        X_train=X_train,
        y_train=y_train_all if test_images else y_train,
        X_test=X_test,
        y_test=y_test,
        scaler_bytes=scaler_bytes,
        label_map=json.dumps(label_map),
        feature_config=json.dumps(builder.config),
    )
    logger.info(f"已保存: {out_path}")
    logger.info(
        f"  X_train shape: {X_train.shape}, "
        f"X_test shape: {X_test.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 HOG/HSV/HOG+HSV 特征并保存为 .npz"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hog+hsv",
        choices=["hog", "hsv", "hog+hsv"],
        help="特征模式 (默认: hog+hsv)",
    )
    args = parser.parse_args()
    build_features(args.mode)


if __name__ == "__main__":
    main()
