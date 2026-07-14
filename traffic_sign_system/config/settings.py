"""集中管理所有路径、HOG 参数、SVM 超参、数据划分比例等配置。"""

from dataclasses import dataclass, field
from pathlib import Path

# ──────────────────────────────────────────────
# 项目根目录
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent.resolve()

# ──────────────────────────────────────────────
# 子目录路径
# ──────────────────────────────────────────────
CONFIG_DIR = ROOT_DIR / "config"
DATASET_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
MODEL_ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"
DATA_PROCESSING_DIR = ROOT_DIR / "data_processing"
FEATURES_DIR = ROOT_DIR / "features"
EVALUATION_DIR = ROOT_DIR / "evaluation"
RECOGNITION_DIR = ROOT_DIR / "recognition"
UI_DIR = ROOT_DIR / "ui"

# ──────────────────────────────────────────────
# 图像预处理
# ──────────────────────────────────────────────
IMG_SIZE = 64  # 统一 resize 为 IMG_SIZE × IMG_SIZE

# ──────────────────────────────────────────────
# HOG 特征提取参数
# ──────────────────────────────────────────────
HOG_WIN_SIZE = (64, 64)
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_NBINS = 9

# ──────────────────────────────────────────────
# SVM 超参
# ──────────────────────────────────────────────
SVM_C = 10.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
SVM_PROBABILITY = True
SVM_RANDOM_STATE = 42

# ──────────────────────────────────────────────
# 数据划分
# ──────────────────────────────────────────────
TEST_SIZE = 0.2
VAL_SIZE = 0.15
RANDOM_STATE = 42
