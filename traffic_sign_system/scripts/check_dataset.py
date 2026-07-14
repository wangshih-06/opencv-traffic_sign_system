"""独立运行的数据集检查脚本。

用法（项目根目录下）：
    python -m traffic_sign_system.scripts.check_dataset

或直接：
    python scripts/check_dataset.py
"""

import logging
import sys
from pathlib import Path

# ── 让上层 traffic_sign_system 包可被导入 ──────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from config.labels import load_labels, get_all_labels
from data_processing.data_loader import load_train_data
from data_processing.dataset_statistics import summarize

# ── 日志配置 ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # 路径（相对于项目根目录）
    # GTSRB 实际结构: dataset/train/Train/<class_id>/*.png
    DATASET_ROOT = ROOT / "dataset" / "train" / "Train"
    LABELS_CSV   = ROOT / "dataset" / "labels.csv"

    logger.info(f"数据集根目录: {DATASET_ROOT.resolve()}")
    logger.info(f"标签文件    : {LABELS_CSV.resolve()}")

    # 1. 加载标签
    logger.info("加载标签映射 …")
    load_labels(LABELS_CSV)          # 用 GTSRB 内置名称（CSV 为空/缺省时 fallback）
    label_map = get_all_labels()
    logger.info(f"共 {len(label_map)} 个类别已加载。")

    # 2. 扫描图像
    logger.info("扫描图像文件 …（可能需要几秒）")
    images, labels, class_ids, bad_log = load_train_data(DATASET_ROOT)

    # 3. 统计
    logger.info("生成分类统计 …")
    df = summarize(class_ids, label_map)

    # 4. 汇总打印
    print("\n" + "== " * 20)
    print(f"  成功加载图像 : {len(images)} 张")
    print(f"  损坏/跳过图像: {len(bad_log)} 张")
    print(f"  类别总数     : {len(df)}")
    if len(df) > 0:
        print(f"  样本数范围   : {df['count'].min()} ~ {df['count'].max()}")
    print("== " * 20)

    if bad_log:
        print("\n前 10 条损坏记录:")
        for item in bad_log[:10]:
            print(f"  [{item['class_id']}] {Path(item['path']).name}  →  {item['reason']}")

    png_path = ROOT / "models" / "artifacts" / "class_distribution.png"
    print(f"\n柱状图已保存: {png_path.resolve()}")


if __name__ == "__main__":
    main()
