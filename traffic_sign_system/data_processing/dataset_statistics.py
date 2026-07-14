"""数据集统计：生成分类分布 DataFrame 与柱状图。"""

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def summarize(
    class_ids: list[int],
    label_map: dict[int, str],
    output_png: Path | None = None,
) -> pd.DataFrame:
    """
    统计各类别样本数，输出 DataFrame 并绘制分布柱状图。

    参数
    ----
    class_ids : list[int]
        每张图像对应的 ClassId 列表（长度 = 样本总数）。
    label_map : dict[int, str]
        {class_id: name} 标签映射。
    output_png : Path | None, 可选
        柱状图保存路径，默认保存到 models/artifacts/class_distribution.png。

    返回
    ----
    pd.DataFrame
        列: class_id(int), name(str), count(int)
        按 count 降序排列。
    """
    if not class_ids:
        logger.warning("class_ids 为空，返回空 DataFrame。")
        df = pd.DataFrame(columns=["class_id", "name", "count"])
        print(df)
        return df

    # 统计
    unique, counts = np.unique(class_ids, return_counts=True)
    records = []
    for cid, cnt in zip(unique, counts):
        records.append({
            "class_id": int(cid),
            "name": label_map.get(int(cid), "Unknown"),
            "count": int(cnt),
        })

    df = pd.DataFrame(records).sort_values("count", ascending=False).reset_index(drop=True)

    # 打印
    print(f"\n{'='*50}")
    print(f"数据集统计：共 {len(class_ids)} 张图像，{len(df)} 个类别")
    print(f"{'='*50}")
    print(df.to_string(index=False))
    print(f"{'='*50}\n")

    # 绘图
    if output_png is None:
        output_png = (
            Path(__file__).parent.parent
            / "models"
            / "artifacts"
            / "class_distribution.png"
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(df)))
    bars = ax.bar(df["name"].astype(str), df["count"], color=colors)

    ax.set_xlabel("Traffic Sign Class", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("GTSRB Class Distribution", fontsize=14)
    plt.xticks(rotation=75, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    # 在柱顶标注数量
    for bar, cnt in zip(bars, df["count"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(cnt),
            ha="center",
            va="bottom",
            fontsize=6,
        )

    plt.tight_layout()
    plt.savefig(output_png, dpi=120)
    plt.close()
    logger.info(f"类别分布图已保存至: {output_png.resolve()}")

    return df
