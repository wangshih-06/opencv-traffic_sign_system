"""类别 ID → 名称映射。默认使用 GTSRB 内置 43 类名称，可通过 load_labels(csv_path) 从外部 CSV 覆盖。"""

import logging
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────
# GTSRB 官方 43 类标志名称（按 ClassId 0~42 顺序）
# ──────────────────────────────────────────────
GTSRB_LABELS: dict[int, str] = {
    0: "限速20公里/小时",
    1: "限速30公里/小时",
    2: "限速50公里/小时",
    3: "限速60公里/小时",
    4: "限速70公里/小时",
    5: "限速80公里/小时",
    6: "解除限速80公里/小时",
    7: "限速100公里/小时",
    8: "限速120公里/小时",
    9: "禁止超车",
    10: "禁止3.5吨以上车辆超车",
    11: "下一路口优先通行",
    12: "优先道路",
    13: "让行",
    14: "停车让行",
    15: "禁止车辆通行",
    16: "禁止3.5吨以上车辆通行",
    17: "禁止驶入",
    18: "注意危险",
    19: "向左急弯",
    20: "向右急弯",
    21: "连续弯路",
    22: "路面不平",
    23: "路面湿滑",
    24: "右侧道路变窄",
    25: "道路施工",
    26: "交通信号灯",
    27: "注意行人",
    28: "注意儿童",
    29: "注意自行车",
    30: "注意冰雪路面",
    31: "注意野生动物",
    32: "解除限速与超车限制",
    33: "前方右转",
    34: "前方左转",
    35: "直行",
    36: "直行或右转",
    37: "直行或左转",
    38: "靠右行驶",
    39: "靠左行驶",
    40: "环岛行驶",
    41: "解除禁止超车",
    42: "解除3.5吨以上车辆禁止超车",
}

# 默认使用 GTSRB 内置名称
DEFAULT_LABELS: dict[int, str] = dict(GTSRB_LABELS)

# 运行时标签映射（可被 load_labels 覆盖）
_labels: dict[int, str] = dict(GTSRB_LABELS)


def load_labels(csv_path: Optional[Path | str] = None) -> dict[int, str]:
    """
    读取 class_id,name 格式的标签 CSV 文件。

    参数
    ----
    csv_path : Path | str, 可选
        CSV 文件路径，格式: class_id,name。
        若为 None 或文件不存在，使用 GTSRB 内置 43 类名称。

    返回
    ----
    dict[int, str]
        {class_id: name} 映射。

    异常
    ----
    FileNotFoundError : csv_path 指定了文件但文件不存在时抛出。
    """
    global _labels

    if csv_path is None:
        _labels = dict(GTSRB_LABELS)
        logging.info("未指定标签文件，使用 GTSRB 内置 43 类名称。")
        return _labels

    csv_path = Path(csv_path)

    # 文件明确指定但不存在 → 友好警告 + fallback
    if not csv_path.exists():
        logging.warning(
            "标签文件不存在: %s\n"
            "   使用 GTSRB 内置 43 类名称。如需自定义标签，请创建 dataset/labels.csv。",
            csv_path.resolve(),
        )
        _labels = dict(GTSRB_LABELS)
        return _labels

    loaded: dict[int, str] = {}
    missing_header_warn = True
    with open(csv_path, "r", encoding="utf-8") as f:
        header = next(f, "").strip()
        if "class_id" not in header.lower() and "name" not in header.lower():
            if missing_header_warn:
                logging.warning(
                    "标签文件表头缺少 'class_id' 或 'name' 列，尝试直接解析…"
                )
                missing_header_warn = False

        for lineno, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                logging.warning(f"第 {lineno} 行格式异常，已跳过: {line}")
                continue
            try:
                cid = int(parts[0].strip())
                name = parts[1].strip()
                loaded[cid] = name
            except ValueError as ve:
                logging.warning(
                    f"第 {lineno} 行解析失败，已跳过: {line} （错误: {ve}）"
                )

    if loaded:
        _labels = loaded
        logging.info(f"已从 {csv_path.name} 加载 {len(_labels)} 条标签映射。")
    else:
        _labels = dict(GTSRB_LABELS)
        logging.warning(
            f"{csv_path.name} 为空或解析失败， fallback 到 GTSRB 内置标签。"
        )

    return _labels


def get_label(class_id: int) -> str:
    """根据 class_id 查找标签名称，未找到时返回空字符串。"""
    return _labels.get(class_id, "")


def get_all_labels() -> dict[int, str]:
    """返回当前全部标签映射的副本。"""
    return dict(_labels)


def num_classes() -> int:
    """返回当前已知类别总数。"""
    return len(_labels)
