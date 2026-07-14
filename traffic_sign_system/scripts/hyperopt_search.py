"""贝叶斯超参搜索（基于 scikit-optimize）。

目标：在 GTSRB 数据集上，为 SVM + HOG/HSV 联合特征搜索最优超参组合。
搜索维度：
- C             : SVM 正则化参数（log 分布）
- gamma_type    : 'scale' | 'auto'
- gamma_value   : 仅当 gamma_type=='auto' 时生效
- img_size      : HOG/HSV 输入图像边长
- h_bins / s_bins: HSV 直方图分箱数

目标函数：3-fold StratifiedKFold 的 mean accuracy。

产出
----
- 控制台打印 best params + best score
- ``models/artifacts/hyperopt_history.json`` 记录每轮 (params, score)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args

from traffic_sign_system.config.settings import (
    DATASET_DIR,
    MODEL_ARTIFACTS_DIR,
    RANDOM_STATE,
)
from traffic_sign_system.data_processing.data_loader import load_train_data
from traffic_sign_system.features.feature_fusion import FeatureBuilder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 搜索空间
# ─────────────────────────────────────────────────────────────────────────────
SPACE = [
    Real(0.1, 100, name="C", prior="log-uniform"),
    Categorical(["scale", "auto"], name="gamma_type"),
    Real(1e-4, 1.0, name="gamma_value", prior="log-uniform"),
    Integer(64, 128, name="img_size"),
    Integer(4, 16, name="h_bins"),
    Integer(4, 16, name="s_bins"),
]


def _resolve_train_dir(value: Path | None) -> Path:
    if value is not None:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"训练目录不存在: {path}")
        return path
    candidates = (DATASET_DIR / "Train", DATASET_DIR / "train" / "Train")
    for candidate in candidates:
        if candidate.exists() and any(
            child.is_dir() and child.name.isdigit() for child in candidate.iterdir()
        ):
            return candidate
    raise FileNotFoundError("未找到 GTSRB 训练数据目录")


def _extract_features(
    images: list[np.ndarray],
    labels: np.ndarray,
    *,
    img_size: int,
    h_bins: int,
    s_bins: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """按指定参数构建特征矩阵。"""
    builder = FeatureBuilder(mode=mode, img_size=img_size, h_bins=h_bins, s_bins=s_bins)
    feats = builder.extract_batch(images).astype(np.float32, copy=False)
    return feats, labels


def _cv_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    gamma: str | float,
    n_splits: int,
    random_state: int,
) -> float:
    """3-fold StratifiedKFold mean accuracy。"""
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores: list[float] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        clf = SVC(C=C, kernel="rbf", gamma=gamma, probability=False,
                  class_weight="balanced", random_state=random_state)
        clf.fit(X[tr_idx], y[tr_idx])
        acc = float(clf.score(X[va_idx], y[va_idx]))
        scores.append(acc)
        logger.info(
            "  fold %d/%d acc=%.4f", fold_idx + 1, n_splits, acc
        )
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# 目标函数
# ─────────────────────────────────────────────────────────────────────────────
def make_objective(
    images: list[np.ndarray],
    labels: np.ndarray,
    *,
    mode: str,
    n_splits: int,
    random_state: int,
    feature_cache: dict[tuple[int, int, int], np.ndarray],
):
    """返回一个 ``@use_named_args`` 装饰的 objective。

    feature_cache 在不同 (img_size, h_bins, s_bins) 之间共享特征矩阵，
    避免重复抽取；HOG/HSV 抽取本身在 5w 张图上需 10+ 分钟。
    """

    @use_named_args(SPACE)
    def objective(
        C: float,
        gamma_type: str,
        gamma_value: float,
        img_size: int,
        h_bins: int,
        s_bins: int,
    ) -> float:
        # gamma 解析：scale/auto 用字符串，gamma_value 仅 auto 模式有效
        gamma = gamma_type if gamma_type == "scale" else float(gamma_value)

        cache_key = (int(img_size), int(h_bins), int(s_bins))
        if cache_key not in feature_cache:
            logger.info("抽取特征 img_size=%d, h_bins=%d, s_bins=%d ...", *cache_key)
            t0 = time.perf_counter()
            X, _ = _extract_features(
                images,
                labels,
                img_size=cache_key[0],
                h_bins=cache_key[1],
                s_bins=cache_key[2],
                mode=mode,
            )
            feature_cache[cache_key] = X
            logger.info("  抽取耗时 %.1fs, shape=%s", time.perf_counter() - t0, X.shape)
        X = feature_cache[cache_key]

        logger.info(
            "目标函数: C=%.4f, gamma=%s, img_size=%d, h_bins=%d, s_bins=%d",
            C, gamma, *cache_key,
        )
        t0 = time.perf_counter()
        score = _cv_accuracy(
            X,
            labels,
            C=float(C),
            gamma=gamma,
            n_splits=n_splits,
            random_state=random_state,
        )
        elapsed = time.perf_counter() - t0
        logger.info("  score=%.4f (%.1fs)", score, elapsed)
        # skopt 是最小化，所以取负值
        return -score

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> dict[str, Any]:
    train_dir = _resolve_train_dir(args.train_dir)
    logger.info("加载训练数据：%s", train_dir)
    images, labels, _, _ = load_train_data(train_dir)
    if not images:
        raise RuntimeError(f"训练目录 {train_dir} 中未发现有效图像")
    y = np.asarray(labels, dtype=np.int32)
    if args.max_samples is not None:
        # 与 scripts/train.py 同样的等量下采样
        rng = np.random.default_rng(args.random_state)
        if args.max_samples < len(np.unique(y)) * 3:
            raise ValueError("--max-samples 太小，至少要 类别数 * 3")
        per_class = args.max_samples // len(np.unique(y))
        keep = []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            take = min(per_class, len(idx))
            keep.extend(rng.choice(idx, size=take, replace=False).tolist())
        keep = np.asarray(keep)
        rng.shuffle(keep)
        images = [images[i] for i in keep]
        y = y[keep]
        logger.warning(
            "下采样到 %d 张，每类 ~%d 张", len(images), per_class
        )

    feature_cache: dict[tuple[int, int, int], np.ndarray] = {}
    objective = make_objective(
        images,
        y,
        mode=args.mode,
        n_splits=args.cv_folds,
        random_state=args.random_state,
        feature_cache=feature_cache,
    )

    logger.info(
        "开始贝叶斯搜索：n_calls=%d, n_initial_points=%d, mode=%s",
        args.n_calls, args.n_initial_points, args.mode,
    )
    result = gp_minimize(
        func=objective,
        dimensions=SPACE,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial_points,
        acq_func="gp_hedge",
        acq_optimizer="auto",
        random_state=args.random_state,
        verbose=False,
    )

    # 整理历史
    def _coerce(value):
        """将 numpy 标量/整数转 Python 原生类型，便于 JSON 序列化。"""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (np.ndarray,)):
            return [_coerce(v) for v in value.tolist()]
        return value

    history = {
        "iter": list(range(1, len(result.func_vals) + 1)),
        "params": [
            {dim.name: _coerce(value) for dim, value in zip(SPACE, x)}
            for x in result.x_iters
        ],
        "negative_score": [float(v) for v in result.func_vals],
        "score": [float(-v) for v in result.func_vals],
        "best_iter": int(np.argmin(result.func_vals) + 1),
        "best_params": {dim.name: _coerce(value) for dim, value in zip(SPACE, result.x)},
        "best_score": float(-result.fun),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    logger.info("搜索历史已写入：%s", out_path)

    print(f"best_params: {history['best_params']}")
    print(f"best_score: {history['best_score']:.4f}")
    print(f"best_iter: {history['best_iter']}")
    print(f"history_path: {out_path}")
    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="贝叶斯超参搜索（SVM + HOG/HSV 联合特征）"
    )
    parser.add_argument("--mode", choices=FeatureBuilder.VALID_MODES, default="hog+hsv")
    parser.add_argument("--n-calls", type=int, default=30, help="贝叶斯迭代次数")
    parser.add_argument("--n-initial-points", type=int, default=10,
                        help="初始随机采样数（用于填充 GP 先验）")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="每类等量下采样，加速搜索")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_ARTIFACTS_DIR / "hyperopt_history.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    if args.n_calls < 1:
        raise ValueError("--n-calls 必须 >= 1")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds 必须 >= 2")
    run(args)


if __name__ == "__main__":
    main()