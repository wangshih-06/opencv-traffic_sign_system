"""Soft-voting ensemble classifier (SVM + KNN + RF).

将若干个已训练好的 ``predict_proba``-支持模型做加权平均，按
``self.classes_`` 上的 argmax 得到最终预测。权重通常由验证集
accuracy 派生（参考 ``scripts/train.py`` 中的 ``--ensemble`` 流程），
也可以在 ``fit`` 时根据 ``sample_weight``-like 思路做更细粒度调整。

参考
----
- sklearn.ensemble.VotingClassifier(voting="soft")
- Louppe, "Understanding Random Forests" (博士论文)
"""

from __future__ import annotations

import logging
import time
from typing import Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """软投票集成：SVM + KNN + RF，按验证集 accuracy 加权投票。

    参数
    ----
    estimators : list[tuple[str, estimator]]
        ``[(name, fitted_or_unfitted_clf), ...]``；每个分类器需提供
        ``predict_proba``，否则在 ``predict_proba`` 中会被忽略并打 WARNING。
    weights : list[float] | None, 默认 None
        与 ``estimators`` 等长的正权重；为 None 时按等权平均。
    classes_ : np.ndarray | None, 默认 None
        类索引数组；为 None 时在 ``fit`` 后从首个具备 ``classes_``
        属性的估计器推断。

    备注
    ----
    - ``fit`` / ``predict`` / ``predict_proba`` 接口与 sklearn 一致，
      可直接被 ``joblib.dump`` / ``model_manager.save_bundle`` 序列化。
    - 集成预测的耗时近似于 ``sum(predict_proba)``；SVM 的 proba
      来自 5 折 Platt 校准，是慢项。
    """

    def __init__(
        self,
        estimators: Sequence[tuple[str, object]],
        weights: Sequence[float] | None = None,
        classes_: np.ndarray | None = None,
    ):
        if not estimators:
            raise ValueError("estimators must be a non-empty list of (name, clf) tuples")
        names = [name for name, _ in estimators]
        if len(set(names)) != len(names):
            raise ValueError(f"estimator names must be unique; got {names}")
        if weights is not None:
            if len(weights) != len(estimators):
                raise ValueError(
                    f"weights length ({len(weights)}) must match estimators ({len(estimators)})"
                )
            if any(w <= 0 for w in weights):
                raise ValueError("all weights must be > 0")
        self.estimators = list(estimators)
        self.weights = np.asarray(weights, dtype=np.float64) if weights is not None else None
        self.classes_: np.ndarray | None = (
            np.asarray(classes_) if classes_ is not None else None
        )
        self._fitted = False

    # ──────────────────────────────────────────────────────────────────────
    # sklearn 风格接口
    # ──────────────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleClassifier":
        """依次 fit 所有子估计器（每个必须支持 ``fit``）。"""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape={X.shape}")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        for name, clf in self.estimators:
            logger.info("[ensemble] fitting %s ...", name)
            t0 = time.perf_counter()
            clf.fit(X, y)
            logger.info("[ensemble] %s fitted in %.2fs", name, time.perf_counter() - t0)
        # 推断 classes_
        for _, clf in self.estimators:
            if hasattr(clf, "classes_"):
                self.classes_ = np.asarray(getattr(clf, "classes_"))
                break
        if self.classes_ is None:
            self.classes_ = np.unique(y)
        self._fitted = True
        return self

    def _collect_probas(self, X: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
        """返回每个子分类器对 X 的 predict_proba 与对应的 name。"""
        probas: list[np.ndarray] = []
        used_names: list[str] = []
        for name, clf in self.estimators:
            if not hasattr(clf, "predict_proba"):
                logger.warning(
                    "[ensemble] estimator %s lacks predict_proba — skipped", name
                )
                continue
            try:
                proba = clf.predict_proba(X)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[ensemble] %s.predict_proba failed: %s", name, exc)
                continue
            # 对齐到 self.classes_
            if hasattr(clf, "classes_"):
                clf_classes = np.asarray(getattr(clf, "classes_"))
                if not np.array_equal(clf_classes, self.classes_):
                    aligned = np.zeros((proba.shape[0], len(self.classes_)), dtype=proba.dtype)
                    col_map = {int(c): i for i, c in enumerate(clf_classes)}
                    for j, c in enumerate(self.classes_):
                        if int(c) in col_map:
                            aligned[:, j] = proba[:, col_map[int(c)]]
                    proba = aligned
            probas.append(proba)
            used_names.append(name)
        return probas, used_names

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EnsembleClassifier must be fit before predict_proba")
        if self.classes_ is None:
            raise RuntimeError("classes_ is not set; cannot compute proba")
        probas, used_names = self._collect_probas(X)
        if not probas:
            raise RuntimeError("no estimator produced probas")

        # 重新计算与有效估计器数量匹配的权重
        if self.weights is not None and len(self.weights) == len(probas):
            weights = self.weights
        elif self.weights is not None:
            # 原始权重与有效数量不一致 → 退回等权
            logger.warning(
                "[ensemble] weights length %d != active estimators %d; falling back to equal weights",
                len(self.weights),
                len(probas),
            )
            weights = np.ones(len(probas), dtype=np.float64)
        else:
            weights = np.ones(len(probas), dtype=np.float64)

        weights = weights / weights.sum()
        avg = np.average(np.stack(probas, axis=0), axis=0, weights=weights)
        # 数值安全：clip 到 [0,1] 并归一化（防止浮点漂移）
        avg = np.clip(avg, 0.0, 1.0)
        row_sum = avg.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return avg / row_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        if self.classes_ is None:
            raise RuntimeError("classes_ is not set")
        idx = np.argmax(probas, axis=1)
        return self.classes_[idx]

    # ──────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────
    def set_weights_from_scores(self, scores: Sequence[float]) -> None:
        """根据验证集 accuracy 列表更新权重（自动归一化）。"""
        if len(scores) != len(self.estimators):
            raise ValueError(
                f"scores length ({len(scores)}) must match estimators ({len(self.estimators)})"
            )
        cleaned = [max(float(s), 1e-6) for s in scores]
        arr = np.asarray(cleaned, dtype=np.float64)
        arr = arr / arr.sum()
        self.weights = arr
        logger.info("[ensemble] weights updated from scores: %s", dict(zip(
            [n for n, _ in self.estimators], cleaned
        )))

    def get_params(self, deep: bool = True) -> dict:
        return {
            "estimators": self.estimators,
            "weights": None if self.weights is None else self.weights.tolist(),
            "classes_": None if self.classes_ is None else self.classes_.tolist(),
        }

    def __repr__(self) -> str:
        names = [name for name, _ in self.estimators]
        return (
            f"EnsembleClassifier(estimators={names}, "
            f"weights={None if self.weights is None else self.weights.tolist()})"
        )


def build_default_estimators(
    svm_C: float = 10.0,
    svm_kernel: str = "rbf",
    svm_gamma: str | float = "scale",
    svm_class_weight: str | dict | None = "balanced",
    knn_neighbors: int = 5,
    rf_estimators: int = 200,
    random_state: int = 42,
    n_jobs: int = 1,
) -> list[tuple[str, object]]:
    """构造 (SVM, KNN, RF) 三个未训练的 sklearn 估计器。

    返回
    ----
    list[tuple[str, estimator]]
        ``[("svm", SVC(...)), ("knn", KNN(...)), ("rf", RF(...))]``。
    """
    # 局部 import 避免循环依赖
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    svm = SVC(
        C=svm_C,
        kernel=svm_kernel,
        gamma=svm_gamma,
        probability=True,
        class_weight=svm_class_weight,
        random_state=random_state,
    )
    knn = KNeighborsClassifier(n_neighbors=knn_neighbors, n_jobs=-1)
    rf = RandomForestClassifier(
        n_estimators=rf_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    return [("svm", svm), ("knn", knn), ("rf", rf)]