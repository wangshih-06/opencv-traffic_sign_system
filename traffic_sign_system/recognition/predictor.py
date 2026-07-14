"""Unified single-image prediction from a saved model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from traffic_sign_system.config.labels import GTSRB_LABELS
from traffic_sign_system.features.feature_fusion import FeatureBuilder
from traffic_sign_system.models.model_manager import load_bundle


class Predictor:
    """Connect bundle loading, feature extraction, scaling, and classification.

    支持单图预测与批量预测；批量预测一次性走完特征抽取→scaler→predict，
    通常比循环 predict 快 3~5 倍（HOG 模式下尤其明显）。

    参数
    ----
    bundle_path : Path | str
        ``.joblib`` 模型包路径。
    use_cache : bool, 默认 True
        是否启用基于图像字节的 LRU 缓存；视频/摄像头场景下，相邻帧
        的预测结果高度相似，命中率通常 > 80%。
    cache_maxsize : int, 默认 512
        缓存条目上限；超出时按 FIFO 淘汰最早的条目（dict 维持插入顺序）。
    """

    _FEATURE_CONFIG_KEYS = {"mode", "img_size", "h_bins", "s_bins"}

    def __init__(
        self,
        bundle_path: Path | str,
        *,
        use_cache: bool = True,
        cache_maxsize: int = 512,
    ):
        if cache_maxsize < 1:
            raise ValueError("cache_maxsize 必须 >= 1")
        self.bundle_path = Path(bundle_path)
        bundle = load_bundle(self.bundle_path)
        self.classifier = bundle["classifier"]
        self.scaler = bundle["scaler"]
        bundle_label_map = {
            int(class_id): str(class_name)
            for class_id, class_name in bundle["label_map"].items()
        }
        # GTSRB bundles use numeric class IDs 0-42. Keep the classifier and
        # scaler untouched, but present the canonical Chinese names at runtime
        # so existing trained bundles do not need to be retrained.
        if set(bundle_label_map) == set(GTSRB_LABELS):
            self.label_map = dict(GTSRB_LABELS)
        else:
            self.label_map = bundle_label_map
        self.feature_config = dict(bundle["feature_config"])
        self.summary = dict(bundle["summary"])

        unsupported = set(self.feature_config) - self._FEATURE_CONFIG_KEYS
        if unsupported:
            raise ValueError(
                "Unsupported FeatureBuilder configuration keys in bundle: "
                f"{sorted(unsupported)}"
            )
        self.builder = FeatureBuilder(**self.feature_config)
        # FeatureBuilder owns the training-time Preprocessor used by HOG modes.
        # It is exposed here for inspection; HSV-only mode intentionally has none.
        self.preprocessor = self.builder.prep_gray
        self.feature_dim = int(self.builder.feature_dim())
        self._validate_feature_dimensions()

        # ── 缓存 ─────────────────────────────────────────────────────────
        self.use_cache = bool(use_cache)
        self.cache_maxsize = int(cache_maxsize)
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {} if self.use_cache else None  # type: ignore[assignment]
        self.cache_hits = 0
        self.cache_misses = 0

    def _validate_feature_dimensions(self) -> None:
        """Reject bundles whose scaler/model dimensions differ from the builder."""
        scaler_mean = getattr(self.scaler, "mean_", None)
        if scaler_mean is None:
            raise ValueError(
                "Bundle scaler has no fitted mean_; expected a fitted StandardScaler"
            )
        scaler_mean = np.asarray(scaler_mean)
        if scaler_mean.ndim != 1 or scaler_mean.shape[0] != self.feature_dim:
            raise ValueError(
                "Feature dimension mismatch: FeatureBuilder creates "
                f"{self.feature_dim} features, but scaler.mean_ has shape "
                f"{scaler_mean.shape}"
            )

        scaler_dim = getattr(self.scaler, "n_features_in_", self.feature_dim)
        if int(scaler_dim) != self.feature_dim:
            raise ValueError(
                "Feature dimension mismatch: FeatureBuilder creates "
                f"{self.feature_dim} features, but scaler expects {scaler_dim}"
            )

        summary_dim = self.summary.get("feature_dim")
        if summary_dim is not None and int(summary_dim) != self.feature_dim:
            raise ValueError(
                "Feature dimension mismatch: FeatureBuilder creates "
                f"{self.feature_dim} features, but bundle summary records "
                f"{summary_dim}"
            )

        classifier_dim = getattr(self.classifier, "n_features_in_", None)
        if classifier_dim is not None and int(classifier_dim) != self.feature_dim:
            raise ValueError(
                "Feature dimension mismatch: FeatureBuilder creates "
                f"{self.feature_dim} features, but classifier expects "
                f"{classifier_dim}"
            )

    @staticmethod
    def _validate_image(img_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(img_bgr, np.ndarray):
            raise TypeError(
                f"img_bgr must be a numpy.ndarray; got {type(img_bgr).__name__}"
            )
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(
                "img_bgr must have shape (height, width, 3) in BGR order; "
                f"got {img_bgr.shape}"
            )
        if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
            raise ValueError("img_bgr must not be empty")
        if not np.issubdtype(img_bgr.dtype, np.number):
            raise TypeError(f"img_bgr must have a numeric dtype; got {img_bgr.dtype}")
        if not np.isfinite(img_bgr).all():
            raise ValueError("img_bgr contains NaN or infinite values")

        if img_bgr.dtype == np.uint8:
            return img_bgr
        # OpenCV feature extractors in this project are trained on uint8 images.
        return np.clip(img_bgr, 0, 255).astype(np.uint8)

    def _prediction_confidence(self, X_scaled: np.ndarray, class_id: int) -> float | None:
        if not hasattr(self.classifier, "predict_proba"):
            return None

        probabilities = np.asarray(self.classifier.predict_proba(X_scaled))
        classes = np.asarray(getattr(self.classifier, "classes_", []))
        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise ValueError(
                "classifier.predict_proba returned an invalid shape: "
                f"{probabilities.shape}"
            )
        if classes.ndim != 1 or len(classes) != probabilities.shape[1]:
            raise ValueError(
                "Classifier probability columns do not align with classifier.classes_"
            )

        matches = np.flatnonzero(classes == class_id)
        if len(matches) != 1:
            raise ValueError(
                f"Predicted class {class_id} is missing or duplicated in classifier.classes_"
            )
        confidence = float(probabilities[0, int(matches[0])])
        if not np.isfinite(confidence):
            raise ValueError("Classifier returned a non-finite prediction probability")
        return confidence

    def predict(self, img_bgr: np.ndarray) -> dict[str, Any]:
        """Predict one BGR image and return id, display name, and confidence.

        启用缓存时，相同字节内容的图像直接返回缓存结果，避免重复的
        特征抽取 + scaler + predict 流程。
        """
        image = self._validate_image(img_bgr)
        if self._cache is not None:
            cache_key = (self._img_hash(image), self._preprocess_cache_token())
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return dict(cached)
            self.cache_misses += 1

        result = self._predict_uncached(image)

        if self._cache is not None:
            if len(self._cache) >= self.cache_maxsize:
                # FIFO 淘汰最早插入的条目（dict 维持插入顺序）
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = dict(result)
        return result

    def _predict_uncached(self, image: np.ndarray) -> dict[str, Any]:
        """执行单张预测，不读/写缓存。"""
        features = np.asarray(self.builder.extract_one(image), dtype=np.float32)
        if features.ndim != 1 or features.shape[0] != self.feature_dim:
            raise ValueError(
                "Extracted feature dimension mismatch: expected "
                f"{self.feature_dim}, got shape {features.shape}"
            )

        X_scaled = np.asarray(self.scaler.transform(features[None, :]))
        if X_scaled.shape != (1, self.feature_dim):
            raise ValueError(
                "Scaler returned an unexpected shape: "
                f"expected (1, {self.feature_dim}), got {X_scaled.shape}"
            )

        prediction = np.asarray(self.classifier.predict(X_scaled)).reshape(-1)
        if len(prediction) != 1:
            raise ValueError(
                f"Classifier returned {len(prediction)} predictions for one image"
            )
        class_id = int(prediction[0])
        class_name = self.label_map.get(class_id, str(class_id))
        confidence = self._prediction_confidence(X_scaled, class_id)
        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
        }

    def predict_batch(self, imgs_bgr: list[np.ndarray]) -> list[dict[str, Any]]:
        """批量预测：一次性抽取特征 → scaler → predict，比逐张快 3-5x。

        参数
        ----
        imgs_bgr : list[np.ndarray]
            长度 ≥ 1 的 BGR 图像列表，每张 shape=(H,W,3)。

        返回
        ----
        list[dict]
            每张图像一个 ``{class_id, class_name, confidence}``。

        备注
        ----
        - 当前实现不写入缓存（批量场景下命中率低且 key 计算昂贵）。
        - 当分类器支持 ``predict_proba`` 时只对整批计算一次，比单张循环
          调用 ``predict_proba`` 节省大量时间（SVM 内部有 5 折校准开销）。
        """
        if not isinstance(imgs_bgr, (list, tuple)):
            raise TypeError(
                f"imgs_bgr must be a list/tuple; got {type(imgs_bgr).__name__}"
            )
        if len(imgs_bgr) == 0:
            return []

        images = [self._validate_image(img) for img in imgs_bgr]
        # 1. 一次性抽取特征矩阵 (N, D)
        X = self.builder.extract_batch(images).astype(np.float32, copy=False)
        if X.ndim != 2 or X.shape[1] != self.feature_dim:
            raise ValueError(
                "Extracted feature dimension mismatch: expected "
                f"{self.feature_dim}, got shape {X.shape}"
            )
        # 2. scaler 一次性处理整批
        X_scaled = np.asarray(self.scaler.transform(X))
        # 3. classifier 一次性预测整批
        y_preds = np.asarray(self.classifier.predict(X_scaled)).reshape(-1)

        # 4. 置信度：仅当分类器支持时，一次性算 (N, n_classes) proba
        proba_all: np.ndarray | None = None
        if hasattr(self.classifier, "predict_proba"):
            proba_all = np.asarray(self.classifier.predict_proba(X_scaled))

        results: list[dict[str, Any]] = []
        for i, y_pred in enumerate(y_preds):
            class_id = int(y_pred)
            confidence: float | None = None
            if proba_all is not None:
                classes = np.asarray(getattr(self.classifier, "classes_", []))
                matches = np.flatnonzero(classes == class_id)
                if len(matches) == 1:
                    confidence = float(proba_all[i, int(matches[0])])
            results.append({
                "class_id": class_id,
                "class_name": self.label_map.get(class_id, str(class_id)),
                "confidence": confidence,
            })
        return results

    def clear_cache(self) -> None:
        """清空缓存（命中率统计保留）。"""
        if self._cache is not None:
            self._cache.clear()

    def cache_stats(self) -> dict[str, int | float]:
        """返回缓存命中/未命中统计与命中率。"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": hit_rate,
            "size": 0 if self._cache is None else len(self._cache),
            "maxsize": self.cache_maxsize,
        }

    def _preprocess_cache_token(self) -> tuple[Any, ...]:
        """Return settings that can change features for identical image bytes."""
        preprocessor = self.preprocessor
        if preprocessor is None:
            return ("no_preprocessor",)
        adaptive = bool(getattr(preprocessor, "adaptive", False))
        runtime = getattr(preprocessor, "_runtime_params", {}) if adaptive else {}
        runtime_items = tuple(sorted((str(key), repr(value)) for key, value in runtime.items()))
        return ("adaptive", adaptive, runtime_items)

    @staticmethod
    def _img_hash(img: np.ndarray) -> int:
        """基于图像字节内容的哈希，用作缓存 key。

        对 uint8 BGR 图像使用 ``hash(tobytes())``，足够区分绝大多数
        视频相邻帧；同时保留 dtype/shape 在 key 中，避免不同尺寸
        图像巧合碰撞。
        """
        return hash((img.shape, img.dtype, img.tobytes()))
