"""ONNX-backed predictor with the same public surface as :class:`Predictor`.

This module is **lazy** about its dependencies: importing it does NOT require
``onnxruntime``. The runtime is imported on first predictor construction, so
projects that never instantiate :class:`OnnxPredictor` pay no cost.

The class mirrors :class:`traffic_sign_system.recognition.predictor.Predictor`'s
public attributes (``classifier``, ``scaler``, ``builder``, ``label_map``,
``feature_config``, ``feature_dim``, ``summary``, ``preprocessor``) and methods
(``predict``, ``predict_batch``, ``clear_cache``, ``cache_stats``) so callers
such as :func:`api.app._top_k` and :func:`ui.workers.compute_topk` continue to
work without modification. The trick is :class:`_OnnxProbaShim`, which wraps
the ONNX session in a sklearn-shaped estimator that exposes
``predict_proba`` and ``classes_``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from traffic_sign_system.config.labels import GTSRB_LABELS
from traffic_sign_system.features.feature_fusion import FeatureBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class OnnxPredictor:
    """ONNX-runtime-based predictor compatible with :class:`Predictor`.

    参数
    ----
    onnx_path : Path | str
        Path to the ``.onnx`` model file. The file's metadata must have been
        embedded by :func:`models.onnx_exporter.export_bundle_to_onnx`.
    use_cache : bool, 默认 True
        是否对单图推理启用 LRU 缓存(同 Predictor 语义)。
    cache_maxsize : int, 默认 512
        LRU 缓存容量上限。
    """

    _FEATURE_CONFIG_KEYS = {"mode", "img_size", "h_bins", "s_bins"}

    def __init__(
        self,
        onnx_path: Path | str,
        *,
        use_cache: bool = True,
        cache_maxsize: int = 512,
    ):
        if cache_maxsize < 1:
            raise ValueError("cache_maxsize 必须 >= 1")

        self.onnx_path = Path(onnx_path).resolve()
        metadata = self._load_metadata()
        self._raw_metadata = metadata
        self.label_map = self._resolve_label_map(metadata.get("label_map", {}))
        self.feature_config = dict(metadata.get("feature_config", {}))
        self.summary = dict(metadata.get("summary", {}))
        self.feature_dim = int(metadata.get("feature_dim", 0))

        unsupported = set(self.feature_config) - self._FEATURE_CONFIG_KEYS
        if unsupported:
            raise ValueError(
                "Unsupported FeatureBuilder configuration keys in ONNX bundle: "
                f"{sorted(unsupported)}"
            )

        # Reconstruct the same FeatureBuilder used at training time.
        self.builder = FeatureBuilder(**self.feature_config)
        self.preprocessor = self.builder.prep_gray

        # Reconstruct the StandardScaler from the embedded metadata.
        self.scaler = self._reconstruct_scaler(metadata)

        # Build the ort session(s) and the shim that mimics sklearn's API.
        self._session, self._ensemble_sessions = self._make_sessions(metadata)
        self.classifier = _OnnxProbaShim(
            session=self._session,
            ensemble_sessions=self._ensemble_sessions,
            label_map=self.label_map,
            scaler=self.scaler,
            builder=self.builder,
            is_ensemble=bool(metadata.get("is_ensemble", False)),
            ensemble_weights=metadata.get("ensemble_weights") or [],
        )

        # ── Cache (mirrors Predictor) ─────────────────────────────────
        self.use_cache = bool(use_cache)
        self.cache_maxsize = int(cache_maxsize)
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {} if self.use_cache else None
        self.cache_hits = 0
        self.cache_misses = 0

    # ── Public methods (Predictor-shaped) ──────────────────────────────

    def predict(self, img_bgr: np.ndarray) -> dict[str, Any]:
        """Predict one BGR image; same return shape as Predictor.predict."""
        image = self._validate_image(img_bgr)
        cache_key = (self._img_hash(image),)
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.cache_hits += 1
                return dict(cached)
            self.cache_misses += 1

        result = self._predict_uncached(image)

        if self._cache is not None:
            if len(self._cache) >= self.cache_maxsize:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = dict(result)
        return result

    def predict_batch(self, imgs_bgr: list[np.ndarray]) -> list[dict[str, Any]]:
        """Batch predict without cache (matches Predictor.predict_batch)."""
        if not isinstance(imgs_bgr, (list, tuple)):
            raise TypeError(
                f"imgs_bgr must be a list/tuple; got {type(imgs_bgr).__name__}"
            )
        if len(imgs_bgr) == 0:
            return []
        images = [self._validate_image(img) for img in imgs_bgr]
        features = np.asarray(
            self.builder.extract_batch(images), dtype=np.float32
        )
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Extracted feature dim mismatch: expected {self.feature_dim}, "
                f"got {features.shape}"
            )
        X_scaled = np.asarray(self.scaler.transform(features))
        probas = self.classifier.predict_proba(X_scaled)
        preds = np.argmax(probas, axis=1)
        classes = self.classifier.classes_
        results: list[dict[str, Any]] = []
        for i, class_id in enumerate(preds):
            cid = int(class_id)
            results.append(
                {
                    "class_id": cid,
                    "class_name": self.label_map.get(cid, str(cid)),
                    "confidence": float(probas[i, int(np.where(classes == cid)[0][0])]),
                }
            )
        return results

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()

    def cache_stats(self) -> dict[str, int | float]:
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

    # ── Internals ──────────────────────────────────────────────────────

    def _predict_uncached(self, image: np.ndarray) -> dict[str, Any]:
        features = np.asarray(
            self.builder.extract_one(image), dtype=np.float32
        )
        if features.shape[0] != self.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.feature_dim}, got {features.shape}"
            )
        X_scaled = np.asarray(self.scaler.transform(features[None, :]))
        probas = self.classifier.predict_proba(X_scaled)
        classes = self.classifier.classes_
        idx = int(np.argmax(probas[0]))
        class_id = int(classes[idx])
        confidence = float(probas[0, idx])
        return {
            "class_id": class_id,
            "class_name": self.label_map.get(class_id, str(class_id)),
            "confidence": confidence,
        }

    @staticmethod
    def _validate_image(img_bgr: np.ndarray) -> np.ndarray:
        if not isinstance(img_bgr, np.ndarray):
            raise TypeError(
                f"img_bgr must be a numpy.ndarray; got {type(img_bgr).__name__}"
            )
        if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            raise ValueError(
                f"img_bgr must have shape (H, W, 3); got {img_bgr.shape}"
            )
        if img_bgr.shape[0] == 0 or img_bgr.shape[1] == 0:
            raise ValueError("img_bgr must not be empty")
        if img_bgr.dtype == np.uint8:
            return img_bgr
        return np.clip(img_bgr, 0, 255).astype(np.uint8)

    @staticmethod
    def _img_hash(img: np.ndarray) -> int:
        return hash((img.shape, img.dtype, img.tobytes()))

    def _load_metadata(self) -> dict[str, Any]:
        try:
            from traffic_sign_system.models.onnx_exporter import read_onnx_metadata
        except ImportError as exc:  # pragma: no cover - circular safety
            raise ImportError(
                "onnx_exporter must be importable; ensure "
                "traffic_sign_system is on PYTHONPATH"
            ) from exc
        meta = read_onnx_metadata(self.onnx_path)
        if "feature_dim" not in meta or int(meta["feature_dim"]) <= 0:
            raise ValueError(
                f"ONNX bundle {self.onnx_path} is missing feature_dim metadata"
            )
        return meta

    @staticmethod
    def _resolve_label_map(label_map: dict[str, str]) -> dict[int, str]:
        out: dict[int, str] = {}
        for key, value in label_map.items():
            out[int(key)] = str(value)
        if set(out) == set(GTSRB_LABELS):
            return dict(GTSRB_LABELS)
        return out

    @staticmethod
    def _reconstruct_scaler(metadata: dict[str, Any]) -> "_NumpyStandardScaler":
        # Scaler params are stored at the top level of metadata (not inside
        # ``summary``) because the exporter reads them directly from the
        # joblib's fitted StandardScaler.
        mean = metadata.get("scaler_mean")
        scale = metadata.get("scaler_scale")
        # Backward compat: older bundles may have stored them in summary.
        if mean is None or scale is None:
            summary = metadata.get("summary") or {}
            mean = mean if mean is not None else summary.get("scaler_mean")
            scale = scale if scale is not None else summary.get("scaler_scale")
        if mean is None or scale is None:
            raise ValueError(
                "ONNX bundle is missing scaler_mean / scaler_scale; "
                "the bundle was exported without StandardScaler parameters."
            )
        mean_arr = np.asarray(mean, dtype=np.float64)
        scale_arr = np.asarray(scale, dtype=np.float64)
        if mean_arr.shape != scale_arr.shape:
            raise ValueError(
                f"scaler_mean and scaler_scale shape mismatch: "
                f"{mean_arr.shape} vs {scale_arr.shape}"
            )
        return _NumpyStandardScaler(mean=mean_arr, scale=scale_arr)

    def _make_sessions(
        self, metadata: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for OnnxPredictor; "
                "install with: pip install onnxruntime"
            ) from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1  # workers are already parallel
        options.inter_op_num_threads = 1
        session_kwargs = {
            "path_or_bytes": str(self.onnx_path),
            "sess_options": options,
            "providers": ["CPUExecutionProvider"],
        }
        main_session = ort.InferenceSession(**session_kwargs)

        ensemble_sessions: dict[str, Any] = {}
        if metadata.get("is_ensemble"):
            sub_paths = metadata.get("ensemble_sub_paths") or {}
            for name, filename in sub_paths.items():
                sub_path = self.onnx_path.with_name(filename)
                if not sub_path.is_file():
                    raise FileNotFoundError(
                        f"Ensemble sub-model file missing: {sub_path}"
                    )
                ensemble_sessions[name] = ort.InferenceSession(
                    path_or_bytes=str(sub_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
        return main_session, ensemble_sessions


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class _NumpyStandardScaler:
    """Minimal in-memory :class:`sklearn.preprocessing.StandardScaler` replacement."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        # `n_features_in_` is what newer sklearn uses; expose both for safety.
        self.n_features_in_ = int(self.mean_.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features but scaler expects {self.n_features_in_}"
            )
        safe_scale = np.where(self.scale_ == 0, 1.0, self.scale_)
        return ((arr - self.mean_) / safe_scale).astype(np.float32, copy=False)


class _OnnxProbaShim:
    """Sklearn-shaped wrapper around an ONNX Runtime session.

    Exposes ``predict_proba(X_scaled)`` and ``classes_`` so existing callers
    (SignDetector, _top_k, compute_topk) work without modification. For
    ensemble bundles, the shim runs each sub-session and returns a
    weighted average, matching EnsembleClassifier.predict_proba.
    """

    _LOCK = threading.Lock()  # ort sessions are not always thread-safe across calls

    def __init__(
        self,
        *,
        session: Any,
        ensemble_sessions: dict[str, Any],
        label_map: dict[int, str],
        scaler: _NumpyStandardScaler,
        builder: FeatureBuilder,
        is_ensemble: bool,
        ensemble_weights: list[float],
    ):
        self._session = session
        self._ensemble_sessions = ensemble_sessions
        self._label_map = label_map
        self._scaler = scaler
        self._builder = builder
        self._is_ensemble = bool(is_ensemble)
        self._ensemble_weights = (
            list(ensemble_weights) if ensemble_weights else []
        )
        # Sort class ids numerically for stable column ordering.
        self.classes_ = np.asarray(
            sorted(int(k) for k in label_map.keys()), dtype=np.int64
        )

    @property
    def is_ensemble(self) -> bool:
        return self._is_ensemble

    def predict_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        arr = np.asarray(X_scaled, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        n_samples = arr.shape[0]
        if not self._is_ensemble:
            return self._run_single_session(self._session, arr, n_samples)
        # Ensemble: run every sub-model, align by class ids, weighted average.
        weights = self._ensemble_weights or [1.0] * len(self._ensemble_sessions)
        weights_arr = np.asarray(weights, dtype=np.float64)
        if len(weights_arr) != len(self._ensemble_sessions):
            weights_arr = np.ones(len(self._ensemble_sessions), dtype=np.float64)
        weights_arr = weights_arr / weights_arr.sum()

        probas_stack: list[np.ndarray] = []
        for name, sess in self._ensemble_sessions.items():
            single = self._run_single_session(sess, arr, n_samples)
            probas_stack.append(single)
        avg = np.average(np.stack(probas_stack, axis=0), axis=0, weights=weights_arr)
        avg = np.clip(avg, 0.0, 1.0)
        row_sum = avg.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return avg / row_sum

    def _run_single_session(
        self, session: Any, arr: np.ndarray, n_samples: int
    ) -> np.ndarray:
        """Run a single ONNX session and return a (N, n_classes) probability matrix."""
        input_name = session.get_inputs()[0].name
        with self._LOCK:
            outputs = session.run(None, {input_name: arr})
        proba = self._extract_proba(outputs, n_samples)
        return self._align_columns(proba)

    def _extract_proba(self, outputs: list[Any], n_samples: int) -> np.ndarray:
        """Pull a (N, n_classes) probability tensor out of skl2onnx's outputs.

        skl2onnx (with ``zipmap=False``) produces a 2D float tensor named
        ``output_probability``. Older versions may produce a list-of-dict
        (``ZipMap``); fall back to a dict of class ids if so.
        """
        for output in outputs:
            arr = np.asarray(output)
            if arr.ndim == 2 and arr.shape[0] == n_samples:
                return arr.astype(np.float64, copy=False)
        # Fallback: list-of-dict (ZipMap output).
        for output in outputs:
            if isinstance(output, list) and output and isinstance(output[0], dict):
                keys = sorted(output[0].keys(), key=lambda k: int(float(k)))
                matrix = np.zeros((n_samples, len(keys)), dtype=np.float64)
                for i, row in enumerate(output):
                    for j, k in enumerate(keys):
                        matrix[i, j] = float(row[k])
                return matrix
        raise RuntimeError(
            "Could not interpret ONNX outputs as a probability matrix; "
            f"got {[type(o).__name__ for o in outputs]}"
        )

    def _align_columns(self, proba: np.ndarray) -> np.ndarray:
        """If the ONNX model's column ordering differs from self.classes_, permute."""
        n_classes = len(self.classes_)
        if proba.shape[1] == n_classes:
            return proba
        # Cannot safely permute without class id mapping; assume identity for v1.
        if proba.shape[1] < n_classes:
            padded = np.zeros((proba.shape[0], n_classes), dtype=proba.dtype)
            padded[:, : proba.shape[1]] = proba
            return padded
        return proba[:, :n_classes]