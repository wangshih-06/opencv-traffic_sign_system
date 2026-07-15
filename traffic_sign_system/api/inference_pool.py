"""ProcessPoolExecutor wrapper for CPU-bound ML inference.

Why a separate process pool?
----------------------------
- sklearn's SVC.proba / KNN / RF hold the GIL during the hot path.
- starlette's default thread pool (40 threads) is shared between HTTP and
  WebSocket. A burst of WebSocket frames can starve HTTP request handlers.
- A process pool gives each worker its own Python interpreter → no GIL
  contention, model loaded once and cached in process memory.

Worker side
-----------
The module-level ``_WORKER_PREDICTORS`` dict caches the loaded predictor
per ``bundle_path`` along with the file's mtime. If the bundle is replaced
on disk (e.g. after ``scripts/train.py``), the next call detects the mtime
change and rebuilds. The decision between ``Predictor`` (joblib) and
``OnnxPredictor`` is made by reading the artifact directory: if a sibling
``.onnx`` file exists for the bundle name, ONNX is preferred; otherwise
joblib is used. This auto-fallback lets you incrementally enable ONNX
per-bundle without code changes.

Main side
---------
``InferencePool.predict`` / ``predict_batch`` / ``detect`` / ``clear_cache``
each submit a worker-side remote function via ``loop.run_in_executor`` and
return the result to the FastAPI handler. Image bytes are passed raw (no
base64) for the lowest IPC overhead.

Public lifecycle
----------------
- ``InferencePool(max_workers=None, default_bundle=None)``
- ``await pool.predict(bundle_path, image, top_k=5) -> dict``
- ``await pool.predict_batch(bundle_path, images) -> list[dict]``
- ``await pool.detect(bundle_path, image) -> dict``
- ``await pool.clear_cache(bundle_path) -> dict``
- ``await pool.warm() -> None``  — load default_bundle in every worker
- ``await pool.shutdown() -> None``  — best-effort cancel and exit
- ``pool.cache_stats_for(bundle_path) -> dict``  — local cache aggregator

Cache stats
-----------
Predictor cache lives in worker processes; we expose ``cache_stats_for``
which submits a probe (``_stats_remote``) and caches the result locally
for ``cache_stats_ttl_seconds`` to avoid round-trips on every API call.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker side: cached predictors and remote callables
# ---------------------------------------------------------------------------
#
# These run in *worker processes*. Module-level state is per-worker because
# multiprocessing.spawn re-imports the module in each child.

_WORKER_PREDICTORS: dict[str, tuple[Any, float]] = {}  # path -> (predictor, mtime)
_WORKER_LOCK = threading.Lock()


def _bundle_paths_for(bundle_path: str) -> tuple[Path, Path | None]:
    """Return (joblib_path, onnx_path_or_None) for a bundle."""
    p = Path(bundle_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Bundle does not exist: {bundle_path}")
    onnx_path = p.with_suffix(".onnx")
    return p, (onnx_path if onnx_path.is_file() else None)


def _get_worker_predictor(bundle_path: str) -> Any:
    """Return the worker-local predictor for *bundle_path*, loading on first call.

    Honors bundle file mtime: replacing the .joblib / .onnx on disk forces a
    fresh load on the next request.
    """
    from traffic_sign_system.recognition.predictor import Predictor
    from traffic_sign_system.recognition.onnx_predictor import OnnxPredictor

    joblib_path, onnx_path = _bundle_paths_for(bundle_path)
    cache_key = str(onnx_path if onnx_path is not None else joblib_path)
    current_mtime = (
        onnx_path.stat().st_mtime if onnx_path is not None else joblib_path.stat().st_mtime
    )

    with _WORKER_LOCK:
        cached = _WORKER_PREDICTORS.get(cache_key)
        if cached is not None and cached[1] == current_mtime:
            return cached[0]

    # Build outside the lock — heavy work that we don't want blocking other workers.
    if onnx_path is not None:
        logger.info("[worker] loading ONNX predictor: %s", onnx_path)
        predictor = OnnxPredictor(onnx_path)
    else:
        logger.info("[worker] loading joblib predictor: %s", joblib_path)
        predictor = Predictor(joblib_path)

    with _WORKER_LOCK:
        _WORKER_PREDICTORS[cache_key] = (predictor, current_mtime)
    return predictor


# ── Remote functions (must be top-level for pickle) ────────────────────


def _decode_image(image_bytes: bytes, h: int, w: int) -> np.ndarray:
    """Reconstruct a uint8 BGR ndarray from raw bytes + shape."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    expected = h * w * 3
    if arr.size != expected:
        raise ValueError(
            f"Image byte size mismatch: got {arr.size}, expected {expected} "
            f"(h={h}, w={w}, channels=3)"
        )
    return arr.reshape(h, w, 3)


def _predict_remote(
    bundle_path: str,
    image_bytes: bytes,
    h: int,
    w: int,
    top_k: int,
) -> dict[str, Any]:
    predictor = _get_worker_predictor(bundle_path)
    image = _decode_image(image_bytes, h, w)
    result = predictor.predict(image)
    if top_k > 1 and hasattr(predictor.classifier, "predict_proba"):
        try:
            top_k_list = _top_k_from_predictor(predictor, image, top_k)
        except Exception:  # noqa: BLE001 — top_k is best-effort
            top_k_list = []
    else:
        top_k_list = []
    return {
        "result": result,
        "top_k": top_k_list,
        "cache": predictor.cache_stats(),
    }


def _predict_batch_remote(
    bundle_path: str,
    payloads: list[tuple[bytes, int, int]],
) -> dict[str, Any]:
    predictor = _get_worker_predictor(bundle_path)
    images = [_decode_image(b, h, w) for b, h, w in payloads]
    return {
        "results": predictor.predict_batch(images),
        "cache": predictor.cache_stats(),
    }


def _detect_remote(
    bundle_path: str,
    image_bytes: bytes,
    h: int,
    w: int,
) -> dict[str, Any]:
    """Detect + classify traffic signs on a single image."""
    from traffic_sign_system.recognition.scene_aware import SceneAnalyzer
    from traffic_sign_system.recognition.sign_detector import SignDetector

    predictor = _get_worker_predictor(bundle_path)
    image = _decode_image(image_bytes, h, w)
    scene_analyzer = SceneAnalyzer()
    scene = scene_analyzer.analyze(image)
    scene["recommendations"] = scene_analyzer.recommend_params(scene)
    detector = SignDetector(predictor)
    detections = detector.detect(image)
    return {
        "detections": detections,
        "count": len(detections),
        "cache": predictor.cache_stats(),
        "scene": scene,
    }


def _clear_cache_remote(bundle_path: str) -> dict[str, int | float]:
    predictor = _get_worker_predictor(bundle_path)
    predictor.clear_cache()
    return predictor.cache_stats()


def _stats_remote(bundle_path: str) -> dict[str, int | float]:
    predictor = _get_worker_predictor(bundle_path)
    return predictor.cache_stats()


def _top_k_from_predictor(
    predictor: Any, image: np.ndarray, limit: int
) -> list[dict[str, Any]]:
    """Reproduce the API's _top_k helper inside the worker."""
    classifier = predictor.classifier
    if not hasattr(classifier, "predict_proba"):
        return []
    features = predictor.builder.extract_one(image).astype(np.float32, copy=False)
    X_scaled = predictor.scaler.transform(features[None, :])
    probabilities = np.asarray(classifier.predict_proba(X_scaled))[0]
    classes = np.asarray(classifier.classes_)
    order = np.argsort(probabilities)[::-1][:limit]
    return [
        {
            "class_id": int(classes[i]),
            "class_name": predictor.label_map.get(int(classes[i]), str(int(classes[i]))),
            "confidence": float(probabilities[i]),
        }
        for i in order
    ]


# ---------------------------------------------------------------------------
# Main side: InferencePool
# ---------------------------------------------------------------------------


class InferencePool:
    """Async-friendly wrapper around a ProcessPoolExecutor for ML inference."""

    DEFAULT_MAX_WORKERS = 4  # cap so RF bundle × 4 workers ≈ 1.8GB
    CACHE_STATS_TTL_SECONDS = 2.0

    def __init__(
        self,
        max_workers: int | None = None,
        default_bundle: Path | str | None = None,
    ):
        if max_workers is None:
            env = os.environ.get("INFERENCE_POOL_WORKERS")
            if env:
                try:
                    max_workers = int(env)
                except ValueError:
                    max_workers = self.DEFAULT_MAX_WORKERS
            else:
                cpu = os.cpu_count() or 2
                max_workers = min(self.DEFAULT_MAX_WORKERS, cpu)
        if max_workers < 1:
            max_workers = 1
        self.max_workers = max_workers
        self.default_bundle = Path(default_bundle).resolve() if default_bundle else None
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        # Local cache stats aggregator — populated by predict/detect/clear.
        self._cache_stats: dict[str, tuple[float, dict[str, int | float]]] = {}
        self._shutdown = False
        logger.info("InferencePool initialized with %d workers", max_workers)

    # ── Public async API ──────────────────────────────────────────────

    async def predict(
        self,
        bundle_path: Path | str,
        image: np.ndarray,
        top_k: int = 5,
        *,
        timeout: float | None = 30.0,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        bundle_path, image_bytes, h, w = self._serialize_single(bundle_path, image)
        result = await self._submit(
            _predict_remote,
            (bundle_path, image_bytes, h, w, int(top_k)),
            timeout=timeout,
        )
        # Update local cache stats aggregator with the worker's report.
        if "cache" in result:
            self._record_cache_stats(bundle_path, result["cache"])
        return result

    async def predict_batch(
        self,
        bundle_path: Path | str,
        images: list[np.ndarray],
        *,
        timeout: float | None = 60.0,
    ) -> list[dict[str, Any]]:
        loop = asyncio.get_running_loop()
        bundle_path, payloads = self._serialize_batch(bundle_path, images)
        payload = await self._submit(
            _predict_batch_remote,
            (bundle_path, payloads),
            timeout=timeout,
        )
        if not isinstance(payload, dict) or "results" not in payload:
            raise ValueError("batch worker returned an invalid response")
        cache = payload.get("cache")
        if isinstance(cache, dict):
            self._record_cache_stats(bundle_path, cache)
        results = payload["results"]
        if not isinstance(results, list):
            raise ValueError("batch worker returned invalid results")
        return results

    async def detect(
        self,
        bundle_path: Path | str,
        image: np.ndarray,
        *,
        timeout: float | None = 30.0,
    ) -> dict[str, Any]:
        bundle_path, image_bytes, h, w = self._serialize_single(bundle_path, image)
        result = await self._submit(
            _detect_remote,
            (bundle_path, image_bytes, h, w),
            timeout=timeout,
        )
        if "cache" in result:
            self._record_cache_stats(bundle_path, result["cache"])
        return result

    async def clear_cache(
        self,
        bundle_path: Path | str,
        *,
        timeout: float | None = 10.0,
    ) -> dict[str, int | float]:
        bundle_path = str(Path(bundle_path).resolve())
        stats = await self._submit(_clear_cache_remote, (bundle_path,), timeout=timeout)
        self._record_cache_stats(bundle_path, stats)
        return stats

    async def warm(
        self, default_bundle: Path | str | None = None
    ) -> None:
        """Pre-load *default_bundle* in every worker process.

        Skipped silently if no default bundle is configured. Each worker
        receives one dummy predict job; the predictor stays resident afterwards.
        """
        bundle = default_bundle or self.default_bundle
        if bundle is None:
            return
        bundle_path = str(Path(bundle).resolve())
        try:
            h, w = 64, 64  # FeatureBuilder default; doesn't matter — workers
            # cache the predictor regardless of input.
            image_bytes = (np.zeros((h, w, 3), dtype=np.uint8)).tobytes()
        except Exception as exc:  # noqa: BLE001
            logger.warning("warm() skipped: failed to build dummy image: %s", exc)
            return
        try:
            await asyncio.gather(
                *(
                    self._submit(
                        _predict_remote,
                        (bundle_path, image_bytes, h, w, 1),
                        timeout=120.0,
                    )
                    for _ in range(self.max_workers)
                )
            )
            logger.info(
                "InferencePool warmed with %d workers using %s",
                self.max_workers,
                Path(bundle_path).name,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("warm() failed: %s", exc)

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        # cancel_futures rejects queued jobs; in-flight ones still run to
        # completion. wait=False returns immediately.
        self._executor.shutdown(wait=False, cancel_futures=True)
        logger.info("InferencePool shut down")

    def cache_stats_for(self, bundle_path: Path | str) -> dict[str, int | float] | None:
        """Return the most recent cache stats for *bundle_path* (TTL-bounded)."""
        key = str(Path(bundle_path).resolve())
        entry = self._cache_stats.get(key)
        if entry is None:
            return None
        ts, stats = entry
        if time.monotonic() - ts > self.CACHE_STATS_TTL_SECONDS:
            return None
        return stats

    # ── Internals ─────────────────────────────────────────────────────

    def _serialize_single(
        self, bundle_path: Path | str, image: np.ndarray
    ) -> tuple[str, bytes, int, int]:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"image must have shape (H, W, 3); got {image.shape}")
        h, w = int(image.shape[0]), int(image.shape[1])
        # Pass the bytes directly; numpy `tobytes()` shares underlying memory.
        return str(Path(bundle_path).resolve()), image.tobytes(), h, w

    def _serialize_batch(
        self, bundle_path: Path | str, images: list[np.ndarray]
    ) -> tuple[str, list[tuple[bytes, int, int]]]:
        if not isinstance(images, (list, tuple)):
            raise TypeError(f"images must be a list; got {type(images).__name__}")
        if not images:
            raise ValueError("images must be a non-empty list")
        payloads: list[tuple[bytes, int, int]] = []
        for img in images:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(
                    f"each image must have shape (H, W, 3); got {img.shape}"
                )
            payloads.append((img.tobytes(), int(img.shape[0]), int(img.shape[1])))
        return str(Path(bundle_path).resolve()), payloads

    def _record_cache_stats(self, bundle_path: str, stats: dict[str, int | float]) -> None:
        # Pick the worker with the highest hit count to capture realistic stats
        # in the presence of multiple workers. For LRU-cache purposes, the
        # total count is what matters; we report the most-recent worker.
        self._cache_stats[bundle_path] = (time.monotonic(), dict(stats))

    async def _submit(
        self,
        fn: Any,
        args: tuple[Any, ...],
        *,
        timeout: float | None,
    ) -> Any:
        loop = asyncio.get_running_loop()
        coro = loop.run_in_executor(self._executor, fn, *args)
        if timeout is not None:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro