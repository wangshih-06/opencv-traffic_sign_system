"""Web API used by the React frontend.

Transport-only layer: uploads, JSON serialization, model lifecycle and
WebSocket frames. All CPU-bound recognition runs in a
:class:`~traffic_sign_system.api.inference_pool.InferencePool` so the event
loop is never blocked by sklearn or ONNX inference.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from traffic_sign_system.api.feedback_store import FeedbackStore
from traffic_sign_system.api.inference_pool import InferencePool
from traffic_sign_system.config.labels import get_all_labels
from traffic_sign_system.models.model_manager import load_bundle
from traffic_sign_system.recognition.tracker import SimpleTracker

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_DIR = PACKAGE_ROOT / "models" / "artifacts"
DEFAULT_BUNDLE_NAME = "svm_hog+hsv.joblib"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_BATCH_FILES = 50
FEEDBACK_DB_PATH = PROJECT_ROOT / "runtime" / "feedback.db"
FEEDBACK_STATUSES = {"new", "reviewed", "exported"}


# ---------------------------------------------------------------------------
# Lifespan: build the inference pool at startup, shut it down at exit.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    default_bundle = ARTIFACTS_DIR / DEFAULT_BUNDLE_NAME
    pool = InferencePool(default_bundle=default_bundle)
    app.state.inference_pool = pool
    app.state.active_bundle: str | None = DEFAULT_BUNDLE_NAME
    app.state.bundle_meta_cache: dict[str, dict[str, Any]] = {}
    app.state.bundle_meta_lock = RLock()
    app.state.feedback_store = FeedbackStore(FEEDBACK_DB_PATH)
    # Best-effort warm; never block startup on it.
    try:
        await asyncio.wait_for(pool.warm(), timeout=120.0)
    except (asyncio.TimeoutError, Exception) as exc:  # noqa: BLE001
        logger.warning("inference pool warm-up skipped: %s", exc)
    try:
        yield
    finally:
        await pool.shutdown()


app = FastAPI(
    title="交通标志分类识别 API",
    description="为 React Web 前端复用现有 Predictor、SignDetector 的轻量服务层。",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ModelLoadRequest(BaseModel):
    """Select and warm a model bundle."""

    name: str


class FeedbackCreateRequest(BaseModel):
    """Human verification or correction for one recognition result."""

    history_id: str | None = None
    source: Literal["image", "camera", "video", "batch"]
    filename: str = Field(min_length=1, max_length=255)
    model: str | None = Field(default=None, max_length=255)
    predicted_class_id: int = Field(ge=0)
    predicted_class_name: str = Field(min_length=1, max_length=255)
    predicted_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    corrected_class_id: int | None = Field(default=None, ge=0)
    corrected_class_name: str | None = Field(default=None, max_length=255)
    verdict: Literal["correct", "incorrect"]
    note: str = Field(default="", max_length=1000)
    bbox: list[float] | None = None


class FeedbackStatusRequest(BaseModel):
    status: Literal["new", "reviewed", "exported"]



def _feedback_store() -> FeedbackStore:
    store: FeedbackStore | None = getattr(app.state, "feedback_store", None)
    if store is None:
        store = FeedbackStore(FEEDBACK_DB_PATH)
        app.state.feedback_store = store
    return store


def _normalise_feedback(request: FeedbackCreateRequest) -> dict[str, Any]:
    labels = get_all_labels()
    if request.predicted_class_id not in labels:
        raise HTTPException(status_code=400, detail="预测类别编号不在当前标签集中")

    if request.verdict == "correct":
        corrected_id = request.predicted_class_id
    elif request.corrected_class_id is None:
        raise HTTPException(status_code=400, detail="错例必须选择纠正后的真实类别")
    else:
        corrected_id = request.corrected_class_id

    if corrected_id not in labels:
        raise HTTPException(status_code=400, detail="纠正类别编号不在当前标签集中")
    if request.bbox is not None and len(request.bbox) != 4:
        raise HTTPException(status_code=400, detail="bbox 必须包含 x、y、width、height 四个值")

    payload = request.model_dump()
    payload["corrected_class_id"] = corrected_id
    payload["corrected_class_name"] = labels[corrected_id]
    payload["predicted_class_name"] = request.predicted_class_name.strip()
    payload["filename"] = request.filename.strip()
    payload["note"] = request.note.strip()
    return payload


def _bundle_files() -> list[Path]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(ARTIFACTS_DIR.glob("*.joblib"), key=lambda path: path.name.lower())


def _resolve_bundle(bundle: str | None = None, *, active: str | None = None) -> Path:
    """Resolve a model name/path while preventing traversal outside artifacts."""
    requested = bundle or active or DEFAULT_BUNDLE_NAME
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = ARTIFACTS_DIR / candidate.name
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"模型文件不存在：{candidate.name}") from exc

    artifacts_resolved = ARTIFACTS_DIR.resolve()
    if artifacts_resolved not in resolved.parents or resolved.suffix.lower() != ".joblib":
        raise HTTPException(status_code=400, detail="仅允许加载模型目录中的 .joblib 文件")
    return resolved


def _peek_bundle_meta(path: Path) -> dict[str, Any]:
    """Read feature_mode / feature_dim / classifier_type from a bundle on disk.

    Uses ``load_bundle`` which deserializes the full classifier — heavy for RF
    (~5 s), but acceptable for the metadata endpoint. Results are cached on
    ``app.state.bundle_meta_cache`` keyed by mtime so repeated calls are cheap.
    """
    cache: dict[str, dict[str, Any]] = app.state.bundle_meta_cache
    lock: RLock = app.state.bundle_meta_lock
    key = str(path.resolve())
    mtime = path.stat().st_mtime
    with lock:
        cached = cache.get(key)
        if cached and cached.get("_mtime") == mtime:
            return cached

    bundle = load_bundle(path)
    classifier = bundle["classifier"]
    classifier_name = type(classifier).__name__
    is_ensemble = classifier_name == "EnsembleClassifier"
    feature_config = dict(bundle["feature_config"])
    summary = dict(bundle["summary"])
    onnx_sibling = path.with_suffix(".onnx")
    is_onnx = onnx_sibling.is_file()
    backend = "onnx" if is_onnx else "joblib"
    meta = {
        "_mtime": mtime,
        "classifier": classifier_name,
        "feature_mode": feature_config.get("mode"),
        "feature_dim": summary.get("feature_dim"),
        "classes": len(bundle["label_map"]),
        "is_ensemble": is_ensemble,
        "is_onnx": is_onnx,
        "backend": backend,
    }
    with lock:
        cache[key] = meta
    return meta


def _decode_image(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="图片内容为空")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="图片不能超过 20 MB")
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="无法解析图片，请上传 JPG、PNG、WebP 或 BMP")
    return image


async def _read_upload(image: UploadFile) -> tuple[bytes, np.ndarray]:
    if image.content_type and image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail=f"不支持的图片类型：{image.content_type}")
    raw = await image.read(MAX_IMAGE_BYTES + 1)
    return raw, _decode_image(raw)


def _model_metadata(path: Path) -> dict[str, Any]:
    """Build the per-bundle metadata dict returned by ``/api/models``."""
    key = str(path.resolve())
    cached_meta = _peek_bundle_meta(path)
    pool: InferencePool = app.state.inference_pool
    cache_stats = pool.cache_stats_for(key)
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "modified_at": path.stat().st_mtime,
        "loaded": cache_stats is not None,
        "active": path.name == app.state.active_bundle,
        "classifier": cached_meta["classifier"],
        "feature_mode": cached_meta["feature_mode"],
        "feature_dim": cached_meta["feature_dim"],
        "backend": cached_meta["backend"],
        "cache": cache_stats,
    }


def _build_predict_response(
    *,
    name: str,
    filename: str | None,
    image: np.ndarray,
    elapsed: float,
    worker_result: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the legacy ``/api/predict`` response shape from a worker result."""
    result = worker_result.get("result") or {}
    return {
        "model": name,
        "filename": filename,
        **result,
        "predict_seconds": float(elapsed),
        "top_k": worker_result.get("top_k", []),
        "cache": worker_result.get("cache") or {},
        "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
    }


def _select_primary_track(tracks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the most useful tracked object for the compact current-result card."""
    if not tracks:
        return None
    return min(
        tracks,
        key=lambda item: (
            int(item.get("lost_count", 0)) > 0,
            -(
                float(item["confidence"])
                if item.get("confidence") is not None
                else -1.0
            ),
            int(item.get("track_id", 0)),
        ),
    )


def _batch_error_payload(exc: Exception) -> dict[str, Any]:
    status_code = int(getattr(exc, "status_code", 500))
    detail = getattr(exc, "detail", str(exc))
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False)
    code_by_status = {
        400: "invalid_image",
        413: "file_too_large",
        415: "unsupported_media_type",
    }
    return {
        "code": code_by_status.get(
            status_code, "inference_error" if status_code >= 500 else "file_error"
        ),
        "message": detail or "\u6587\u4ef6\u5904\u7406\u5931\u8d25",
        "status_code": status_code,
    }


def _batch_failure_item(filename: str, exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "model": None,
        "filename": filename,
        "class_id": None,
        "class_name": None,
        "confidence": None,
        "predict_seconds": 0.0,
        "top_k": [],
        "cache": {},
        "image": None,
        "error": _batch_error_payload(exc),
    }


def _batch_success_item(
    *,
    model: str,
    filename: str,
    image: np.ndarray,
    result: dict[str, Any],
    elapsed: float,
    cache: dict[str, int | float],
) -> dict[str, Any]:
    class_id = result.get("class_id")
    class_name = result.get("class_name")
    if class_id is None or class_name is None:
        raise ValueError("\u6a21\u578b\u8fd4\u56de\u4e86\u4e0d\u5b8c\u6574\u7684\u5206\u7c7b\u7ed3\u679c")
    top_k = result.get("top_k") or [
        {
            "class_id": int(class_id),
            "class_name": str(class_name),
            "confidence": result.get("confidence"),
        }
    ]
    return {
        "ok": True,
        "model": model,
        "filename": filename,
        "class_id": int(class_id),
        "class_name": str(class_name),
        "confidence": result.get("confidence"),
        "predict_seconds": float(elapsed),
        "top_k": top_k,
        "cache": result.get("cache") or cache,
        "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "error": None,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
def health() -> dict[str, Any]:
    bundles = _bundle_files()
    pool: InferencePool | None = getattr(app.state, "inference_pool", None)
    return {
        "status": "ok",
        "service": "traffic-sign-recognition",
        "model_available": bool(bundles),
        "active_model": getattr(app.state, "active_bundle", None),
        "loaded_models": len(getattr(app.state, "bundle_meta_cache", {})),
        "pool_workers": pool.max_workers if pool else 0,
        "labels": len(get_all_labels()),
    }



@app.get("/api/feedback")
def list_feedback(
    status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    if status is not None and status not in FEEDBACK_STATUSES:
        raise HTTPException(status_code=400, detail="不支持的反馈状态")
    store = _feedback_store()
    items, count = store.list(status=status, limit=limit, offset=offset)
    return {"items": items, "count": count, "stats": store.stats()}


@app.post("/api/feedback", status_code=201)
def create_feedback(request: FeedbackCreateRequest) -> dict[str, Any]:
    record = _feedback_store().create(_normalise_feedback(request))
    return {"ok": True, "item": record, "stats": _feedback_store().stats()}


@app.patch("/api/feedback/{feedback_id}")
def update_feedback(feedback_id: str, request: FeedbackStatusRequest) -> dict[str, Any]:
    record = _feedback_store().update_status(feedback_id, request.status)
    if record is None:
        raise HTTPException(status_code=404, detail="反馈记录不存在")
    return {"ok": True, "item": record, "stats": _feedback_store().stats()}


@app.delete("/api/feedback/{feedback_id}")
def delete_feedback(feedback_id: str) -> dict[str, Any]:
    if not _feedback_store().delete(feedback_id):
        raise HTTPException(status_code=404, detail="反馈记录不存在")
    return {"ok": True, "stats": _feedback_store().stats()}


@app.get("/api/feedback/export")
def export_feedback(status: str | None = Query(default=None)) -> StreamingResponse:
    if status is not None and status not in FEEDBACK_STATUSES:
        raise HTTPException(status_code=400, detail="不支持的反馈状态")
    csv_text = _feedback_store().export_csv(status=status)
    headers = {"Content-Disposition": 'attachment; filename="traffic-sign-feedback.csv"'}
    return StreamingResponse(
        iter(["\ufeff" + csv_text]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@app.get("/api/labels")
def labels() -> dict[str, Any]:
    return {
        "count": len(get_all_labels()),
        "items": [
            {"class_id": class_id, "class_name": class_name}
            for class_id, class_name in sorted(get_all_labels().items())
        ],
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {
        "active_model": app.state.active_bundle,
        "default_model": DEFAULT_BUNDLE_NAME,
        "bundles": [_model_metadata(path) for path in _bundle_files()],
    }


@app.post("/api/models/load")
async def load_model(request: ModelLoadRequest) -> dict[str, Any]:
    path = _resolve_bundle(request.name)
    meta = _peek_bundle_meta(path)
    app.state.active_bundle = path.name
    # Best-effort warm in background; do not block the response.
    pool: InferencePool = app.state.inference_pool
    if pool.default_bundle != path:
        # Only the configured default bundle is auto-warmed; explicit loads
        # are warm-on-first-use.
        logger.info("active bundle switched to %s (lazy warm)", path.name)
    return {
        "ok": True,
        "model": _model_metadata(path),
        "summary": {
            "classifier": meta["classifier"],
            "feature_mode": meta["feature_mode"],
            "feature_dim": meta["feature_dim"],
            "classes": meta["classes"],
            "backend": meta["backend"],
        },
    }


@app.delete("/api/models/cache")
async def clear_model_cache(bundle: str | None = Query(default=None)) -> dict[str, Any]:
    path = _resolve_bundle(bundle, active=app.state.active_bundle)
    pool: InferencePool = app.state.inference_pool
    stats = await pool.clear_cache(path)
    return {"ok": True, "model": path.name, "cache": stats}


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    bundle: str | None = Query(default=None),
    top_k: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    _raw, decoded = await _read_upload(image)
    path = _resolve_bundle(bundle, active=app.state.active_bundle)
    pool: InferencePool = app.state.inference_pool
    started = time.perf_counter()
    worker_result = await pool.predict(path, decoded, top_k=top_k)
    elapsed = time.perf_counter() - started
    return _build_predict_response(
        name=path.name,
        filename=image.filename,
        image=decoded,
        elapsed=elapsed,
        worker_result=worker_result,
    )


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    bundle: str | None = Query(default=None),
) -> dict[str, Any]:
    _raw, decoded = await _read_upload(image)
    path = _resolve_bundle(bundle, active=app.state.active_bundle)
    pool: InferencePool = app.state.inference_pool
    started = time.perf_counter()
    worker_result = await pool.detect(path, decoded)
    elapsed = time.perf_counter() - started
    return {
        "model": path.name,
        "filename": image.filename,
        "detections": worker_result.get("detections", []),
        "count": worker_result.get("count", 0),
        "detect_seconds": elapsed,
        "cache": worker_result.get("cache") or {},
        "scene": worker_result.get("scene") or {},
        "image": {"width": int(decoded.shape[1]), "height": int(decoded.shape[0])},
    }


@app.post("/api/batch")
async def batch_predict(
    images: list[UploadFile] = File(...),
    bundle: str | None = Query(default=None),
) -> dict[str, Any]:
    """Batch classify files while isolating per-file upload/inference failures.

    The fast vectorised path is used whenever possible. If one valid image
    makes the vectorised predictor fail, the request is retried one file at a
    time so healthy files still produce results and the failed filename gets a
    structured error item.
    """
    if not images:
        raise HTTPException(status_code=400, detail="\u8bf7\u81f3\u5c11\u9009\u62e9\u4e00\u5f20\u56fe\u7247")
    if len(images) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"\u5355\u6b21\u6700\u591a\u5904\u7406 {MAX_BATCH_FILES} \u5f20\u56fe\u7247")

    path = _resolve_bundle(bundle, active=app.state.active_bundle)
    pool: InferencePool = app.state.inference_pool
    items: list[dict[str, Any] | None] = [None] * len(images)
    valid_entries: list[tuple[int, str, np.ndarray]] = []

    for index, upload in enumerate(images):
        filename = upload.filename or f"image-{index + 1}"
        try:
            _raw, decoded = await _read_upload(upload)
        except Exception as exc:  # noqa: BLE001 - preserve one-file failure
            items[index] = _batch_failure_item(filename, exc)
        else:
            valid_entries.append((index, filename, decoded))

    inference_started = time.perf_counter()
    if valid_entries:
        valid_images = [entry[2] for entry in valid_entries]
        try:
            batch_results = await pool.predict_batch(path, valid_images)
            if len(batch_results) != len(valid_entries):
                raise ValueError(
                    "\u6279\u91cf\u6a21\u578b\u8fd4\u56de\u6570\u91cf\u4e0e\u8f93\u5165\u6587\u4ef6\u6570\u91cf\u4e0d\u4e00\u81f4"
                )
            elapsed = time.perf_counter() - inference_started
            per_file_elapsed = elapsed / len(valid_entries)
            cache = pool.cache_stats_for(path) or {}
            for (index, filename, image), result in zip(
                valid_entries, batch_results, strict=True
            ):
                items[index] = _batch_success_item(
                    model=path.name,
                    filename=filename,
                    image=image,
                    result=result,
                    elapsed=per_file_elapsed,
                    cache=cache,
                )
        except Exception as batch_exc:  # noqa: BLE001 - retry per file
            logger.warning(
                "batch inference failed; retrying files individually: %s",
                batch_exc,
            )

            async def predict_one(
                entry: tuple[int, str, np.ndarray],
            ) -> tuple[int, dict[str, Any]]:
                index, filename, image = entry
                started = time.perf_counter()
                try:
                    worker_result = await pool.predict(path, image, top_k=1)
                    elapsed = time.perf_counter() - started
                    cache = pool.cache_stats_for(path) or {}
                    item = _batch_success_item(
                        model=path.name,
                        filename=filename,
                        image=image,
                        result=worker_result.get("result") or {},
                        elapsed=elapsed,
                        cache=worker_result.get("cache") or cache,
                    )
                except Exception as exc:  # noqa: BLE001 - isolate one file
                    item = _batch_failure_item(filename, exc)
                return index, item

            individual_results = await asyncio.gather(
                *(predict_one(entry) for entry in valid_entries)
            )
            for index, item in individual_results:
                items[index] = item

    final_items = [item for item in items if item is not None]
    success_count = sum(bool(item["ok"]) for item in final_items)
    failed_count = len(final_items) - success_count
    elapsed = time.perf_counter() - inference_started if valid_entries else 0.0
    return {
        "model": path.name,
        "count": len(final_items),
        "success_count": success_count,
        "failed_count": failed_count,
        "predict_seconds": elapsed,
        "cache": pool.cache_stats_for(path) or {},
        "items": final_items,
    }


@app.websocket("/ws/stream")
async def stream_frames(
    websocket: WebSocket,
    bundle: str | None = Query(default=None),
    skip_frames: int = Query(default=1, ge=0, le=10),
) -> None:
    """Detect and track multiple traffic signs in browser-sent JPEG frames.

    Detection/classification is CPU-heavy and therefore runs in the process
    pool. The lightweight ``SimpleTracker`` stays local to this WebSocket
    connection, which preserves track IDs without introducing cross-process
    session affinity. Skipped frames reuse the latest tracked boxes.
    """
    await websocket.accept()
    pool: InferencePool = app.state.inference_pool
    try:
        path = _resolve_bundle(bundle, active=app.state.active_bundle)
        await websocket.send_json(
            {"type": "ready", "model": path.name, "mode": "detect-track"}
        )
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "message": str(exc.detail)})
        await websocket.close(code=1008)
        return

    tracker = SimpleTracker(
        iou_threshold=0.3,
        max_lost=5,
        history_size=7,
        bbox_smoothing=0.65,
        confidence_smoothing=0.65,
    )
    frame_index = 0
    processed_in_window = 0
    processed_total = 0
    window_start = time.perf_counter()
    last_tracks: list[dict[str, Any]] | None = None
    last_cache: dict[str, int | float] = {}
    last_scene: dict[str, Any] = {}

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            raw = message.get("bytes")
            if raw is None:
                text = message.get("text")
                if text == "ping":
                    await websocket.send_json({"type": "pong"})
                continue

            frame_index += 1
            image = _decode_image(raw)
            should_detect = (
                last_tracks is None
                or (frame_index - 1) % (skip_frames + 1) == 0
            )
            detection_ms = 0.0
            tracker_ms = 0.0

            if should_detect:
                started = time.perf_counter()
                worker_result = await pool.detect(path, image)
                detection_ms = (time.perf_counter() - started) * 1000.0

                tracker_started = time.perf_counter()
                last_tracks = tracker.update(
                    list(worker_result.get("detections") or [])
                )
                tracker_ms = (time.perf_counter() - tracker_started) * 1000.0
                last_cache = worker_result.get("cache") or {}
                last_scene = worker_result.get("scene") or {}
                processed_in_window += 1
                processed_total += 1

            tracks = last_tracks or []
            primary = _select_primary_track(tracks)
            now = time.perf_counter()
            elapsed = max(now - window_start, 1e-6)
            detection_fps = processed_in_window / elapsed
            if elapsed >= 2.0:
                processed_in_window = 0
                window_start = now

            active_count = sum(
                int(item.get("lost_count", 0)) == 0 for item in tracks
            )
            await websocket.send_json(
                {
                    "type": "prediction",
                    "mode": "detect-track",
                    "frame_index": frame_index,
                    "processed_frames": processed_total,
                    "result": (
                        {**primary, "reused": not should_detect}
                        if primary is not None
                        else None
                    ),
                    "detections": tracks,
                    "detection_count": active_count,
                    "tracked_count": len(tracks),
                    "reused": not should_detect,
                    "predict_ms": detection_ms,
                    "tracker_ms": tracker_ms,
                    "fps": detection_fps,
                    "cache": last_cache,
                    "scene": last_scene,
                    "scene_reused": not should_detect,
                    "image": {
                        "width": int(image.shape[1]),
                        "height": int(image.shape[0]),
                    },
                }
            )
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("WebSocket detection stream failed")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1011)
        except RuntimeError:
            pass
    finally:
        tracker.reset()

def main() -> None:
    """Development server entry point."""
    import uvicorn

    uvicorn.run(
        "traffic_sign_system.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
