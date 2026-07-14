"""Web API used by the React frontend.

The module intentionally keeps all recognition logic in the existing Python
classes.  It only handles transport concerns: uploads, JSON serialization,
model lifecycle and browser WebSocket frames.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from threading import RLock
from typing import Any

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
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from traffic_sign_system.config.labels import get_all_labels
from traffic_sign_system.recognition.predictor import Predictor
from traffic_sign_system.recognition.sign_detector import SignDetector

logger = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
ARTIFACTS_DIR = PACKAGE_ROOT / "models" / "artifacts"
DEFAULT_BUNDLE_NAME = "svm_hog+hsv.joblib"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_BATCH_FILES = 50

app = FastAPI(
    title="交通标志分类识别 API",
    description="为 React Web 前端复用现有 Predictor、SignDetector 的轻量服务层。",
    version="1.0.0",
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

_PREDICTORS: dict[str, Predictor] = {}
_DETECTORS: dict[str, SignDetector] = {}
_REGISTRY_LOCK = RLock()
_ACTIVE_BUNDLE: str | None = None


class ModelLoadRequest(BaseModel):
    """Select and warm a model bundle."""

    name: str


def _bundle_files() -> list[Path]:
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(ARTIFACTS_DIR.glob("*.joblib"), key=lambda path: path.name.lower())


def _resolve_bundle(bundle: str | None = None) -> Path:
    """Resolve a model name/path while preventing traversal outside artifacts."""
    global _ACTIVE_BUNDLE

    requested = bundle or _ACTIVE_BUNDLE or DEFAULT_BUNDLE_NAME
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


def _get_predictor(bundle: str | None = None) -> tuple[str, Predictor]:
    global _ACTIVE_BUNDLE

    path = _resolve_bundle(bundle)
    key = str(path)
    with _REGISTRY_LOCK:
        predictor = _PREDICTORS.get(key)
        if predictor is None:
            try:
                predictor = Predictor(path)
            except Exception as exc:  # model files can fail in many library-specific ways
                logger.exception("Failed to load model bundle %s", path)
                raise HTTPException(status_code=400, detail=f"模型加载失败：{exc}") from exc
            _PREDICTORS[key] = predictor
        _ACTIVE_BUNDLE = path.name
    return path.name, predictor


def _get_detector(bundle: str | None = None) -> tuple[str, SignDetector, Predictor]:
    name, predictor = _get_predictor(bundle)
    key = str(_resolve_bundle(name))
    with _REGISTRY_LOCK:
        detector = _DETECTORS.get(key)
        if detector is None:
            detector = SignDetector(predictor)
            _DETECTORS[key] = detector
    return name, detector, predictor


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


def _top_k(predictor: Predictor, image: np.ndarray, limit: int) -> list[dict[str, Any]]:
    if not hasattr(predictor.classifier, "predict_proba"):
        return []
    features = predictor.builder.extract_one(image).astype(np.float32, copy=False)
    scaled = predictor.scaler.transform(features[None, :])
    probabilities = np.asarray(predictor.classifier.predict_proba(scaled))[0]
    classes = np.asarray(predictor.classifier.classes_)
    order = np.argsort(probabilities)[::-1][:limit]
    return [
        {
            "class_id": int(classes[index]),
            "class_name": predictor.label_map.get(int(classes[index]), str(int(classes[index]))),
            "confidence": float(probabilities[index]),
        }
        for index in order
    ]


def _predict_payload(
    predictor: Predictor,
    image: np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    start = time.perf_counter()
    result = predictor.predict(image)
    elapsed = time.perf_counter() - start
    return {
        **result,
        "predict_seconds": elapsed,
        "top_k": _top_k(predictor, image, top_k),
        "cache": predictor.cache_stats(),
        "image": {"width": int(image.shape[1]), "height": int(image.shape[0])},
    }


def _model_metadata(path: Path) -> dict[str, Any]:
    key = str(path.resolve())
    predictor = _PREDICTORS.get(key)
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "modified_at": path.stat().st_mtime,
        "loaded": predictor is not None,
        "active": path.name == _ACTIVE_BUNDLE,
        "classifier": predictor.classifier.__class__.__name__ if predictor else None,
        "feature_mode": predictor.feature_config.get("mode") if predictor else None,
        "feature_dim": predictor.feature_dim if predictor else None,
        "cache": predictor.cache_stats() if predictor else None,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    bundles = _bundle_files()
    return {
        "status": "ok",
        "service": "traffic-sign-recognition",
        "model_available": bool(bundles),
        "active_model": _ACTIVE_BUNDLE,
        "loaded_models": len(_PREDICTORS),
        "labels": len(get_all_labels()),
    }


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
        "active_model": _ACTIVE_BUNDLE,
        "default_model": DEFAULT_BUNDLE_NAME,
        "bundles": [_model_metadata(path) for path in _bundle_files()],
    }


@app.post("/api/models/load")
async def load_model(request: ModelLoadRequest) -> dict[str, Any]:
    name, predictor = await run_in_threadpool(_get_predictor, request.name)
    path = _resolve_bundle(name)
    return {
        "ok": True,
        "model": _model_metadata(path),
        "summary": {
            "classifier": predictor.classifier.__class__.__name__,
            "feature_mode": predictor.feature_config.get("mode"),
            "feature_dim": predictor.feature_dim,
            "classes": len(predictor.label_map),
        },
    }


@app.delete("/api/models/cache")
async def clear_model_cache(bundle: str | None = Query(default=None)) -> dict[str, Any]:
    name, predictor = await run_in_threadpool(_get_predictor, bundle)
    predictor.clear_cache()
    return {"ok": True, "model": name, "cache": predictor.cache_stats()}


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(...),
    bundle: str | None = Query(default=None),
    top_k: int = Query(default=5, ge=1, le=10),
) -> dict[str, Any]:
    _raw, decoded = await _read_upload(image)
    name, predictor = await run_in_threadpool(_get_predictor, bundle)
    payload = await run_in_threadpool(_predict_payload, predictor, decoded, top_k=top_k)
    return {"model": name, "filename": image.filename, **payload}


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    bundle: str | None = Query(default=None),
) -> dict[str, Any]:
    _raw, decoded = await _read_upload(image)
    name, detector, predictor = await run_in_threadpool(_get_detector, bundle)
    start = time.perf_counter()
    detections = await run_in_threadpool(detector.detect, decoded)
    elapsed = time.perf_counter() - start
    return {
        "model": name,
        "filename": image.filename,
        "detections": detections,
        "count": len(detections),
        "detect_seconds": elapsed,
        "cache": predictor.cache_stats(),
        "image": {"width": int(decoded.shape[1]), "height": int(decoded.shape[0])},
    }


@app.post("/api/batch")
async def batch_predict(
    images: list[UploadFile] = File(...),
    bundle: str | None = Query(default=None),
) -> dict[str, Any]:
    if not images:
        raise HTTPException(status_code=400, detail="请至少选择一张图片")
    if len(images) > MAX_BATCH_FILES:
        raise HTTPException(status_code=400, detail=f"单次最多处理 {MAX_BATCH_FILES} 张图片")

    decoded_images: list[np.ndarray] = []
    filenames: list[str] = []
    for upload in images:
        _raw, decoded = await _read_upload(upload)
        decoded_images.append(decoded)
        filenames.append(upload.filename or f"image-{len(filenames) + 1}")

    name, predictor = await run_in_threadpool(_get_predictor, bundle)
    start = time.perf_counter()
    results = await run_in_threadpool(predictor.predict_batch, decoded_images)
    elapsed = time.perf_counter() - start
    return {
        "model": name,
        "count": len(results),
        "predict_seconds": elapsed,
        "items": [
            {"filename": filename, **result}
            for filename, result in zip(filenames, results, strict=True)
        ],
        "cache": predictor.cache_stats(),
    }


@app.websocket("/ws/stream")
async def stream_frames(
    websocket: WebSocket,
    bundle: str | None = Query(default=None),
    skip_frames: int = Query(default=1, ge=0, le=10),
) -> None:
    """Classify browser-sent JPEG frames and return compact JSON metadata."""
    await websocket.accept()
    try:
        name, predictor = await run_in_threadpool(_get_predictor, bundle)
        await websocket.send_json({"type": "ready", "model": name})
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "message": str(exc.detail)})
        await websocket.close(code=1008)
        return

    frame_index = 0
    processed = 0
    window_start = time.perf_counter()
    last_result: dict[str, Any] | None = None

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
            should_predict = last_result is None or frame_index % (skip_frames + 1) == 0
            inference_ms = 0.0
            if should_predict:
                started = time.perf_counter()
                last_result = await run_in_threadpool(predictor.predict, image)
                inference_ms = (time.perf_counter() - started) * 1000
                processed += 1

            now = time.perf_counter()
            elapsed = max(now - window_start, 1e-6)
            fps = processed / elapsed
            if elapsed >= 2.0:
                processed = 0
                window_start = now

            await websocket.send_json(
                {
                    "type": "prediction",
                    "frame_index": frame_index,
                    "result": {**(last_result or {}), "reused": not should_predict},
                    "predict_ms": inference_ms,
                    "fps": fps,
                    "cache": predictor.cache_stats(),
                }
            )
    except WebSocketDisconnect:
        pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("WebSocket stream failed")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1011)
        except RuntimeError:
            pass


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
