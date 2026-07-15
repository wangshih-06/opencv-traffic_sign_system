"""Pluggable traffic-sign detection engines.

The project keeps the original HSV/contour detector as the zero-dependency
baseline and adds an optional OpenCV-DNN ONNX object detector.  The ONNX
backend is deliberately model-format tolerant: it accepts common exported
YOLO outputs (raw ``cx, cy, w, h`` tensors as well as NMS-ready ``xyxy``
rows).  A small JSON sidecar next to the model can describe class names and
thresholds.

Expected deep-model layout::

    traffic_sign_system/models/detectors/
    ├── traffic_sign_detector.onnx
    └── traffic_sign_detector.json  # optional

Example sidecar::

    {
      "input_size": 640,
      "num_classes": 43,
      "labels": {"0": "限速20公里/小时"},
      "confidence_threshold": 0.35,
      "nms_threshold": 0.45,
      "output_format": "auto"
    }

The ``hybrid`` engine runs both backends and merges overlapping boxes while
retaining source/confidence metadata.  If no deep model is installed, hybrid
mode intentionally degrades to the traditional engine instead of breaking an
existing local deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from traffic_sign_system.config.labels import GTSRB_LABELS
from traffic_sign_system.recognition.sign_detector import SignDetector

DetectionEngine = Literal["traditional", "deep", "hybrid"]
ENGINE_IDS: tuple[str, ...] = ("traditional", "deep", "hybrid")


class DeepOnnxDetector:
    """OpenCV-DNN ONNX object detector for traffic-sign scenes.

    The detector does not require ``onnxruntime``; OpenCV's DNN module is
    already part of the project's base dependency.  A ``net`` argument is
    accepted for deterministic unit tests and for applications that manage a
    shared OpenCV network themselves.
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        net: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if net is None and not self.model_path.is_file():
            raise FileNotFoundError(f"深度检测模型不存在: {self.model_path}")
        self.metadata = dict(metadata) if metadata is not None else self._load_metadata()
        self.input_size = self._resolve_input_size(self.metadata.get("input_size", 640))
        self.confidence_threshold = float(
            self.metadata.get("confidence_threshold", 0.35)
        )
        self.nms_threshold = float(self.metadata.get("nms_threshold", 0.45))
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.nms_threshold <= 1.0:
            raise ValueError("nms_threshold must be between 0 and 1")
        self.output_format = str(self.metadata.get("output_format", "auto")).lower()
        self.num_classes = int(self.metadata.get("num_classes", 43))
        self.label_map = self._resolve_labels(self.metadata.get("labels"))
        self.net = net if net is not None else cv2.dnn.readNetFromONNX(str(self.model_path))
        self.last_inference_ms = 0.0

    def detect(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        image = self._validate_image(image_bgr)
        started = cv2.getTickCount()
        blob, scale, pad_x, pad_y = self._letterbox_blob(image)
        self.net.setInput(blob)
        outputs = self.net.forward()
        detections = self._decode_outputs(
            outputs,
            image_width=image.shape[1],
            image_height=image.shape[0],
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
        )
        self.last_inference_ms = (
            (cv2.getTickCount() - started) / cv2.getTickFrequency() * 1000.0
        )
        return detections

    def metadata_summary(self) -> dict[str, Any]:
        return {
            "name": self.model_path.name,
            "input_size": self.input_size,
            "classes": len(self.label_map),
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "output_format": self.output_format,
        }

    def _load_metadata(self) -> dict[str, Any]:
        sidecar = self.model_path.with_suffix(".json")
        if not sidecar.is_file():
            return {}
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"深度检测模型配置无效: {sidecar}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"深度检测模型配置必须是 JSON 对象: {sidecar}")
        return payload

    @staticmethod
    def _resolve_input_size(value: Any) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            width, height = int(value[0]), int(value[1])
        else:
            width = height = int(value)
        if width < 32 or height < 32 or width > 4096 or height > 4096:
            raise ValueError("深度模型 input_size 必须位于 32~4096 范围内")
        return width, height

    def _resolve_labels(self, labels: Any) -> dict[int, str]:
        if labels is None:
            return dict(GTSRB_LABELS) if self.num_classes == len(GTSRB_LABELS) else {
                index: str(index) for index in range(self.num_classes)
            }
        if isinstance(labels, list):
            return {index: str(value) for index, value in enumerate(labels)}
        if isinstance(labels, dict):
            resolved = {int(key): str(value) for key, value in labels.items()}
            if resolved:
                return resolved
        raise ValueError("深度模型 labels 必须是数组或对象")

    def _letterbox_blob(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        input_w, input_h = self.input_size
        image_h, image_w = image.shape[:2]
        scale = min(input_w / max(image_w, 1), input_h / max(image_h, 1))
        resized_w = max(1, int(round(image_w * scale)))
        resized_h = max(1, int(round(image_h * scale)))
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - resized_w) // 2
        pad_y = (input_h - resized_h) // 2
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(input_w, input_h),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        return blob, scale, pad_x, pad_y

    def _decode_outputs(
        self,
        outputs: Any,
        *,
        image_width: int,
        image_height: int,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[dict[str, Any]]:
        arrays = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        candidates: list[tuple[float, int, tuple[float, float, float, float]]] = []
        for output in arrays:
            arr = np.asarray(output)
            if arr.size == 0:
                continue
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2:
                continue
            feature_dims = {
                6,
                7,
                self.num_classes + 4,
                self.num_classes + 5,
            }
            if arr.shape[0] in feature_dims and arr.shape[1] not in feature_dims:
                arr = arr.T
            candidates.extend(self._decode_rows(arr, scale, pad_x, pad_y))

        if not candidates:
            return []
        boxes = [self._xyxy_to_xywh(item[2]) for item in candidates]
        scores = [float(item[0]) for item in candidates]
        kept_indices = self._class_aware_nms(
            boxes,
            scores,
            [item[1] for item in candidates],
            self.confidence_threshold,
            self.nms_threshold,
        )
        results: list[dict[str, Any]] = []
        for index in kept_indices:
            confidence, class_id, (x1, y1, x2, y2) = candidates[index]
            x1 = max(0.0, min(float(image_width - 1), x1))
            y1 = max(0.0, min(float(image_height - 1), y1))
            x2 = max(x1 + 1.0, min(float(image_width), x2))
            y2 = max(y1 + 1.0, min(float(image_height), y2))
            width, height = x2 - x1, y2 - y1
            if width < 2.0 or height < 2.0:
                continue
            results.append(
                {
                    "bbox": (round(x1), round(y1), round(width), round(height)),
                    "class_id": int(class_id),
                    "class_name": self.label_map.get(int(class_id), str(class_id)),
                    "confidence": float(np.clip(confidence, 0.0, 1.0)),
                    "colour": self._estimate_colour_placeholder(),
                    "engine": "deep",
                    "sources": ["deep"],
                }
            )
        return results

    def _decode_rows(
        self,
        rows: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[tuple[float, int, tuple[float, float, float, float]]]:
        columns = int(rows.shape[1])
        result: list[tuple[float, int, tuple[float, float, float, float]]] = []
        input_w, input_h = self.input_size
        raw_formats = {"yolov5", "yolov8", "raw_objectness", "raw_no_objectness"}
        for row in rows.astype(np.float32, copy=False):
            if columns == 7 and self.output_format not in raw_formats:
                # OpenCV/SSD NMS format: batch, class, score, x1, y1, x2, y2.
                class_id, confidence = int(row[1]), float(row[2])
                coords = row[3:7].astype(float)
                if np.max(np.abs(coords)) <= 1.5:
                    coords[[0, 2]] *= input_w
                    coords[[1, 3]] *= input_h
                box = self._restore_box(coords, scale, pad_x, pad_y)
            elif columns == 6 and self.output_format not in raw_formats:
                # NMS-ready YOLO format: x1, y1, x2, y2, score, class_id.
                coords = row[:4].astype(float)
                confidence, class_id = float(row[4]), int(round(float(row[5])))
                if np.max(np.abs(coords)) <= 1.5:
                    coords[[0, 2]] *= input_w
                    coords[[1, 3]] *= input_h
                box = self._restore_box(coords, scale, pad_x, pad_y)
            else:
                # YOLO raw format: cx, cy, w, h, [objectness], class scores.
                has_objectness = columns == self.num_classes + 5
                if columns not in {self.num_classes + 4, self.num_classes + 5}:
                    if self.output_format in {"yolov5", "raw_objectness"}:
                        has_objectness = True
                    elif self.output_format in {"yolov8", "raw_no_objectness"}:
                        has_objectness = False
                    else:
                        continue
                objectness = float(row[4]) if has_objectness else 1.0
                class_scores = row[5:] if has_objectness else row[4:]
                if not len(class_scores):
                    continue
                class_id = int(np.argmax(class_scores))
                confidence = objectness * float(class_scores[class_id])
                cx, cy, width, height = (float(value) for value in row[:4])
                if max(abs(cx), abs(cy), abs(width), abs(height)) <= 2.0:
                    cx, width = cx * input_w, width * input_w
                    cy, height = cy * input_h, height * input_h
                coords = np.array(
                    [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
                    dtype=float,
                )
                box = self._restore_box(coords, scale, pad_x, pad_y)
            if confidence >= self.confidence_threshold and box[2] > box[0] and box[3] > box[1]:
                result.append((confidence, class_id, box))
        return result

    @staticmethod
    def _restore_box(
        coords: np.ndarray | tuple[float, float, float, float],
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = (float(value) for value in coords)
        safe_scale = max(float(scale), 1e-8)
        return (
            (x1 - pad_x) / safe_scale,
            (y1 - pad_y) / safe_scale,
            (x2 - pad_x) / safe_scale,
            (y2 - pad_y) / safe_scale,
        )

    @staticmethod
    def _xyxy_to_xywh(box: tuple[float, float, float, float]) -> list[int]:
        x1, y1, x2, y2 = box
        return [round(x1), round(y1), max(1, round(x2 - x1)), max(1, round(y2 - y1))]

    @staticmethod
    def _flatten_indices(indices: Any) -> list[int]:
        if indices is None:
            return []
        return [int(item[0] if isinstance(item, (list, tuple, np.ndarray)) else item) for item in indices]

    @classmethod
    def _class_aware_nms(
        cls,
        boxes: list[list[int]],
        scores: list[float],
        class_ids: list[int],
        confidence_threshold: float,
        nms_threshold: float,
    ) -> list[int]:
        """Run NMS independently per class so adjacent signs are not suppressed."""
        kept: list[int] = []
        for class_id in sorted(set(class_ids)):
            indices = [index for index, value in enumerate(class_ids) if value == class_id]
            class_indices = cls._flatten_indices(
                cv2.dnn.NMSBoxes(
                    [boxes[index] for index in indices],
                    [scores[index] for index in indices],
                    confidence_threshold,
                    nms_threshold,
                )
            )
            kept.extend(indices[index] for index in class_indices)
        return sorted(kept, key=lambda index: scores[index], reverse=True)

    @staticmethod
    def _estimate_colour_placeholder() -> str:
        # Colour is refined by ``add_colour_metadata`` after the detector has
        # access to the source image.  Keep a valid value for callers that use
        # DeepOnnxDetector directly.
        return "blue"

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("image 必须是非空 numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image 必须是 BGR 三通道图像")
        if image.dtype != np.uint8:
            return np.clip(image, 0, 255).astype(np.uint8)
        return image


def add_colour_metadata(
    image_bgr: np.ndarray, detections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Estimate red/blue display colour without changing detector classes."""
    image = DeepOnnxDetector._validate_image(image_bgr)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for detection in detections:
        x, y, width, height = (int(value) for value in detection["bbox"])
        roi = hsv[max(0, y):min(hsv.shape[0], y + height), max(0, x):min(hsv.shape[1], x + width)]
        if roi.size == 0:
            continue
        red = cv2.bitwise_or(
            cv2.inRange(roi, np.array((0, 60, 40), dtype=np.uint8), np.array((12, 255, 255), dtype=np.uint8)),
            cv2.inRange(roi, np.array((168, 60, 40), dtype=np.uint8), np.array((180, 255, 255), dtype=np.uint8)),
        )
        blue = cv2.inRange(roi, np.array((88, 55, 40), dtype=np.uint8), np.array((142, 255, 255), dtype=np.uint8))
        detection["colour"] = "red" if int(np.count_nonzero(red)) >= int(np.count_nonzero(blue)) else "blue"
    return detections


def _bbox_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    left, top = max(x1, x2), max(y1, y2)
    right, bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = w1 * h1 + w2 * h2 - intersection
    return 0.0 if union <= 0 else intersection / union


def merge_detections(
    traditional: list[dict[str, Any]],
    deep: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.35,
) -> list[dict[str, Any]]:
    """Merge two detector outputs while preserving provenance metadata."""
    merged: list[dict[str, Any]] = []
    used_traditional: set[int] = set()
    for deep_item in sorted(deep, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
        matches = [
            (index, item)
            for index, item in enumerate(traditional)
            if index not in used_traditional
            and _bbox_iou(tuple(deep_item["bbox"]), tuple(item["bbox"])) >= iou_threshold
        ]
        if matches:
            index, traditional_item = max(
                matches,
                key=lambda pair: _bbox_iou(tuple(deep_item["bbox"]), tuple(pair[1]["bbox"])),
            )
            used_traditional.add(index)
            combined = dict(traditional_item)
            combined["engine"] = "hybrid"
            combined["sources"] = ["traditional", "deep"]
            combined["detector_confidence"] = deep_item.get("confidence")
            combined["deep_class_id"] = deep_item.get("class_id")
            combined["deep_class_name"] = deep_item.get("class_name")
            merged.append(combined)
        else:
            combined = dict(deep_item)
            combined["engine"] = "hybrid"
            combined["sources"] = ["deep"]
            merged.append(combined)

    for index, item in enumerate(traditional):
        if index not in used_traditional:
            combined = dict(item)
            combined["engine"] = "hybrid"
            combined["sources"] = ["traditional"]
            merged.append(combined)
    return sorted(
        merged,
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )


def run_detection_engine(
    predictor: Any | None,
    image_bgr: np.ndarray,
    engine: str,
    *,
    deep_detector: DeepOnnxDetector | None = None,
) -> dict[str, Any]:
    """Run one of the public engines and return detections plus execution metadata."""
    if engine not in ENGINE_IDS:
        raise ValueError(f"不支持的检测引擎: {engine}")
    traditional: list[dict[str, Any]] = []
    if engine in {"traditional", "hybrid"}:
        if predictor is None:
            raise ValueError("传统检测引擎需要分类模型")
        traditional = SignDetector(predictor).detect(image_bgr)
        for item in traditional:
            item.setdefault("engine", "traditional")
            item.setdefault("sources", ["traditional"])
    if engine == "traditional":
        return {"detections": traditional, "engine_used": "traditional", "fallback": False}

    if deep_detector is None:
        if engine == "deep":
            raise FileNotFoundError("未配置 ONNX 深度检测模型，请先放入 models/detectors")
        return {
            "detections": traditional,
            "engine_used": "traditional",
            "fallback": True,
            "warning": "未找到 ONNX 深度检测模型，混合引擎已回退到传统引擎",
        }

    try:
        deep = add_colour_metadata(image_bgr, deep_detector.detect(image_bgr))
    except Exception:
        if engine == "deep":
            raise
        return {
            "detections": traditional,
            "engine_used": "traditional",
            "fallback": True,
            "warning": "ONNX 深度检测失败，混合引擎已回退到传统引擎",
        }
    if engine == "deep":
        return {
            "detections": deep,
            "engine_used": "deep",
            "fallback": False,
            "deep_inference_ms": deep_detector.last_inference_ms,
        }
    return {
        "detections": merge_detections(traditional, deep),
        "engine_used": "hybrid",
        "fallback": False,
        "deep_inference_ms": deep_detector.last_inference_ms,
    }


def list_engine_metadata(detector_dir: Path) -> list[dict[str, Any]]:
    """Return safe, filesystem-only metadata for API/UI engine discovery."""
    models = sorted(detector_dir.glob("*.onnx")) if detector_dir.is_dir() else []
    deep_models = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": path.stat().st_mtime,
            "metadata": path.with_suffix(".json").is_file(),
        }
        for path in models
    ]
    return deep_models
