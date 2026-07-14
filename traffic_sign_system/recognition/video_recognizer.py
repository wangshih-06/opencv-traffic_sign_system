"""Frame-by-frame video traffic-sign detection with IoU tracking."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from traffic_sign_system.recognition.predictor import Predictor
from traffic_sign_system.recognition.scene_aware import SceneAnalyzer
from traffic_sign_system.recognition.sign_detector import SignDetector, draw_detections
from traffic_sign_system.recognition.tracker import SimpleTracker


class VideoRecognizer:
    """读取视频、检测交通标志，并用帧间跟踪稳定类别与置信度。"""

    def __init__(
        self,
        predictor: Predictor,
        src: str | Path | int,
        *,
        skip_frames: int = 0,
        roi: tuple[int, int, int, int] | None = None,
        detector: SignDetector | None = None,
        tracker: SimpleTracker | None = None,
        scene_analyzer: SceneAnalyzer | None = None,
        adaptive: bool = False,
    ) -> None:
        if skip_frames < 0:
            raise ValueError("skip_frames 必须 >= 0")
        if roi is not None and (len(roi) != 4 or min(roi) < 0):
            raise ValueError("roi 必须是 (x, y, w, h) 且各项 >= 0")

        self.predictor = predictor
        self.src = src
        self.skip_frames = int(skip_frames)
        self.roi = None if roi is None else tuple(int(v) for v in roi)
        self.detector = detector if detector is not None else SignDetector(predictor)
        self.tracker = tracker if tracker is not None else SimpleTracker()
        self.scene_analyzer = scene_analyzer if scene_analyzer is not None else SceneAnalyzer()
        self.adaptive = bool(adaptive)

        self.capture: cv2.VideoCapture | None = None
        self.frame_index = 0
        self.frame_count = 0
        self.source_fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.last_predict_seconds = 0.0
        self.last_frame_seconds = 0.0
        self.last_fps = 0.0
        self._last_result: dict[str, Any] | None = None
        self.frames_predicted = 0
        self.frames_reused = 0

    def open(self) -> bool:
        """打开视频/摄像头源并重置跟踪状态。"""
        self.release()
        source = str(self.src) if isinstance(self.src, Path) else self.src
        self.capture = cv2.VideoCapture(source)
        if self.capture is None or not self.capture.isOpened():
            self.release()
            return False
        self.frame_index = 0
        self.frame_count = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_width = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
        self.frame_height = max(0, int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
        return True

    @property
    def is_opened(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def read(self) -> tuple[bool, np.ndarray | None, dict[str, Any]]:
        """读取、检测、跟踪并标注一帧。"""
        started = time.perf_counter()
        ok, frame = self._read_raw_frame()
        if not ok or frame is None:
            return False, None, {}
        result = self.process_frame(frame, started=started)
        annotated = self.annotate_frame(frame, result)
        if self.roi is not None:
            annotated = self._draw_roi(annotated)
        return True, annotated, result

    def process_frame(
        self, frame: np.ndarray, *, started: float | None = None
    ) -> dict[str, Any]:
        """处理已读取帧，便于测试或外部视频循环复用。"""
        frame_started = time.perf_counter() if started is None else started
        if frame is None or frame.size == 0:
            raise ValueError("frame 不能为空")

        analysis = self.scene_analyzer.analyze(frame)
        self._apply_scene_params(analysis)
        crop = self._crop_roi(frame)
        if crop is None:
            raise ValueError("ROI 超出图像范围")

        should_predict = (
            self._last_result is None
            or self.skip_frames <= 0
            or self.frame_index % (self.skip_frames + 1) == 0
        )
        reused = not should_predict
        if should_predict:
            predict_started = time.perf_counter()
            detections = self.detector.detect(crop)
            detections = self._offset_detections(detections)
            tracked = self.tracker.update(detections)
            self.last_predict_seconds = time.perf_counter() - predict_started
            self.frames_predicted += 1
            result = self._build_result(tracked)
            result["raw_detections"] = [dict(item) for item in detections]
            self._last_result = dict(result)
            self._last_result["detections"] = [dict(item) for item in tracked]
        else:
            assert self._last_result is not None
            result = dict(self._last_result)
            result["detections"] = [dict(item) for item in self._last_result.get("detections", [])]
            self.last_predict_seconds = 0.0
            self.frames_reused += 1

        self.last_frame_seconds = max(time.perf_counter() - frame_started, 1e-9)
        self.last_fps = 1.0 / self.last_frame_seconds
        result.update(
            predict_seconds=self.last_predict_seconds,
            frame_seconds=self.last_frame_seconds,
            fps=self.last_fps,
            frame_index=self.frame_index,
            reused=reused,
            skip_frames=self.skip_frames,
            tracker_seconds=self.tracker.last_update_seconds if should_predict else 0.0,
            scene_analysis=analysis,
            adaptive=self.adaptive,
        )
        if self.roi is not None:
            result["roi"] = self.roi
        return result

    def _apply_scene_params(self, analysis: dict[str, Any]) -> None:
        preprocessor = getattr(self.predictor, "preprocessor", None)
        if preprocessor is None:
            return
        if hasattr(preprocessor, "set_adaptive"):
            preprocessor.set_adaptive(self.adaptive)
        else:
            preprocessor.adaptive = self.adaptive
        if self.adaptive and hasattr(preprocessor, "set_runtime_params"):
            preprocessor.set_runtime_params(**self.scene_analyzer.recommend_params(analysis))
        elif hasattr(preprocessor, "clear_runtime_params"):
            preprocessor.clear_runtime_params()

    def _build_result(self, detections: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"detections": detections, "detection_count": len(detections)}
        if not detections:
            result.update(class_id=-1, class_name="未检测到交通标志", confidence=None, track_id=None)
            return result
        primary = min(
            detections,
            key=lambda det: (
                int(det.get("lost_count", 0)) > 0,
                -(float(det.get("confidence")) if det.get("confidence") is not None else -1.0),
            ),
        )
        result.update(
            class_id=int(primary["class_id"]),
            class_name=str(primary["class_name"]),
            confidence=primary.get("confidence"),
            track_id=int(primary["track_id"]),
            lost_count=int(primary.get("lost_count", 0)),
        )
        return result

    def _offset_detections(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.roi is None:
            return detections
        offset_x, offset_y, _w, _h = self.roi
        shifted: list[dict[str, Any]] = []
        for detection in detections:
            item = dict(detection)
            x, y, w, h = item["bbox"]
            item["bbox"] = (int(x + offset_x), int(y + offset_y), int(w), int(h))
            shifted.append(item)
        return shifted

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray | None:
        if self.roi is None:
            return frame
        x, y, w, h = self.roi
        frame_height, frame_width = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_width, x + w), min(frame_height, y + h)
        if x1 >= x2 or y1 >= y2:
            return None
        crop = np.ascontiguousarray(frame[y1:y2, x1:x2])
        return crop if crop.size else None

    def _draw_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        x, y, w, h = self.roi
        frame_height, frame_width = frame.shape[:2]
        cv2.rectangle(
            frame,
            (x, y),
            (min(frame_width - 1, x + w), min(frame_height - 1, y + h)),
            (0, 255, 255),
            max(1, round(min(frame_width, frame_height) / 250)),
        )
        return frame

    def _read_raw_frame(self) -> tuple[bool, np.ndarray | None]:
        if not self.is_opened:
            return False, None
        assert self.capture is not None
        ok, frame = self.capture.read()
        if not ok or frame is None or frame.size == 0:
            return False, None
        self.frame_index += 1
        if self.frame_width <= 0 or self.frame_height <= 0:
            self.frame_height, self.frame_width = frame.shape[:2]
        return True, frame

    @staticmethod
    def annotate_frame(frame: np.ndarray, result: dict[str, Any]) -> np.ndarray:
        """绘制跟踪框、track_id、主结果和性能指标。"""
        output = draw_detections(frame, list(result.get("detections", [])))
        track_id = result.get("track_id")
        track_text = "--" if track_id is None else str(track_id)
        confidence = result.get("confidence")
        confidence_text = "N/A" if confidence is None else f"{float(confidence):.1%}"
        lines = (
            f"Track: {track_text}  Class ID: {int(result.get('class_id', -1))}",
            f"Confidence: {confidence_text}  Objects: {int(result.get('detection_count', 0))}",
            f"Detect: {float(result.get('predict_seconds', 0.0)) * 1000:.1f} ms  "
            f"Track: {float(result.get('tracker_seconds', 0.0)) * 1000:.2f} ms  "
            f"FPS: {float(result.get('fps', 0.0)):.1f}",
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(0.8, output.shape[1] / 1000.0))
        thickness = max(1, round(scale * 2))
        line_height = max(22, round(29 * scale))
        box_height = line_height * len(lines) + 10
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (output.shape[1], box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, output, 0.35, 0, output)
        for index, line in enumerate(lines):
            cv2.putText(
                output,
                line,
                (10, 7 + (index + 1) * line_height - 5),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
        return output

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        self._last_result = None
        self.frames_predicted = 0
        self.frames_reused = 0
        self.tracker.reset()

    def skip_stats(self) -> dict[str, int | float]:
        total = self.frames_predicted + self.frames_reused
        return {
            "predicted": self.frames_predicted,
            "reused": self.frames_reused,
            "total": total,
            "reuse_rate": self.frames_reused / total if total else 0.0,
            "skip_frames": self.skip_frames,
        }
