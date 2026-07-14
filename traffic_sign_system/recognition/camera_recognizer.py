"""Camera recognizer with centered detection ROI and frame-to-frame tracking."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from traffic_sign_system.recognition.predictor import Predictor
from traffic_sign_system.recognition.scene_aware import SceneAnalyzer
from traffic_sign_system.recognition.sign_detector import SignDetector
from traffic_sign_system.recognition.tracker import SimpleTracker
from traffic_sign_system.recognition.video_recognizer import VideoRecognizer


class CameraRecognizer(VideoRecognizer):
    """在摄像头中心 ROI 内检测，并显示稳定跟踪框与 track_id。"""

    def __init__(
        self,
        predictor: Predictor,
        src: int = 0,
        roi_size: int | tuple[int, int] = 64,
        *,
        skip_frames: int = 1,
        smooth_window: int = 7,
        detector: SignDetector | None = None,
        tracker: SimpleTracker | None = None,
        scene_analyzer: SceneAnalyzer | None = None,
        adaptive: bool = False,
    ) -> None:
        if isinstance(roi_size, int):
            roi_size = (roi_size, roi_size)
        if len(roi_size) != 2 or min(roi_size) <= 0:
            raise ValueError("roi_size 必须是正整数或 (width, height)。")
        if smooth_window < 1:
            raise ValueError("smooth_window 必须 >= 1")
        if tracker is None:
            tracker = SimpleTracker(history_size=int(smooth_window))
        super().__init__(
            predictor,
            int(src),
            skip_frames=skip_frames,
            roi=None,
            detector=detector,
            tracker=tracker,
            scene_analyzer=scene_analyzer,
            adaptive=adaptive,
        )
        self.roi_size = (int(roi_size[0]), int(roi_size[1]))
        self.smooth_window = int(smooth_window)

    def open(self) -> bool:
        ok = super().open()
        if ok and self.frame_width > 0 and self.frame_height > 0:
            self._update_center_roi()
        return ok

    def _read_raw_frame(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = super()._read_raw_frame()
        if ok and frame is not None and self.roi is None:
            self.frame_height, self.frame_width = frame.shape[:2]
            self._update_center_roi()
        return ok, frame

    def _update_center_roi(self) -> None:
        width = min(self.frame_width, self.roi_size[0])
        height = min(self.frame_height, self.roi_size[1])
        x = max(0, (self.frame_width - width) // 2)
        y = max(0, (self.frame_height - height) // 2)
        self.roi = (x, y, width, height)

    def _draw_roi(self, frame: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return frame
        x, y, w, h = self.roi
        frame_height, frame_width = frame.shape[:2]
        red = (0, 0, 255)
        thickness = max(2, round(min(frame_width, frame_height) / 220))
        cv2.rectangle(
            frame,
            (x, y),
            (min(frame_width - 1, x + w), min(frame_height - 1, y + h)),
            red,
            thickness,
        )
        cv2.putText(
            frame,
            "DETECTION ROI",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            red,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
        return frame
