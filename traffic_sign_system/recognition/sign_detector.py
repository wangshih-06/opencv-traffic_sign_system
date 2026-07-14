"""HSV/contour traffic-sign detector with scene-robust post-processing."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from traffic_sign_system.recognition.predictor import Predictor

_DEFAULT_RED_LOW1 = (0, 70, 55)
_DEFAULT_RED_HIGH1 = (12, 255, 255)
_DEFAULT_RED_LOW2 = (168, 70, 55)
_DEFAULT_RED_HIGH2 = (180, 255, 255)
_DEFAULT_BLUE_LOW = (88, 65, 50)
_DEFAULT_BLUE_HIGH = (142, 255, 255)
_DEFAULT_MIN_AREA = 400.0
_DEFAULT_MAX_AREA_RATIO = 0.3
_DEFAULT_MIN_ASPECT = 0.5
_DEFAULT_MAX_ASPECT = 2.0
_DEFAULT_MIN_CIRCULARITY = 0.4
_DEFAULT_CONFIDENCE_THRESHOLD = 0.5
_MORPH_KERNEL = (5, 5)

# GTSRB 类别的典型外形。未知/非标准类别不会被强制判为不匹配。
_TRIANGULAR_CLASSES = {11, 13, *range(18, 32)}
_OCTAGONAL_CLASSES = {14}
_DIAMOND_CLASSES = {12}
_CIRCULAR_CLASSES = {
    *range(0, 11), 15, 16, 17, 32, *range(33, 43)
}


class SignDetector:
    """用 HSV 色域、轮廓几何和类别形状一致性检测交通标志。"""

    def __init__(
        self,
        predictor: Predictor,
        red_hsv_low1: tuple[int, int, int] = _DEFAULT_RED_LOW1,
        red_hsv_high1: tuple[int, int, int] = _DEFAULT_RED_HIGH1,
        red_hsv_low2: tuple[int, int, int] = _DEFAULT_RED_LOW2,
        red_hsv_high2: tuple[int, int, int] = _DEFAULT_RED_HIGH2,
        blue_hsv_low: tuple[int, int, int] = _DEFAULT_BLUE_LOW,
        blue_hsv_high: tuple[int, int, int] = _DEFAULT_BLUE_HIGH,
        min_area: float = _DEFAULT_MIN_AREA,
        max_area_ratio: float = _DEFAULT_MAX_AREA_RATIO,
        min_aspect: float = _DEFAULT_MIN_ASPECT,
        max_aspect: float = _DEFAULT_MAX_ASPECT,
        min_circularity: float = _DEFAULT_MIN_CIRCULARITY,
        confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._predictor = predictor
        self.red_low1 = np.array(red_hsv_low1, dtype=np.uint8)
        self.red_high1 = np.array(red_hsv_high1, dtype=np.uint8)
        self.red_low2 = np.array(red_hsv_low2, dtype=np.uint8)
        self.red_high2 = np.array(red_hsv_high2, dtype=np.uint8)
        self.blue_low = np.array(blue_hsv_low, dtype=np.uint8)
        self.blue_high = np.array(blue_hsv_high, dtype=np.uint8)
        self.min_area = float(min_area)
        self.max_area_ratio = float(max_area_ratio)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)
        self.min_circularity = float(min_circularity)
        self.confidence_threshold = float(confidence_threshold)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _MORPH_KERNEL)

    def detect(self, img_bgr: np.ndarray) -> list[dict[str, Any]]:
        """检测并分类场景中的交通标志候选。"""
        if not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            raise ValueError("img_bgr 必须是非空图像")
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, self.red_low1, self.red_high1),
            cv2.inRange(hsv, self.red_low2, self.red_high2),
        )
        mask_blue = cv2.inRange(hsv, self.blue_low, self.blue_high)
        mask_red = self._morph_clean(mask_red)
        mask_blue = self._morph_clean(mask_blue)

        total_pixels = float(img_bgr.shape[0] * img_bgr.shape[1])
        results = self._process_mask(mask_red, img_bgr, total_pixels, colour="red")
        results.extend(
            self._process_mask(mask_blue, img_bgr, total_pixels, colour="blue")
        )
        return self._non_max_suppression(results, iou_threshold=0.45)

    def _morph_clean(self, mask: np.ndarray) -> np.ndarray:
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._kernel)

    def _process_mask(
        self,
        mask: np.ndarray,
        img_bgr: np.ndarray,
        total_pixels: float,
        colour: str,
    ) -> list[dict[str, Any]]:
        contours, _hierarchy = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        results: list[dict[str, Any]] = []
        for contour in contours:
            geometry = self._valid_contour(contour, total_pixels)
            if geometry is None:
                continue
            x, y, w, h = geometry["bbox"]
            # ?????????????????? 12% ??/?????
            # ??????????????????????
            pad_x, pad_y = max(2, round(w * 0.12)), max(2, round(h * 0.12))
            image_h, image_w = img_bgr.shape[:2]
            roi_x1, roi_y1 = max(0, x - pad_x), max(0, y - pad_y)
            roi_x2, roi_y2 = min(image_w, x + w + pad_x), min(image_h, y + h + pad_y)
            roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            if roi.size == 0:
                continue

            prediction = self._predictor.predict(roi)
            class_id = int(prediction["class_id"])
            confidence_raw = prediction.get("confidence")
            confidence = None if confidence_raw is None else float(confidence_raw)
            shape_match = self._shape_matches_class(class_id, geometry["shape"])
            size_match = self._size_matches_class(
                class_id, float(w * h) / max(total_pixels, 1.0)
            )

            # 高置信结果允许形状受遮挡/模糊影响；低置信且几何不一致时才剔除。
            if (
                confidence is not None
                and confidence < self.confidence_threshold
                and (shape_match is False or not size_match)
            ):
                continue

            results.append(
                {
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "class_id": class_id,
                    "class_name": str(prediction["class_name"]),
                    "confidence": confidence,
                    "colour": colour,
                    "detected_shape": geometry["shape"],
                    "shape_match": shape_match,
                    "size_match": size_match,
                    "circularity": geometry["circularity"],
                    "area_ratio": float(w * h) / max(total_pixels, 1.0),
                }
            )
        return results

    def _valid_contour(
        self, contour: np.ndarray, total_pixels: float
    ) -> dict[str, Any] | None:
        area = float(cv2.contourArea(contour))
        if area < self.min_area or area / max(total_pixels, 1.0) > self.max_area_ratio:
            return None
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            return None
        aspect = float(w) / float(h)
        if not self.min_aspect <= aspect <= self.max_aspect:
            return None
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (4.0 * math.pi * area) / (perimeter * perimeter + 1e-6)
        if circularity < self.min_circularity:
            return None

        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        vertices = len(approx)
        if vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            rect = cv2.minAreaRect(contour)
            angle = abs(float(rect[2]))
            shape = "diamond" if 25.0 <= angle <= 65.0 else "rectangle"
        elif 7 <= vertices <= 9:
            shape = "octagon"
        elif vertices >= 9 or circularity >= 0.72:
            shape = "circle"
        else:
            shape = "unknown"
        return {
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": area,
            "circularity": float(circularity),
            "vertices": vertices,
            "shape": shape,
        }

    @staticmethod
    def _shape_matches_class(class_id: int, detected_shape: str) -> bool | None:
        if class_id in _TRIANGULAR_CLASSES:
            return detected_shape == "triangle"
        if class_id in _OCTAGONAL_CLASSES:
            return detected_shape == "octagon"
        if class_id in _DIAMOND_CLASSES:
            return detected_shape == "diamond"
        if class_id in _CIRCULAR_CLASSES:
            # 遮挡或低分辨率圆形常被近似为八边形，二者均接受。
            return detected_shape in {"circle", "octagon"}
        return None

    @staticmethod
    def _size_matches_class(class_id: int, area_ratio: float) -> bool:
        # 典型画面占比采用宽松区间，作为低置信结果的辅助约束而非硬物理尺度。
        if class_id == 14:  # STOP 通常轮廓更醒目
            lower, upper = 0.00035, 0.22
        elif class_id in _TRIANGULAR_CLASSES:
            lower, upper = 0.00020, 0.18
        else:
            lower, upper = 0.00015, 0.15
        return lower <= area_ratio <= upper

    @staticmethod
    def _bbox_iou(box1, box2) -> float:
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        left, top = max(x1, x2), max(y1, y2)
        right, bottom = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
        inter = max(0, right - left) * max(0, bottom - top)
        union = w1 * h1 + w2 * h2 - inter
        return 0.0 if union <= 0 else inter / union

    def _non_max_suppression(
        self, detections: list[dict[str, Any]], iou_threshold: float
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            detections,
            key=lambda det: -1.0 if det.get("confidence") is None else float(det["confidence"]),
            reverse=True,
        )
        kept: list[dict[str, Any]] = []
        for detection in ordered:
            if all(
                self._bbox_iou(detection["bbox"], previous["bbox"]) < iou_threshold
                for previous in kept
            ):
                kept.append(detection)
        return kept


_COLOUR_MAP = {"red": (0, 0, 255), "blue": (255, 80, 0)}


def _draw_dashed_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    colour: tuple[int, int, int],
    thickness: int,
    dash: int = 9,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    for start in range(x1, x2, dash * 2):
        cv2.line(image, (start, y1), (min(start + dash, x2), y1), colour, thickness)
        cv2.line(image, (start, y2), (min(start + dash, x2), y2), colour, thickness)
    for start in range(y1, y2, dash * 2):
        cv2.line(image, (x1, start), (x1, min(start + dash, y2)), colour, thickness)
        cv2.line(image, (x2, start), (x2, min(start + dash, y2)), colour, thickness)


def draw_detections(
    img_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """绘制检测/跟踪框；``lost_count > 0`` 自动使用虚线框。"""
    annotated = np.ascontiguousarray(img_bgr).copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        x, y, w, h = (int(v) for v in detection["bbox"])
        x, y = max(0, x), max(0, y)
        x2, y2 = min(width - 1, x + max(1, w)), min(height - 1, y + max(1, h))
        colour = _COLOUR_MAP.get(str(detection.get("colour", "red")), (0, 200, 255))
        lost_count = int(detection.get("lost_count", 0))
        if lost_count > 0:
            _draw_dashed_rectangle(annotated, (x, y), (x2, y2), colour, thickness)
        else:
            cv2.rectangle(annotated, (x, y), (x2, y2), colour, thickness)

        track_text = (
            f"T{int(detection['track_id'])} " if detection.get("track_id") is not None else ""
        )
        confidence = detection.get("confidence")
        confidence_text = "" if confidence is None else f" {float(confidence):.0%}"
        lost_text = f" lost:{lost_count}" if lost_count > 0 else ""
        class_name = str(detection.get("class_name", ""))
        ascii_name = class_name.encode("ascii", "ignore").decode("ascii").strip()
        class_text = ascii_name or f"C{detection.get('class_id', '?')}"
        label = f"{track_text}{class_text}{confidence_text}{lost_text}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness)
        )
        label_top = max(0, y - text_h - baseline - 6)
        label_bottom = min(height - 1, label_top + text_h + baseline + 6)
        cv2.rectangle(
            annotated,
            (x, label_top),
            (min(width - 1, x + text_w + 6), label_bottom),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (x + 3, min(label_bottom - baseline - 2, height - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            max(1, thickness),
            cv2.LINE_AA,
        )
    return annotated
