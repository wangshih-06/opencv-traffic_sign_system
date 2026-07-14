"""Lightweight IoU-based multi-object tracking for traffic-sign detections."""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import Any, Iterable


class SimpleTracker:
    """基于 IoU 的简单跟踪器，不需要深度学习。

    检测框统一使用 ``(x, y, width, height)``。匹配采用全局 IoU 从高到低的
    贪心分配；类别通过有限历史窗口多数投票稳定，置信度与位置使用指数平滑。
    未匹配轨迹在 ``max_lost`` 帧内仍会返回，并带有 ``lost_count > 0``，便于
    调用方画虚线框。
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 5,
        *,
        history_size: int = 7,
        bbox_smoothing: float = 0.65,
        confidence_smoothing: float = 0.65,
    ) -> None:
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold 必须在 [0, 1] 范围内")
        if max_lost < 0:
            raise ValueError("max_lost 必须 >= 0")
        if history_size < 1:
            raise ValueError("history_size 必须 >= 1")
        if not 0.0 <= bbox_smoothing < 1.0:
            raise ValueError("bbox_smoothing 必须在 [0, 1) 范围内")
        if not 0.0 <= confidence_smoothing < 1.0:
            raise ValueError("confidence_smoothing 必须在 [0, 1) 范围内")

        self.tracks: dict[int, dict[str, Any]] = {}
        self.next_id = 0
        self.iou_threshold = float(iou_threshold)
        self.max_lost = int(max_lost)
        self.history_size = int(history_size)
        self.bbox_smoothing = float(bbox_smoothing)
        self.confidence_smoothing = float(confidence_smoothing)
        self.last_update_seconds = 0.0

    def reset(self) -> None:
        """删除全部轨迹并从 ID 0 重新开始。"""
        self.tracks.clear()
        self.next_id = 0
        self.last_update_seconds = 0.0

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """输入当前帧检测结果，返回带 ``track_id`` 的稳定结果。

        - 与已有 track 做 IoU 匹配；
        - 匹配轨迹更新平滑框、置信度和类别历史；
        - 未匹配检测创建新轨迹；
        - 未匹配轨迹增加 ``lost_count``，超过 ``max_lost`` 后删除。
        """
        started = time.perf_counter()
        clean_detections = [self._normalise_detection(det) for det in detections]
        track_ids = list(self.tracks)

        # 构造所有超过阈值的候选对，然后从最大 IoU 开始做一对一分配。
        candidates: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            track_box = self.tracks[track_id]["bbox"]
            for det_index, detection in enumerate(clean_detections):
                iou = self._compute_iou(track_box, detection["bbox"])
                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, det_index))
        candidates.sort(key=lambda item: item[0], reverse=True)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for _iou, track_id, det_index in candidates:
            if track_id in matched_tracks or det_index in matched_detections:
                continue
            self._update_track(track_id, clean_detections[det_index])
            matched_tracks.add(track_id)
            matched_detections.add(det_index)

        # 保留短暂丢失轨迹，输出端可用虚线表示。
        for track_id in track_ids:
            if track_id in matched_tracks or track_id not in self.tracks:
                continue
            track = self.tracks[track_id]
            track["lost_count"] += 1
            if track["lost_count"] > self.max_lost:
                del self.tracks[track_id]

        for det_index, detection in enumerate(clean_detections):
            if det_index not in matched_detections:
                self._create_track(detection)

        output = [self._export_track(track_id) for track_id in sorted(self.tracks)]
        # 先显示当前帧真实匹配框，再显示短暂丢失框。
        output.sort(key=lambda item: (int(item["lost_count"]) > 0, int(item["track_id"])))
        self.last_update_seconds = time.perf_counter() - started
        return output

    def _create_track(self, detection: dict[str, Any]) -> None:
        track_id = self.next_id
        self.next_id += 1
        confidence = detection.get("confidence")
        class_id = int(detection["class_id"])
        class_name = str(detection.get("class_name", class_id))
        self.tracks[track_id] = {
            **detection,
            "bbox": tuple(float(v) for v in detection["bbox"]),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": None if confidence is None else float(confidence),
            "lost_count": 0,
            "age": 1,
            "hits": 1,
            "class_history": deque(
                [(class_id, class_name, 0.0 if confidence is None else float(confidence))],
                maxlen=self.history_size,
            ),
        }

    def _update_track(self, track_id: int, detection: dict[str, Any]) -> None:
        track = self.tracks[track_id]
        alpha = self.bbox_smoothing
        old_box = track["bbox"]
        new_box = detection["bbox"]
        track["bbox"] = tuple(
            alpha * float(old) + (1.0 - alpha) * float(new)
            for old, new in zip(old_box, new_box)
        )

        new_confidence = detection.get("confidence")
        old_confidence = track.get("confidence")
        if new_confidence is not None:
            new_confidence = float(new_confidence)
            if old_confidence is None:
                track["confidence"] = new_confidence
            else:
                conf_alpha = self.confidence_smoothing
                track["confidence"] = (
                    conf_alpha * float(old_confidence)
                    + (1.0 - conf_alpha) * new_confidence
                )

        class_id = int(detection["class_id"])
        class_name = str(detection.get("class_name", class_id))
        history = track["class_history"]
        history.append((class_id, class_name, 0.0 if new_confidence is None else new_confidence))

        # 保留检测器附加的 colour/shape 等元数据，但稳定字段由跟踪器管理。
        for key, value in detection.items():
            if key not in {"bbox", "class_id", "class_name", "confidence", "track_id", "lost_count"}:
                track[key] = value
        voted_id, voted_name = self._majority_vote(track_id)
        track["class_id"] = voted_id
        track["class_name"] = voted_name
        track["lost_count"] = 0
        track["age"] += 1
        track["hits"] += 1

    def _normalise_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(detection, dict):
            raise TypeError("每个 detection 必须是 dict")
        for key in ("bbox", "class_id"):
            if key not in detection:
                raise KeyError(f"detection 缺少必要字段: {key}")
        bbox = tuple(float(v) for v in detection["bbox"])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("bbox 必须是 (x, y, width, height)，且宽高 > 0")
        result = dict(detection)
        result["bbox"] = bbox
        result["class_id"] = int(result["class_id"])
        result["class_name"] = str(result.get("class_name", result["class_id"]))
        if result.get("confidence") is not None:
            result["confidence"] = float(result["confidence"])
        return result

    def _export_track(self, track_id: int) -> dict[str, Any]:
        track = self.tracks[track_id]
        exported = {
            key: value
            for key, value in track.items()
            if key != "class_history"
        }
        exported["bbox"] = tuple(int(round(v)) for v in track["bbox"])
        exported["track_id"] = int(track_id)
        return exported

    def _compute_iou(self, box1: Iterable[float], box2: Iterable[float]) -> float:
        """计算两个 ``(x, y, w, h)`` 检测框的 IoU。"""
        x1, y1, w1, h1 = (float(v) for v in box1)
        x2, y2, w2, h2 = (float(v) for v in box2)
        if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
            return 0.0

        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = w1 * h1 + w2 * h2 - intersection
        return 0.0 if union <= 0.0 else float(intersection / union)

    def _majority_vote(self, track_id: int) -> tuple[int, str]:
        """对 track 历史类别做多数投票，返回 ``(class_id, class_name)``。"""
        track = self.tracks[track_id]
        history = list(track.get("class_history", ()))
        if not history:
            return int(track["class_id"]), str(track["class_name"])

        counts = Counter(item[0] for item in history)
        max_count = max(counts.values())
        candidates = {class_id for class_id, count in counts.items() if count == max_count}
        if len(candidates) > 1:
            # 平票时优先累计置信度；仍平票时保留最近出现的类别。
            confidence_sum = {
                class_id: sum(conf for cid, _name, conf in history if cid == class_id)
                for class_id in candidates
            }
            best_confidence = max(confidence_sum.values())
            candidates = {
                class_id
                for class_id in candidates
                if confidence_sum[class_id] == best_confidence
            }
        voted_id = next(
            cid for cid, _name, _confidence in reversed(history) if cid in candidates
        )
        voted_name = next(
            name for cid, name, _confidence in reversed(history) if cid == voted_id
        )
        return int(voted_id), str(voted_name)
