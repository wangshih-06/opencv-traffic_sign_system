"""Validate temporal stability on a real video and print transition statistics.

Usage:
    python -m traffic_sign_system.scripts.verify_tracking VIDEO MODEL.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

from traffic_sign_system.recognition.predictor import Predictor
from traffic_sign_system.recognition.video_recognizer import VideoRecognizer


def transition_rate(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    return sum(a != b for a, b in zip(values, values[1:])) / (len(values) - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="验证视频帧间类别稳定性")
    parser.add_argument("video", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()

    recognizer = VideoRecognizer(Predictor(args.model), args.video, adaptive=args.adaptive)
    if not recognizer.open():
        raise SystemExit(f"无法打开视频: {args.video}")

    raw_classes: list[int] = []
    stable_classes: list[int] = []
    track_ids: list[int] = []
    tracker_ms: list[float] = []
    try:
        while len(stable_classes) < args.max_frames:
            ok, _frame, result = recognizer.read()
            if not ok:
                break
            raw = result.get("raw_detections", [])
            if raw:
                raw_classes.append(int(raw[0]["class_id"]))
            if result.get("track_id") is not None:
                stable_classes.append(int(result["class_id"]))
                track_ids.append(int(result["track_id"]))
            tracker_ms.append(float(result.get("tracker_seconds", 0.0)) * 1000.0)
    finally:
        recognizer.release()

    raw_rate = transition_rate(raw_classes)
    stable_rate = transition_rate(stable_classes)
    reduction = 0.0 if raw_rate == 0 else (raw_rate - stable_rate) / raw_rate
    mean_tracker_ms = sum(tracker_ms) / len(tracker_ms) if tracker_ms else 0.0
    print(f"有效跟踪帧: {len(stable_classes)}")
    print(f"原始类别跳变率: {raw_rate:.2%}")
    print(f"跟踪后跳变率: {stable_rate:.2%}")
    print(f"跳变率降低: {reduction:.2%}（目标 > 60%）")
    print(f"不同 track_id 数: {len(set(track_ids))}")
    print(f"平均跟踪开销: {mean_tracker_ms:.3f} ms/帧（目标 < 2 ms）")


if __name__ == "__main__":
    main()
