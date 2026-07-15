"""WebSocket multi-object detection/tracking protocol tests."""

from __future__ import annotations

import asyncio
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

app_module = importlib.import_module("traffic_sign_system.api.app")


class _FakePool:
    def __init__(self) -> None:
        self.calls = 0

    async def detect(self, _path: Path, _image: np.ndarray) -> dict:
        self.calls += 1
        x = 20 + self.calls
        return {
            "detections": [
                {
                    "bbox": (x, 25, 42, 42),
                    "class_id": 14,
                    "class_name": "\u505c\u8f66",
                    "confidence": 0.91,
                    "colour": "red",
                }
            ],
            "count": 1,
            "cache": {
                "hits": 0,
                "misses": self.calls,
                "total": self.calls,
                "hit_rate": 0.0,
                "size": self.calls,
                "maxsize": 256,
            },
            "scene": {
                "brightness": 96.0,
                "contrast": 48.0,
                "blur_score": 72.0,
                "noise_score": 0.04,
                "degradations": [],
                "quality_score": 97.5,
                "quality_status": "good",
                "quality_components": {
                    "brightness": 100.0,
                    "contrast": 100.0,
                    "sharpness": 100.0,
                    "noise": 87.5,
                },
                "analysis_seconds": 0.001,
                "recommendations": {},
            },
        }


class _FakeWebSocket:
    def __init__(self, frames: list[bytes]) -> None:
        self._messages = [{"bytes": frame} for frame in frames]
        self._messages.append({"type": "websocket.disconnect"})
        self.sent: list[dict] = []
        self.accepted = False
        self.closed_code: int | None = None

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict:
        return self._messages.pop(0)

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)

    async def close(self, code: int) -> None:
        self.closed_code = code


def _jpeg_frame() -> bytes:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("failed to encode test frame")
    return encoded.tobytes()


class StreamTrackingTests(unittest.TestCase):
    def test_primary_track_prefers_visible_object_over_lost_track(self) -> None:
        primary = app_module._select_primary_track(
            [
                {"track_id": 1, "confidence": 0.99, "lost_count": 2},
                {"track_id": 2, "confidence": 0.82, "lost_count": 0},
                {"track_id": 3, "confidence": 0.77, "lost_count": 0},
            ]
        )
        self.assertEqual(primary["track_id"], 2)
        self.assertIsNone(app_module._select_primary_track([]))

    def test_skipped_frame_reuses_latest_tracks_without_running_detector(self) -> None:
        websocket = _FakeWebSocket([_jpeg_frame(), _jpeg_frame(), _jpeg_frame()])
        pool = _FakePool()
        app_module.app.state.inference_pool = pool
        app_module.app.state.active_bundle = "fake.joblib"

        async def run() -> None:
            with patch.object(app_module, "_resolve_bundle", return_value=Path("fake.joblib")):
                await app_module.stream_frames(
                    websocket, bundle="fake.joblib", skip_frames=1
                )

        asyncio.run(run())
        predictions = [item for item in websocket.sent if item["type"] == "prediction"]
        self.assertEqual(pool.calls, 2)
        self.assertFalse(predictions[0]["reused"])
        self.assertTrue(predictions[1]["reused"])
        self.assertEqual(predictions[1]["predict_ms"], 0.0)
        self.assertTrue(predictions[1]["scene_reused"])
        self.assertEqual(predictions[1]["scene"]["quality_score"], 97.5)
        self.assertEqual(predictions[1]["detections"][0]["track_id"], 0)
        self.assertFalse(predictions[2]["reused"])

    def test_websocket_returns_stable_track_ids_and_frame_metadata(self) -> None:
        websocket = _FakeWebSocket([_jpeg_frame(), _jpeg_frame()])
        pool = _FakePool()
        app_module.app.state.inference_pool = pool
        app_module.app.state.active_bundle = "fake.joblib"

        async def run() -> None:
            with patch.object(
                app_module,
                "_resolve_bundle",
                return_value=Path("fake.joblib"),
            ):
                await app_module.stream_frames(
                    websocket,
                    bundle="fake.joblib",
                    skip_frames=0,
                )

        asyncio.run(run())

        self.assertTrue(websocket.accepted)
        self.assertEqual(websocket.sent[0]["type"], "ready")
        self.assertEqual(websocket.sent[0]["mode"], "detect-track")
        predictions = [item for item in websocket.sent if item["type"] == "prediction"]
        self.assertEqual(len(predictions), 2)
        self.assertEqual(pool.calls, 2)
        self.assertEqual(predictions[0]["detections"][0]["track_id"], 0)
        self.assertEqual(predictions[1]["detections"][0]["track_id"], 0)
        self.assertEqual(predictions[1]["detection_count"], 1)
        self.assertEqual(predictions[1]["processed_frames"], 2)
        self.assertEqual(predictions[1]["image"], {"width": 180, "height": 120})
        self.assertEqual(predictions[1]["scene"]["quality_status"], "good")
        self.assertIn("quality_components", predictions[1]["scene"])
        self.assertFalse(predictions[1]["scene_reused"])
        self.assertFalse(predictions[1]["reused"])


if __name__ == "__main__":
    unittest.main()
