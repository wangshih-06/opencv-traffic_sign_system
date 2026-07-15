"""Tests for batch response consistency and per-file fault isolation."""

from __future__ import annotations

import asyncio
import importlib
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi import UploadFile
from starlette.datastructures import Headers

app_module = importlib.import_module("traffic_sign_system.api.app")


def _png_bytes(seed: int) -> bytes:
    image = np.full((32, 32, 3), seed, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("failed to encode fixture")
    return encoded.tobytes()


def _upload(filename: str, payload: bytes, content_type: str = "image/png") -> UploadFile:
    return UploadFile(
        filename=filename,
        file=BytesIO(payload),
        headers=Headers({"content-type": content_type}),
    )


class _BatchPool:
    def __init__(self, *, fail_batch: bool = False, fail_seed: int | None = None) -> None:
        self.fail_batch = fail_batch
        self.fail_seed = fail_seed
        self.batch_calls = 0
        self.single_calls = 0

    async def predict_batch(self, _path: Path, images: list[np.ndarray]) -> list[dict]:
        self.batch_calls += 1
        if self.fail_batch:
            raise ValueError("vectorized feature extraction failed")
        return [self._result(image) for image in images]

    async def predict(self, _path: Path, image: np.ndarray, top_k: int = 1) -> dict:
        self.single_calls += 1
        if self.fail_seed is not None and int(image[0, 0, 0]) == self.fail_seed:
            raise ValueError("single image inference failed")
        result = self._result(image)
        return {
            "result": result,
            "top_k": [result][:top_k],
            "cache": {"hits": 0, "misses": self.single_calls, "total": self.single_calls, "hit_rate": 0.0, "size": 0, "maxsize": 256},
        }

    def cache_stats_for(self, _path: Path) -> dict:
        return {"hits": 0, "misses": self.single_calls, "total": self.single_calls, "hit_rate": 0.0, "size": 0, "maxsize": 256}

    @staticmethod
    def _result(image: np.ndarray) -> dict:
        class_id = int(image[0, 0, 0]) % 3
        return {"class_id": class_id, "class_name": f"class_{class_id}", "confidence": 0.8}


class BatchApiTests(unittest.TestCase):
    def _run(self, uploads: list[UploadFile], pool: _BatchPool) -> dict:
        app_module.app.state.inference_pool = pool
        app_module.app.state.active_bundle = "fake.joblib"

        async def call() -> dict:
            with patch.object(app_module, "_resolve_bundle", return_value=Path("fake.joblib")):
                return await app_module.batch_predict(uploads, bundle="fake.joblib")

        return asyncio.run(call())

    def test_success_response_has_single_prediction_shape_and_cache(self) -> None:
        response = self._run([_upload("good.png", _png_bytes(1))], _BatchPool())

        self.assertEqual(response["count"], 1)
        self.assertEqual(response["success_count"], 1)
        self.assertEqual(response["failed_count"], 0)
        self.assertIn("cache", response)
        item = response["items"][0]
        self.assertTrue(item["ok"])
        self.assertEqual(item["model"], "fake.joblib")
        self.assertEqual(item["top_k"][0]["class_id"], item["class_id"])
        self.assertEqual(item["image"], {"width": 32, "height": 32})
        self.assertIsNone(item["error"])

    def test_invalid_file_does_not_abort_other_files(self) -> None:
        pool = _BatchPool()
        response = self._run(
            [
                _upload("good.png", _png_bytes(1)),
                _upload("broken.png", b"not-an-image"),
            ],
            pool,
        )

        self.assertEqual(response["count"], 2)
        self.assertEqual(response["success_count"], 1)
        self.assertEqual(response["failed_count"], 1)
        self.assertTrue(response["items"][0]["ok"])
        self.assertFalse(response["items"][1]["ok"])
        self.assertEqual(response["items"][1]["error"]["code"], "invalid_image")
        self.assertEqual(response["items"][1]["filename"], "broken.png")

    def test_batch_failure_falls_back_to_single_file_inference(self) -> None:
        pool = _BatchPool(fail_batch=True, fail_seed=2)
        response = self._run(
            [_upload("good.png", _png_bytes(1)), _upload("bad.png", _png_bytes(2))],
            pool,
        )

        self.assertEqual(pool.batch_calls, 1)
        self.assertEqual(pool.single_calls, 2)
        self.assertEqual(response["success_count"], 1)
        self.assertEqual(response["failed_count"], 1)
        self.assertTrue(response["items"][0]["ok"])
        self.assertFalse(response["items"][1]["ok"])
        self.assertEqual(response["items"][1]["error"]["code"], "inference_error")


if __name__ == "__main__":
    unittest.main()
