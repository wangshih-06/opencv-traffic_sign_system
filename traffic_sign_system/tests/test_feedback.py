"""Tests for the human correction feedback store and validation."""

from __future__ import annotations

import csv
import importlib
import io
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from traffic_sign_system.api.feedback_store import FeedbackStore

app_module = importlib.import_module("traffic_sign_system.api.app")


class FeedbackStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = FeedbackStore(Path(self.temp_dir.name) / "feedback.db")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _payload(**overrides):
        payload = {
            "history_id": "history-1",
            "source": "image",
            "filename": "stop.png",
            "model": "svm.joblib",
            "predicted_class_id": 13,
            "predicted_class_name": "让行",
            "predicted_confidence": 0.61,
            "corrected_class_id": 14,
            "corrected_class_name": "停车让行",
            "verdict": "incorrect",
            "note": "形状相似",
            "bbox": [10, 20, 30, 40],
        }
        payload.update(overrides)
        return payload

    def test_create_list_status_and_delete_round_trip(self) -> None:
        created = self.store.create(self._payload())
        self.assertEqual(created["status"], "new")
        self.assertEqual(created["bbox"], [10, 20, 30, 40])

        items, total = self.store.list()
        self.assertEqual(total, 1)
        self.assertEqual(items[0]["corrected_class_id"], 14)
        self.assertEqual(self.store.stats()["incorrect"], 1)

        reviewed = self.store.update_status(created["id"], "reviewed")
        self.assertIsNotNone(reviewed)
        self.assertEqual(reviewed["status"], "reviewed")
        reviewed_items, reviewed_total = self.store.list(status="reviewed")
        self.assertEqual(reviewed_total, 1)
        self.assertEqual(reviewed_items[0]["id"], created["id"])

        self.assertTrue(self.store.delete(created["id"]))
        self.assertFalse(self.store.delete(created["id"]))
        self.assertEqual(self.store.stats()["total"], 0)

    def test_csv_export_contains_correction_fields(self) -> None:
        self.store.create(self._payload(filename="=cmd|calc", note="+unsafe"))
        rows = list(csv.DictReader(io.StringIO(self.store.export_csv())))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["predicted_class_id"], "13")
        self.assertEqual(rows[0]["corrected_class_id"], "14")
        self.assertEqual(rows[0]["verdict"], "incorrect")
        self.assertEqual(rows[0]["filename"], "'=cmd|calc")
        self.assertEqual(rows[0]["note"], "'+unsafe")


class FeedbackValidationTests(unittest.TestCase):
    def test_correct_verdict_uses_predicted_label(self) -> None:
        request = app_module.FeedbackCreateRequest(
            source="image",
            filename="sample.png",
            predicted_class_id=14,
            predicted_class_name="停车让行",
            predicted_confidence=0.91,
            verdict="correct",
        )
        payload = app_module._normalise_feedback(request)
        self.assertEqual(payload["corrected_class_id"], 14)
        self.assertEqual(payload["corrected_class_name"], "停车让行")

    def test_incorrect_verdict_requires_corrected_class(self) -> None:
        request = app_module.FeedbackCreateRequest(
            source="image",
            filename="sample.png",
            predicted_class_id=14,
            predicted_class_name="停车让行",
            verdict="incorrect",
        )
        with self.assertRaises(HTTPException) as context:
            app_module._normalise_feedback(request)
        self.assertEqual(context.exception.status_code, 400)

    def test_bbox_must_contain_four_values(self) -> None:
        request = app_module.FeedbackCreateRequest(
            source="image",
            filename="sample.png",
            predicted_class_id=14,
            predicted_class_name="停车让行",
            verdict="correct",
            bbox=[1, 2, 3],
        )
        with self.assertRaises(HTTPException):
            app_module._normalise_feedback(request)


if __name__ == "__main__":
    unittest.main()
