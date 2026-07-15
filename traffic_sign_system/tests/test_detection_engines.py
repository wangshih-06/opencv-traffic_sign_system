from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from traffic_sign_system.recognition.detection_engines import (
    DeepOnnxDetector,
    merge_detections,
    run_detection_engine,
)


class _FakeNet:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.input = None

    def setInput(self, blob: np.ndarray) -> None:
        self.input = blob

    def forward(self) -> np.ndarray:
        return self.output


class TestDetectionEngines(unittest.TestCase):
    def test_merge_overlapping_boxes_keeps_provenance(self) -> None:
        traditional = [{
            "bbox": (10, 10, 40, 40),
            "class_id": 1,
            "class_name": "traditional",
            "confidence": 0.72,
            "colour": "red",
        }]
        deep = [{
            "bbox": (12, 12, 40, 40),
            "class_id": 1,
            "class_name": "deep",
            "confidence": 0.91,
            "colour": "red",
        }]

        merged = merge_detections(traditional, deep)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["sources"], ["traditional", "deep"])
        self.assertEqual(merged[0]["engine"], "hybrid")
        self.assertEqual(merged[0]["detector_confidence"], 0.91)

    def test_merge_non_overlapping_boxes_keeps_both(self) -> None:
        box = lambda x: {
            "bbox": (x, 0, 20, 20),
            "class_id": 1,
            "class_name": "sign",
            "confidence": 0.8,
            "colour": "blue",
        }

        merged = merge_detections([box(0)], [box(80)])

        self.assertEqual(len(merged), 2)
        self.assertCountEqual([item["sources"] for item in merged], [["traditional"], ["deep"]])

    def test_hybrid_without_deep_model_falls_back(self) -> None:
        traditional = [{"bbox": (0, 0, 10, 10), "class_id": 1, "confidence": 0.6}]
        with patch("traffic_sign_system.recognition.detection_engines.SignDetector") as detector_cls:
            detector_cls.return_value.detect.return_value = traditional
            result = run_detection_engine(object(), np.zeros((32, 32, 3), np.uint8), "hybrid")

        self.assertTrue(result["fallback"])
        self.assertEqual(result["engine_used"], "traditional")
        self.assertEqual(result["detections"], traditional)

    def test_nms_ready_output_restores_letterbox_coordinates(self) -> None:
        # Input is 100x100; original image is 200x100, so y receives 25 px padding.
        net = _FakeNet(np.array([[25, 35, 75, 65, 0.9, 2]], dtype=np.float32))
        detector = DeepOnnxDetector(
            "virtual.onnx",
            net=net,
            metadata={"input_size": 100, "num_classes": 3, "confidence_threshold": 0.1},
        )

        result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

        self.assertEqual(result[0]["bbox"], (50, 20, 100, 60))
        self.assertEqual(result[0]["class_id"], 2)
        self.assertEqual(net.input.shape, (1, 3, 100, 100))

    def test_raw_yolo_output_and_transposed_output(self) -> None:
        # cx, cy, w, h, objectness, class-0, class-1 in normalized input coords.
        output = np.array([[50, 50], [50, 50], [20, 20], [20, 20], [0.9, 0.8], [0.9, 0.1], [0.1, 0.95]], dtype=np.float32)
        detector = DeepOnnxDetector(
            "virtual.onnx",
            net=_FakeNet(output),
            metadata={"input_size": 100, "num_classes": 2, "confidence_threshold": 0.2, "output_format": "raw_objectness"},
        )

        result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["class_id"], 0)
        self.assertAlmostEqual(result[0]["confidence"], 0.81, places=5)

    def test_class_aware_nms_does_not_suppress_different_classes(self) -> None:
        output = np.array([
            [20, 20, 80, 80, 0.9, 0],
            [20, 20, 80, 80, 0.8, 1],
        ], dtype=np.float32)
        detector = DeepOnnxDetector(
            "virtual.onnx",
            net=_FakeNet(output),
            metadata={"input_size": 100, "num_classes": 2, "confidence_threshold": 0.1},
        )

        result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual({item["class_id"] for item in result}, {0, 1})

    def test_sidecar_labels_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "detector.onnx"
            model.write_bytes(b"placeholder")
            model.with_suffix(".json").write_text(
                json.dumps({"input_size": [64, 48], "num_classes": 2, "labels": ["stop", "yield"]}),
                encoding="utf-8",
            )
            detector = DeepOnnxDetector(model, net=_FakeNet(np.empty((0, 6), dtype=np.float32)))

        self.assertEqual(detector.input_size, (64, 48))
        self.assertEqual(detector.label_map[1], "yield")


if __name__ == "__main__":
    unittest.main()
