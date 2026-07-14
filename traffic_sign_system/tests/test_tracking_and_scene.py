"""Unit and performance checks for tracking and adaptive scene processing."""

from __future__ import annotations

import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from traffic_sign_system.data_processing.preprocessing import Preprocessor
from traffic_sign_system.recognition.scene_aware import SceneAnalyzer
from traffic_sign_system.recognition.sign_detector import SignDetector, draw_detections
from traffic_sign_system.recognition.tracker import SimpleTracker
from traffic_sign_system.recognition.video_recognizer import VideoRecognizer


class _FakePredictor:
    preprocessor = None


class _SequenceDetector:
    def __init__(self):
        self.index = 0

    def detect(self, _frame):
        classes = [1, 1, 2, 1, 2, 1]
        class_id = classes[min(self.index, len(classes) - 1)]
        x = 50 + self.index
        self.index += 1
        return [{
            "bbox": (x, 40, 40, 40),
            "class_id": class_id,
            "class_name": f"class-{class_id}",
            "confidence": 0.90 if class_id == 1 else 0.55,
            "colour": "red",
        }]


class _FixedClassPredictor:
    def __init__(self, class_id: int):
        self.class_id = class_id

    def predict(self, _image):
        return {
            "class_id": self.class_id,
            "class_name": str(self.class_id),
            "confidence": 0.2,
        }


class TrackerTests(unittest.TestCase):
    def test_iou_matching_majority_vote_and_lost_lifecycle(self):
        tracker = SimpleTracker(iou_threshold=0.3, max_lost=2, history_size=5)
        observed_ids = []
        observed_classes = []
        for index, class_id in enumerate([1, 1, 2, 1, 2]):
            result = tracker.update([{
                "bbox": (10 + index, 10, 30, 30),
                "class_id": class_id,
                "class_name": str(class_id),
                "confidence": 0.8,
            }])
            observed_ids.append(result[0]["track_id"])
            observed_classes.append(result[0]["class_id"])
        self.assertEqual(len(set(observed_ids)), 1)
        self.assertEqual(observed_classes[-1], 1)

        lost_one = tracker.update([])
        self.assertEqual(lost_one[0]["lost_count"], 1)
        tracker.update([])
        self.assertEqual(tracker.update([]), [])

    def test_tracker_average_overhead_under_two_ms(self):
        tracker = SimpleTracker()
        samples = []
        for frame_index in range(300):
            detections = [
                {
                    "bbox": (20 + frame_index % 3 + i * 60, 30 + i * 5, 36, 36),
                    "class_id": i,
                    "class_name": str(i),
                    "confidence": 0.8,
                }
                for i in range(5)
            ]
            started = time.perf_counter()
            tracker.update(detections)
            samples.append(time.perf_counter() - started)
        self.assertLess(float(np.mean(samples)) * 1000.0, 2.0)

    def test_video_recognizer_uses_tracker_output(self):
        recognizer = VideoRecognizer(
            _FakePredictor(),
            0,
            detector=_SequenceDetector(),
            tracker=SimpleTracker(history_size=5),
        )
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        results = [recognizer.process_frame(frame) for _ in range(6)]
        self.assertEqual({result["track_id"] for result in results}, {0})
        self.assertEqual(results[-1]["class_id"], 1)
        self.assertIn("raw_detections", results[-1])


class DetectorPostprocessTests(unittest.TestCase):
    def test_low_confidence_shape_match_is_kept_and_mismatch_filtered(self):
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(image, (150, 150), 42, (0, 0, 255), 10)
        circle_results = SignDetector(
            _FixedClassPredictor(0), min_area=100, confidence_threshold=0.5
        ).detect(image)
        triangle_results = SignDetector(
            _FixedClassPredictor(18), min_area=100, confidence_threshold=0.5
        ).detect(image)
        self.assertEqual(len(circle_results), 1)
        self.assertTrue(circle_results[0]["shape_match"])
        self.assertEqual(triangle_results, [])

    def test_lost_track_dashed_drawing_does_not_modify_source(self):
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        detection = {
            "bbox": (20, 20, 50, 50), "class_id": 1, "class_name": "one",
            "confidence": 0.8, "track_id": 3, "lost_count": 1, "colour": "red",
        }
        annotated = draw_detections(image, [detection])
        self.assertEqual(int(image.sum()), 0)
        self.assertGreater(int(annotated.sum()), 0)


class AdaptiveSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parent / "assets" / "low_light_test.png"
        cls.low_light = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if cls.low_light is None:
            raise RuntimeError(f"cannot read {path}")

    def test_low_light_raises_clahe_clip_to_four(self):
        processor = Preprocessor(adaptive=True)
        output = processor(self.low_light)
        self.assertEqual(output.shape, (64, 64))
        self.assertLess(processor.last_analysis["brightness"], 80.0)
        self.assertEqual(processor.last_effective_params["clahe_clip"], 4.0)
        self.assertAlmostEqual(processor._clahe.getClipLimit(), 4.0)

    def test_blur_triggers_sharpen_and_disables_gaussian(self):
        blurred = cv2.GaussianBlur(self.low_light, (61, 61), 0)
        processor = Preprocessor(adaptive=True)
        processor(blurred)
        self.assertLess(processor.last_analysis["blur_score"], 50.0)
        self.assertTrue(processor.last_effective_params["sharpen"])
        self.assertEqual(processor.last_effective_params["gaussian_ksize"], 0)

    def test_scene_analyzer_detects_low_light_with_realtime_cost(self):
        analyzer = SceneAnalyzer()
        analyzer.analyze(self.low_light)  # warm up OpenCV paths
        samples = []
        analysis = None
        for _ in range(50):
            started = time.perf_counter()
            analysis = analyzer.analyze(self.low_light)
            samples.append(time.perf_counter() - started)
        self.assertIn("low_light", analysis["degradations"])
        self.assertLess(float(np.mean(samples)) * 1000.0, 5.0)


if __name__ == "__main__":
    unittest.main()

