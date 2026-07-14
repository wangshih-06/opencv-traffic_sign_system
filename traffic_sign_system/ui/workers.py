"""Background Qt workers for model loading and image prediction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from traffic_sign_system.recognition.predictor import Predictor


def compute_topk(predictor: Predictor, img_bgr: np.ndarray, k: int = 3) -> list[dict]:
    """Return the top-*k* predictions for *img_bgr*, sorted by confidence.

    Mirrors the feature-extraction steps in ``Predictor.predict`` using its
    public ``builder``/``scaler``/``classifier`` attributes. Returns an empty
    list when the classifier has no ``predict_proba`` (e.g. a plain LinearSVC
    without probability estimates).
    """
    classifier = predictor.classifier
    if not hasattr(classifier, "predict_proba"):
        return []

    image = img_bgr if img_bgr.dtype == np.uint8 else np.clip(img_bgr, 0, 255).astype(np.uint8)
    features = np.asarray(predictor.builder.extract_one(image), dtype=np.float32)
    X_scaled = np.asarray(predictor.scaler.transform(features[None, :]))
    probabilities = np.asarray(classifier.predict_proba(X_scaled))[0]
    classes = np.asarray(getattr(classifier, "classes_", []))
    if classes.shape[0] != probabilities.shape[0]:
        return []

    top_n = max(0, min(int(k), probabilities.shape[0]))
    order = np.argsort(probabilities)[::-1][:top_n]
    results: list[dict] = []
    for idx in order:
        class_id = int(classes[idx])
        results.append(
            {
                "class_id": class_id,
                "class_name": predictor.label_map.get(class_id, str(class_id)),
                "confidence": float(probabilities[idx]),
            }
        )
    return results


class LoadModelWorker(QThread):
    """Load and validate a model bundle without blocking the GUI thread."""

    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, bundle_path: Path | str, parent=None):
        super().__init__(parent)
        self.bundle_path = Path(bundle_path)

    def run(self) -> None:
        try:
            # Predictor reconstructs FeatureBuilder and calls load_bundle internally,
            # so all model deserialization and validation stays in this thread.
            predictor = Predictor(self.bundle_path)
            self.loaded.emit(predictor)
        except Exception as exc:  # Qt worker boundary: report errors to the UI.
            self.error.emit(f"{type(exc).__name__}: {exc}")


class PredictWorker(QThread):
    """Run one prediction in the background and emit its result dictionary."""

    predicted = pyqtSignal(dict)
    top_k = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, predictor: Predictor, img_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        # Own a stable snapshot even if the main window loads another image.
        self.img_bgr = np.ascontiguousarray(img_bgr).copy()

    def run(self) -> None:
        try:
            result = self.predictor.predict(self.img_bgr)
            self.predicted.emit(dict(result))
            # The result panel consumes the first three entries; the dashboard
            # uses all ten to render its confidence distribution.
            self.top_k.emit(compute_topk(self.predictor, self.img_bgr, k=10))
        except Exception as exc:  # Qt worker boundary: report errors to the UI.
            self.error.emit(f"{type(exc).__name__}: {exc}")


class DetectWorker(QThread):
    """Run SignDetector.detect on one frame in the background."""

    detected = pyqtSignal(list)   # list[dict] — detection results
    annotated = pyqtSignal(object)  # np.ndarray — annotated image copy
    error = pyqtSignal(str)

    def __init__(
        self,
        detector,  # SignDetector  (lazy import to avoid circular dep)
        img_bgr: np.ndarray,
        parent=None,
    ):
        super().__init__(parent)
        self.detector = detector
        self.img_bgr = np.ascontiguousarray(img_bgr).copy()

    def run(self) -> None:
        try:
            from traffic_sign_system.recognition.sign_detector import draw_detections

            results = self.detector.detect(self.img_bgr)
            annotated_img = draw_detections(self.img_bgr, results)
            self.detected.emit(results)
            self.annotated.emit(annotated_img)
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")


class BatchPredictWorker(QThread):
    """Predict every image file under a folder in the background."""

    progress = pyqtSignal(int, int)  # (completed, total)
    finished_batch = pyqtSignal(list)  # list[dict] — avoids clashing with QThread.finished
    error = pyqtSignal(str)

    _EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp")

    def __init__(self, predictor: Predictor, folder: Path | str, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.folder = Path(folder)
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            files = sorted(
                p
                for p in self.folder.rglob("*")
                if p.is_file() and p.suffix.lower() in self._EXTENSIONS
            )
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
            return

        total = len(files)
        results: list[dict] = []
        for index, file_path in enumerate(files, start=1):
            if self._stop:
                break
            entry = {
                "path": str(file_path),
                "class_id": None,
                "class_name": None,
                "confidence": None,
                "error": None,
            }
            try:
                image = cv2.imdecode(
                    np.fromfile(str(file_path), dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None or image.size == 0:
                    raise ValueError("无法解码该图片。")
                prediction = self.predictor.predict(image)
                entry["class_id"] = int(prediction["class_id"])
                entry["class_name"] = str(prediction["class_name"])
                entry["confidence"] = prediction.get("confidence")
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
            self.progress.emit(index, total)

        self.finished_batch.emit(results)
