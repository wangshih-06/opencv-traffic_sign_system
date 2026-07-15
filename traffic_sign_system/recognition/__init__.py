"""Traffic-sign recognition interfaces."""
from __future__ import annotations

from typing import Union

from .onnx_predictor import OnnxPredictor
from .predictor import Predictor
from .scene_aware import SceneAnalyzer
from .sign_detector import SignDetector, draw_detections
from .tracker import SimpleTracker

# Public type alias: anything that quacks like Predictor. Used by SignDetector
# and the inference pool to accept either the joblib-backed Predictor or the
# ONNX-backed OnnxPredictor at runtime.
PredictorLike = Union[Predictor, OnnxPredictor]

__all__ = [
    "Predictor",
    "PredictorLike",
    "OnnxPredictor",
    "SceneAnalyzer",
    "SignDetector",
    "SimpleTracker",
    "draw_detections",
]