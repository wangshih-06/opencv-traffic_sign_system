"""Save, load, and validate serialized model bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import joblib


REQUIRED_KEYS = ("classifier", "scaler", "label_map", "feature_config", "summary")
SUMMARY_KEYS = (
    "model",
    "feature_mode",
    "n_train",
    "n_val",
    "n_test",
    "feature_dim",
    "train_seconds",
    "extras",
)


@dataclass
class TrainSummary:
    """Serializable metadata describing a completed training run."""

    model: str
    feature_mode: str
    n_train: int
    n_val: int
    n_test: int
    feature_dim: int
    train_seconds: float
    extras: dict[str, Any] = field(default_factory=dict)


def _ensure_json_serializable(value: Any, field_name: str) -> None:
    """Raise an early, clear error for invalid metadata values."""
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bundle field {field_name!r} must be JSON serializable") from exc


def validate_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate bundle structure and the minimum interfaces used at inference."""
    if not isinstance(bundle, Mapping):
        raise ValueError("A model bundle must be a dict or another Mapping")

    missing = [key for key in REQUIRED_KEYS if key not in bundle]
    if missing:
        raise ValueError(f"Model bundle is missing required fields: {missing}")

    classifier = bundle["classifier"]
    scaler = bundle["scaler"]
    if not hasattr(classifier, "predict"):
        raise ValueError("Bundle classifier must provide a predict method")
    if not hasattr(scaler, "transform"):
        raise ValueError("Bundle scaler must provide a transform method")

    label_map = bundle["label_map"]
    feature_config = bundle["feature_config"]
    summary = bundle["summary"]
    if not isinstance(label_map, Mapping):
        raise ValueError("Bundle label_map must be a mapping")
    if not isinstance(feature_config, Mapping):
        raise ValueError("Bundle feature_config must be a mapping")
    if "mode" not in feature_config:
        raise ValueError("Bundle feature_config is missing 'mode'")
    if not isinstance(summary, Mapping):
        raise ValueError("Bundle summary must be a mapping")

    missing_summary = [key for key in SUMMARY_KEYS if key not in summary]
    if missing_summary:
        raise ValueError(f"Bundle summary is missing fields: {missing_summary}")

    if summary["feature_mode"] != feature_config["mode"]:
        raise ValueError(
            "Bundle feature_config and summary feature modes differ: "
            f"{feature_config['mode']!r} != {summary['feature_mode']!r}"
        )


def save_bundle(
    out_path: Path | str,
    classifier: Any,
    scaler: Any,
    label_map: Mapping[int, str],
    feature_config: Mapping[str, Any],
    summary: TrainSummary,
) -> Path:
    """Persist classifier, scaler, labels, features, and summary in one joblib file.

    The file is first written beside the destination and then atomically replaced,
    avoiding partial bundles if a run is interrupted while saving.
    """
    if not isinstance(summary, TrainSummary):
        raise TypeError("summary must be a TrainSummary instance")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_map_dict = dict(label_map)
    feature_config_dict = dict(feature_config)
    summary_dict = asdict(summary)
    _ensure_json_serializable(label_map_dict, "label_map")
    _ensure_json_serializable(feature_config_dict, "feature_config")
    _ensure_json_serializable(summary_dict, "summary")

    bundle: dict[str, Any] = {
        "classifier": classifier,
        "scaler": scaler,
        "label_map": label_map_dict,
        "feature_config": feature_config_dict,
        "summary": summary_dict,
    }
    validate_bundle(bundle)

    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    try:
        joblib.dump(bundle, tmp_path)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return out_path


def load_bundle(path: Path | str) -> dict[str, Any]:
    """Load and validate a joblib bundle from a trusted source.

    Joblib/pickle files can execute code while deserializing, so never load an
    artifact obtained from an untrusted source.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model bundle does not exist: {path}")
    bundle = joblib.load(path)
    validate_bundle(bundle)
    return dict(bundle)
