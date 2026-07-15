"""Convert trained sklearn bundles into ONNX for faster inference.

The exporter is **lazy** about its dependencies: importing this module does
NOT require ``skl2onnx`` or ``onnx``. They are imported on first call to
:func:`export_bundle_to_onnx`, and the function raises a clear
:class:`ImportError` if they are missing.

Usage::

    from traffic_sign_system.models.onnx_exporter import export_bundle_to_onnx
    out = export_bundle_to_onnx(Path("models/artifacts/svm_hog+hsv.joblib"))
    # writes <stem>.onnx beside the input file

For the custom :class:`~traffic_sign_system.models.train_ensemble.EnsembleClassifier`,
each base estimator (SVC / KNN / RF) is converted into its own ONNX submodel
(``<stem>__svm.onnx`` etc.) and the ensemble weights are stored in the main
``<stem>.onnx``'s ``metadata_props``. The runtime shim reconstructs the
soft-vote by averaging each sub-model's ``predict_proba``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_bundle_to_onnx(
    bundle_path: Path | str,
    out_path: Path | str | None = None,
    *,
    target_opset: int = 17,
) -> Path:
    """Convert a joblib bundle to ONNX.

    Parameters
    ----------
    bundle_path : Path | str
        Path to the ``.joblib`` model bundle.
    out_path : Path | str | None
        Where to write the ``.onnx`` file. Defaults to ``<bundle_dir>/<stem>.onnx``.
    target_opset : int
        ONNX opset version passed to ``skl2onnx.to_onnx``.

    Returns
    -------
    Path
        The absolute path to the main ``.onnx`` file. For ensembles, sibling
        sub-model files (``<stem>__<name>.onnx``) are also written; their
        locations are referenced via the main file's metadata.

    Raises
    ------
    FileNotFoundError
        If ``bundle_path`` does not exist.
    ValueError
        If the bundle is missing required fields or the classifier is unfit.
    ImportError
        If ``skl2onnx`` or ``onnx`` is not installed.
    """
    bundle_path = Path(bundle_path).resolve()
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Bundle does not exist: {bundle_path}")
    out_path = (
        Path(out_path).resolve()
        if out_path is not None
        else bundle_path.with_suffix(".onnx")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(bundle_path)
    classifier = bundle["classifier"]
    scaler = bundle["scaler"]
    label_map = dict(bundle["label_map"])
    feature_config = dict(bundle["feature_config"])
    summary = dict(bundle["summary"])
    feature_dim = int(bundle["summary"]["feature_dim"])

    scaler_mean, scaler_scale = _extract_scaler_params(scaler)

    _skl2onnx, onnx, FloatTensorType, onnx_to_onnx = _load_sklearn_converter()

    from traffic_sign_system.models.train_ensemble import EnsembleClassifier

    is_ensemble = isinstance(classifier, EnsembleClassifier)

    sub_paths: dict[str, str] = {}  # name -> relative path to sub-onnx
    ensemble_weights: list[float] = []
    ensemble_submodel_types: list[str] = []

    if is_ensemble:
        if classifier.classes_ is None:
            raise ValueError(
                "EnsembleClassifier has no classes_; was it trained?"
            )
        estimators = list(classifier.estimators)
        weights = (
            list(classifier.weights)
            if classifier.weights is not None
            else [1.0] * len(estimators)
        )
        for (name, sub_clf), w in zip(estimators, weights):
            sub_out = out_path.with_name(f".{out_path.stem}__{name}.onnx")
            sub_model = onnx_to_onnx(
                sub_clf,
                initial_types=[("float_input", FloatTensorType([None, feature_dim]))],
                target_opset=target_opset,
            )
            onnx.save(sub_model, str(sub_out))
            sub_paths[name] = sub_out.name
            ensemble_weights.append(float(w))
            ensemble_submodel_types.append(type(sub_clf).__name__)
            logger.info(
                "exported ensemble sub-model %s -> %s (weight=%.3f)",
                name,
                sub_out.name,
                float(w),
            )
        # Main graph: a no-op placeholder so the file still loads as a valid
        # ONNX model. The shim ignores the graph output for ensembles.
        onnx_model = _identity_onnx(feature_dim, target_opset, onnx, FloatTensorType)
    else:
        onnx_model = onnx_to_onnx(
            classifier,
            initial_types=[("float_input", FloatTensorType([None, feature_dim]))],
            target_opset=target_opset,
        )

    classifier_type = (
        "EnsembleClassifier" if is_ensemble else type(classifier).__name__
    )
    _embed_metadata(
        onnx_model,
        label_map=label_map,
        feature_config=feature_config,
        summary=summary,
        classifier_type=classifier_type,
        is_ensemble=is_ensemble,
        ensemble_weights=ensemble_weights,
        ensemble_submodel_types=ensemble_submodel_types,
        ensemble_sub_paths=sub_paths,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )

    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    started = time.perf_counter()
    try:
        onnx.save(onnx_model, str(tmp_path))
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    elapsed = time.perf_counter() - started
    logger.info(
        "Exported %s -> %s in %.2fs (opset=%d, dim=%d, ensemble=%s)",
        bundle_path.name,
        out_path.name,
        elapsed,
        target_opset,
        feature_dim,
        is_ensemble,
    )
    return out_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sklearn_converter():
    """Lazy import of skl2onnx and onnx; raises clear ImportError if missing."""
    try:
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
    except ImportError as exc:
        raise ImportError(
            "skl2onnx and onnx are required for ONNX export. "
            "Install with: pip install skl2onnx onnx"
        ) from exc
    return None, onnx, FloatTensorType, to_onnx


def _identity_onnx(feature_dim: int, target_opset: int, onnx, FloatTensorType):
    """Build a trivial identity ONNX graph used as the main file for ensembles.

    The shim does not actually run this graph for ensemble bundles — it
    routes each input through the per-sub-model ONNX files. The identity
    graph exists only so ``onnx.load`` on the main file succeeds and the
    metadata is preserved.
    """
    from onnx import helper, TensorProto

    input_tensor = helper.make_tensor_value_info(
        "float_input", TensorProto.FLOAT, [None, feature_dim]
    )
    output_tensor = helper.make_tensor_value_info(
        "identity_output", TensorProto.FLOAT, [None, feature_dim]
    )
    node = helper.make_node("Identity", ["float_input"], ["identity_output"])
    graph = helper.make_graph(
        nodes=[node],
        name="ensemble_placeholder",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )
    opset = (helper.make_opsetid("", target_opset),)
    return helper.make_model(
        graph, opset_imports=opset, producer_name="traffic_sign_exporter"
    )


def _extract_scaler_params(scaler: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, scale) arrays from a fitted StandardScaler-like object.

    Raises
    ------
    ValueError
        If the scaler does not expose ``mean_`` / ``scale_`` attributes.
    """
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)
    if mean is None or scale is None:
        raise ValueError(
            "Bundle scaler must be a fitted StandardScaler (with mean_/scale_); "
            f"got {type(scaler).__name__}"
        )
    return np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def _embed_metadata(
    onnx_model: Any,
    *,
    label_map: dict[int, str],
    feature_config: dict[str, Any],
    summary: dict[str, Any],
    classifier_type: str,
    is_ensemble: bool,
    ensemble_weights: list[float],
    ensemble_submodel_types: list[str],
    ensemble_sub_paths: dict[str, str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
) -> None:
    """Embed JSON-encoded metadata into the ONNX model's metadata_props."""
    props = {
        "label_map": json.dumps(
            {str(int(k)): str(v) for k, v in label_map.items()}, ensure_ascii=False
        ),
        "feature_config": json.dumps(feature_config, ensure_ascii=False),
        "summary": json.dumps(summary, ensure_ascii=False),
        "classifier_type": classifier_type,
        "is_ensemble": "1" if is_ensemble else "0",
        "feature_dim": str(int(summary.get("feature_dim", 0))),
        "scaler_mean": json.dumps(
            np.asarray(scaler_mean, dtype=np.float64).tolist(), ensure_ascii=False
        ),
        "scaler_scale": json.dumps(
            np.asarray(scaler_scale, dtype=np.float64).tolist(), ensure_ascii=False
        ),
    }
    if is_ensemble:
        props["ensemble_weights"] = json.dumps(ensemble_weights, ensure_ascii=False)
        props["ensemble_submodel_types"] = json.dumps(
            ensemble_submodel_types, ensure_ascii=False
        )
        props["ensemble_sub_paths"] = json.dumps(ensemble_sub_paths, ensure_ascii=False)

    existing = list(getattr(onnx_model, "metadata_props", None) or [])
    new_keys = set(props)
    kept = [p for p in existing if p.key not in new_keys]
    for key, value in props.items():
        kept.append(onnx.StringStringEntryProto(key=key, value=value))
    onnx_model.metadata_props.clear()
    for p in kept:
        onnx_model.metadata_props.add().CopyFrom(p)


def read_onnx_metadata(onnx_path: Path | str) -> dict[str, Any]:
    """Read metadata embedded by :func:`export_bundle_to_onnx`.

    Exposed for the runtime shim and tests. Decodes JSON values where applicable.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    out: dict[str, Any] = {}
    for prop in getattr(model, "metadata_props", []):
        out[prop.key] = prop.value
    for key in (
        "label_map",
        "feature_config",
        "summary",
        "ensemble_weights",
        "ensemble_submodel_types",
        "ensemble_sub_paths",
        "scaler_mean",
        "scaler_scale",
    ):
        if key in out:
            try:
                out[key] = json.loads(out[key])
            except (TypeError, ValueError):
                pass
    if "is_ensemble" in out:
        out["is_ensemble"] = out["is_ensemble"] == "1"
    if "feature_dim" in out:
        try:
            out["feature_dim"] = int(out["feature_dim"])
        except (TypeError, ValueError):
            pass
    return out