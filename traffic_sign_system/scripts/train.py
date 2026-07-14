"""????????????????????????

???
    python -m traffic_sign_system.scripts.train --model svm --mode hog
    python -m traffic_sign_system.scripts.train --model svm --mode "hog+hsv" --grid-search

???????????????????????????????
???????????????????????????????
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from traffic_sign_system.config.labels import load_labels
from traffic_sign_system.config.settings import (
    DATASET_DIR,
    IMG_SIZE,
    MODEL_ARTIFACTS_DIR,
    RANDOM_STATE,
    SVM_C,
    SVM_GAMMA,
    SVM_KERNEL,
    TEST_SIZE,
    VAL_SIZE,
)
from traffic_sign_system.data_processing.data_loader import load_train_data
from traffic_sign_system.features.feature_fusion import FeatureBuilder
from traffic_sign_system.models.model_manager import TrainSummary, save_bundle
from traffic_sign_system.models.train_ensemble import (
    EnsembleClassifier,
    build_default_estimators,
)
from traffic_sign_system.models.train_svm import train_svm

logger = logging.getLogger(__name__)


def _parse_gamma(value: str) -> str | float:
    """?? sklearn ? ``scale``/``auto`` ???? gamma?"""
    if value in {"scale", "auto"}:
        return value
    try:
        gamma = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "gamma ??? 'scale'?'auto' ?????"
        ) from exc
    if gamma <= 0:
        raise argparse.ArgumentTypeError("gamma ???? 0")
    return gamma


def _resolve_train_dir(value: Path | None) -> Path:
    """????????? GTSRB ???????"""
    if value is not None:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"?????????: {path}")
        return path

    candidates = (DATASET_DIR / "Train", DATASET_DIR / "train" / "Train")
    for candidate in candidates:
        # Windows ????????????dataset/Train ??????
        # dataset/train????????????????????????
        if candidate.exists() and any(
            child.is_dir() and child.name.isdigit() for child in candidate.iterdir()
        ):
            return candidate
    searched = "?".join(str(item) for item in candidates)
    raise FileNotFoundError(f"??? GTSRB ?????????{searched}")


def _limit_balanced_samples(
    images: list[np.ndarray],
    labels: np.ndarray,
    max_samples: int | None,
    random_state: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """????????????????????????"""
    if max_samples is None or max_samples >= len(images):
        return images, labels
    if max_samples < len(np.unique(labels)) * 3:
        raise ValueError("--max-samples ?????????? 3 ???")

    rng = np.random.default_rng(random_state)
    classes = np.unique(labels)
    per_class = max_samples // len(classes)
    if per_class < 3:
        raise ValueError("--max-samples ????????? train/val/test ??")

    selected: list[int] = []
    for class_id in classes:
        class_indices = np.flatnonzero(labels == class_id)
        take = min(per_class, len(class_indices))
        selected.extend(rng.choice(class_indices, size=take, replace=False).tolist())

    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    logger.warning(
        "??? --max-samples?? %d ??????? %d ?????? %d ???",
        len(images),
        len(selected_array),
        per_class,
    )
    return [images[index] for index in selected_array], labels[selected_array]


def _split_indices(
    labels: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """???????? train / val / test ????"""
    if not 0 < test_size < 1:
        raise ValueError("--test-size ??? (0, 1) ???")
    if not 0 < val_size < 1:
        raise ValueError("--val-size ??? (0, 1) ?????????????")

    indices = np.arange(len(labels))
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=labels[train_val_idx],
        random_state=random_state,
    )
    return train_idx, val_idx, test_idx


def _extract_splits(
    builder: FeatureBuilder,
    images: list[np.ndarray],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """??????????????????????????"""
    started = time.perf_counter()
    outputs: list[np.ndarray] = []
    for split_name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        logger.info("?? %s ???N=%d, mode=%s", split_name, len(indices), builder.mode)
        split_images = [images[index] for index in indices]
        features = builder.extract_batch(split_images).astype(np.float32, copy=False)
        outputs.append(features)
        logger.info("%s ?? shape=%s", split_name, features.shape)
    return outputs[0], outputs[1], outputs[2], time.perf_counter() - started


def _grid_search_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, str | float, float, dict[str, Any]]:
    """??????? SVM ??? CV????????????"""
    class_counts = np.unique(y_train, return_counts=True)[1]
    if class_counts.min() < args.cv_folds:
        raise ValueError(
            f"?????????? {args.cv_folds} ???? {args.cv_folds} ? CV?"
            f"????? {class_counts.min()}"
        )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel=args.kernel,
                    probability=True,
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.random_state,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid={"svc__C": args.C_grid, "svc__gamma": args.gamma_grid},
        scoring="accuracy",
        cv=cv,
        n_jobs=args.n_jobs,
        refit=False,
        verbose=1,
    )
    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    best_params = search.best_params_
    selected_c = float(best_params["svc__C"])
    selected_gamma = best_params["svc__gamma"]
    details = {
        "cv_folds": args.cv_folds,
        "cv_candidates": len(search.cv_results_["params"]),
        "cv_best_accuracy": float(search.best_score_),
        "cv_seconds": elapsed,
    }
    logger.info(
        "???? CV ???best C=%s, gamma=%s, accuracy=%.4f",
        selected_c,
        selected_gamma,
        search.best_score_,
    )
    return selected_c, selected_gamma, elapsed, details


def _train_classifier(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    cv_features: np.ndarray | None = None,
) -> tuple[Any, float, dict[str, Any]]:
    """??? ``--model`` ?????????????????????"""
    if model_name == "svm":
        selected_c: float = args.C
        selected_gamma: str | float = args.gamma
        cv_seconds = 0.0
        extras: dict[str, Any] = {
            "kernel": args.kernel,
            "grid_search": args.grid_search,
            "class_weight": args.class_weight,
        }
        if args.grid_search:
            if cv_features is None:
                raise ValueError("SVM ??????????????")
            selected_c, selected_gamma, cv_seconds, cv_details = _grid_search_svm(
                cv_features, y_train, args
            )
            extras.update(cv_details)

        classifier, fit_seconds = train_svm(
            X_train,
            y_train,
            C=selected_c,
            kernel=args.kernel,
            gamma=selected_gamma,
            class_weight=args.class_weight,
            random_state=args.random_state,
        )
        extras.update({"C": selected_c, "gamma": selected_gamma, "fit_seconds": fit_seconds})
        return classifier, cv_seconds + fit_seconds, extras

    if args.grid_search:
        raise ValueError("--grid-search ????? --model svm")

    if model_name == "knn":
        classifier = KNeighborsClassifier(n_neighbors=args.n_neighbors, n_jobs=args.n_jobs)
        params = {"n_neighbors": args.n_neighbors}
    elif model_name == "rf":
        classifier = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        params = {"n_estimators": args.n_estimators}
    else:  # argparse ??? choices??????????????
        raise ValueError(f"??????: {model_name}")

    started = time.perf_counter()
    classifier.fit(X_train, y_train)
    return classifier, time.perf_counter() - started, params


def _train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[EnsembleClassifier, float, dict[str, Any]]:
    """训练 SVM+KNN+RF 集成，按 val accuracy 计算权重。"""
    estimators = build_default_estimators(
        svm_C=args.C,
        svm_kernel=args.kernel,
        svm_gamma=args.gamma,
        svm_class_weight=args.class_weight,
        knn_neighbors=args.n_neighbors,
        rf_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    ensemble = EnsembleClassifier(estimators=estimators)
    started = time.perf_counter()
    ensemble.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    # 收集每个子分类器的 val accuracy
    scores: list[float] = []
    for name, clf in ensemble.estimators:
        if not hasattr(clf, "predict"):
            scores.append(0.0)
            continue
        acc = float(accuracy_score(y_val, clf.predict(X_val)))
        scores.append(acc)
        logger.info("[ensemble] %s val_acc=%.4f", name, acc)
    ensemble.set_weights_from_scores(scores)

    extras: dict[str, Any] = {
        "members": [name for name, _ in estimators],
        "val_member_accuracy": dict(zip([n for n, _ in estimators], scores)),
        "ensemble_weights": (
            None if ensemble.weights is None else ensemble.weights.tolist()
        ),
        "class_weight": args.class_weight,
        "fit_seconds": fit_seconds,
    }
    return ensemble, fit_seconds, extras


def train(args: argparse.Namespace) -> Path:
    """??????????????????????"""
    train_dir = _resolve_train_dir(args.train_dir)
    logger.info("???????%s", train_dir)
    images, labels, _, bad_log = load_train_data(train_dir)
    if not images:
        raise RuntimeError(f"?? {train_dir} ???????????")

    y_all = np.asarray(labels, dtype=np.int32)
    images, y_all = _limit_balanced_samples(
        images, y_all, args.max_samples, args.random_state
    )
    if len(np.unique(y_all)) < 2:
        raise ValueError("????????????")

    train_idx, val_idx, test_idx = _split_indices(
        y_all, args.test_size, args.val_size, args.random_state
    )
    logger.info(
        "???????train=%d, val=%d, test=%d????????",
        len(train_idx), len(val_idx), len(test_idx),
    )

    builder = FeatureBuilder(
        mode=args.mode,
        img_size=args.img_size,
        h_bins=args.h_bins,
        s_bins=args.s_bins,
    )
    # ??????????????? bundle ????????????
    feature_config = dict(builder.config)
    X_train_raw, X_val_raw, X_test_raw, feature_seconds = _extract_splits(
        builder, images, train_idx, val_idx, test_idx
    )
    y_train, y_val, y_test = y_all[train_idx], y_all[val_idx], y_all[test_idx]

    # StandardScaler ????? fit?????????? transform?
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    if args.ensemble:
        classifier, train_seconds, model_extras = _train_ensemble(
            X_train, y_train, X_val, y_val, args
        )
    else:
        classifier, train_seconds, model_extras = _train_classifier(
            args.model, X_train, y_train, args, cv_features=X_train_raw
        )
    val_accuracy = float(accuracy_score(y_val, classifier.predict(X_val)))

    # ????????????????????????
    test_accuracy = float(accuracy_score(y_test, classifier.predict(X_test)))

    labels_path = DATASET_DIR / "labels.csv"
    class_names = load_labels(labels_path if labels_path.exists() else None)
    label_map = {
        int(class_id): class_names.get(int(class_id), str(int(class_id)))
        for class_id in sorted(np.unique(y_all))
    }

    output = args.output or (
        MODEL_ARTIFACTS_DIR / f"{'ensemble' if args.ensemble else args.model}_{args.mode}.joblib"
    )
    summary = TrainSummary(
        model="ensemble" if args.ensemble else args.model,
        feature_mode=args.mode,
        n_train=len(y_train),
        n_val=len(y_val),
        n_test=len(y_test),
        feature_dim=int(X_train.shape[1]),
        train_seconds=float(train_seconds),
        extras={
            "val_accuracy": val_accuracy,
            "test_accuracy": test_accuracy,
            "feature_seconds": feature_seconds,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "val_size": args.val_size,
            "bad_image_count": len(bad_log),
            **model_extras,
        },
    )
    out_path = save_bundle(
        output,
        classifier=classifier,
        scaler=scaler,
        label_map=label_map,
        feature_config=feature_config,
        summary=summary,
    )

    print(f"train_seconds: {train_seconds:.3f}")
    print(f"val_acc: {val_accuracy:.4f}")
    print(f"test_acc: {test_accuracy:.4f}")
    print(f"saved_bundle: {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="???????????? classifier + scaler + metadata ????"
    )
    parser.add_argument("--model", choices=("svm", "knn", "rf"), default="svm")
    parser.add_argument("--mode", choices=FeatureBuilder.VALID_MODES, default="hog")
    parser.add_argument("--C", type=float, default=SVM_C, help="SVM ? C????10?")
    parser.add_argument(
        "--gamma", type=_parse_gamma, default=SVM_GAMMA,
        help="SVM gamma?scale?auto ?????????scale?",
    )
    parser.add_argument(
        "--kernel", choices=("rbf", "linear", "poly", "sigmoid"), default=SVM_KERNEL
    )
    parser.add_argument("--n-neighbors", type=int, default=5, help="KNN ???")
    parser.add_argument("--n-estimators", type=int, default=200, help="RF ???")
    parser.add_argument("--n-jobs", type=int, default=1, help="Grid/RF/KNN ?????????1??????????")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--h-bins", type=int, default=8)
    parser.add_argument("--s-bins", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--val-size", type=float, default=VAL_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--train-dir", type=Path, default=None, help="????????????")
    parser.add_argument("--output", type=Path, default=None, help="?? .joblib ??")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="??????????????????????????",
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="???????? SVM C/gamma ???????",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--C-grid", type=float, nargs="+", default=(0.1, 1.0, 10.0))
    parser.add_argument(
        "--gamma-grid", type=_parse_gamma, nargs="+", default=("scale", 0.001, 0.01)
    )
    parser.add_argument(
        "--class-weight",
        choices=("balanced", "none"),
        default="balanced",
        help="SVM 类别权重策略；balanced 会按 1/freq 自动平衡 43 类样本",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="启用 SVM+KNN+RF 软投票集成（覆盖 --model）",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args(argv)
    if args.n_neighbors < 1:
        raise ValueError("--n-neighbors ???? 0")
    if args.n_estimators < 1:
        raise ValueError("--n-estimators ???? 0")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds ????? 2")
    if args.C <= 0:
        raise ValueError("--C ???? 0")
    if any(value <= 0 for value in args.C_grid):
        raise ValueError("--C-grid ????????? 0")
    # 将 CLI 字符串 'none' 翻译为 None，便于 sklearn 直接使用
    args.class_weight = None if args.class_weight == "none" else args.class_weight
    train(args)


if __name__ == "__main__":
    main()
