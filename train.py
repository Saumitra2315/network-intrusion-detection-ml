"""Compare models on grouped validation data, then evaluate the selected artifact."""

import argparse
import hashlib
import json
import platform
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold

from nids import (
    BUNDLE_VERSION,
    CATEGORICAL,
    DATA_FILES,
    DATASET_PAGE,
    FEATURES,
    MIRROR_BASE,
    MODEL_NAMES,
    NUMERIC,
    THRESHOLD,
    THRESHOLDS,
    apply_threshold,
    build_pipeline,
    prepare_features,
    read_csv,
    sha256,
)


def load_partition(path: Path) -> tuple[pd.DataFrame, pd.Series, dict]:
    frame = read_csv(path.read_bytes(), max_rows=None, max_bytes=None)
    if "label" not in frame:
        raise ValueError(f"{path.name}: training and test files require a 'label' column.")
    labels = pd.to_numeric(frame["label"], errors="coerce")
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError(f"{path.name}: labels must contain both 0 (normal) and 1 (attack).")
    features = prepare_features(frame)
    digest = sha256(path)
    published = DATA_FILES.get(path.name)
    profile = {
        "filename": path.name,
        "sha256": digest,
        "rows": len(frame),
        "columns": frame.columns.tolist(),
        "labels": {str(int(k)): int(v) for k, v in labels.value_counts().items()},
        "missing_features": {k: int(v) for k, v in features.isna().sum().items()},
        "matches_inspected_mirror": bool(published and published["sha256"] == digest),
        "reference_download_url": MIRROR_BASE + path.name if published else None,
    }
    return features, labels.astype(int), profile


def split_training(features: pd.DataFrame, labels: pd.Series):
    groups = pd.util.hash_pandas_object(features, index=False)
    if groups.nunique() < 5 or labels.value_counts().min() < 5:
        raise ValueError(
            "Grouped validation requires at least five feature groups and five records per class."
        )
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fitting, validation = next(splitter.split(features, labels, groups))
    if any(labels.iloc[index].nunique() != 2 for index in (fitting, validation)):
        raise ValueError("The grouped split must contain both classes in fitting and validation.")
    shared = len(set(groups.iloc[fitting]) & set(groups.iloc[validation]))
    if shared:
        raise ValueError("Grouped split unexpectedly shares feature hashes.")
    details = {
        "method": "First StratifiedGroupKFold split, five folds, shuffle=True",
        "seed": 42,
        "grouping": "pandas feature hash, excluding row index and targets",
        "fitting_rows": len(fitting),
        "validation_rows": len(validation),
        "shared_feature_hashes": shared,
        "fitting_labels": {
            str(int(k)): int(v) for k, v in labels.iloc[fitting].value_counts().items()
        },
        "validation_labels": {
            str(int(k)): int(v) for k, v in labels.iloc[validation].value_counts().items()
        },
        "fitting_indices_sha256": hashlib.sha256(fitting.astype("<i8").tobytes()).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(validation.astype("<i8").tobytes()).hexdigest(),
    }
    return fitting, validation, details


def evaluate(labels: pd.Series, scores: np.ndarray, threshold: float = THRESHOLD) -> dict:
    decisions = apply_threshold(
        pd.DataFrame({"record": np.arange(1, len(scores) + 1), "attack_score": scores}), threshold
    ).prediction.eq("Attack")
    tn, fp, fn, tp = confusion_matrix(labels, decisions, labels=[0, 1]).ravel()
    both_classes = len(set(labels)) == 2
    return {
        "accuracy": float(accuracy_score(labels, decisions)),
        "attack_precision": float(precision_score(labels, decisions, zero_division=0)),
        "attack_recall": float(recall_score(labels, decisions, zero_division=0)),
        "attack_f1": float(f1_score(labels, decisions, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores))
        if both_classes
        else None,
        "threshold": threshold,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "confusion_matrix_order": ["Normal", "Attack"],
    }


def select_model(validation_metrics: dict) -> str:
    # Iteration order gives Random Forest the exact-tie preference.
    return max(MODEL_NAMES, key=lambda name: validation_metrics[name]["average_precision"])


def global_importance(model) -> dict:
    transform = model.named_steps["prepare"]
    encoder = transform.named_transformers_["categorical"].named_steps["encode"]
    names = list(NUMERIC) + [
        name for name, categories in zip(CATEGORICAL, encoder.categories_) for _ in categories
    ]
    importance = pd.Series(model.named_steps["forest"].feature_importances_, index=names)
    grouped = importance.groupby(level=0).sum().reindex(FEATURES)
    return {name: float(value) for name, value in grouped.items()}


def validation_curves(labels, scores) -> dict:
    precision, recall, _ = precision_recall_curve(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    def points(x, y):
        indices = np.unique(np.linspace(0, len(x) - 1, min(200, len(x)), dtype=int))
        return [{"x": float(x[i]), "y": float(y[i])} for i in indices]

    return {
        "precision_recall": points(recall, precision),
        "roc": points(fpr, tpr),
        "note": "Curves show at most 200 sampled points; reported metrics use all validation scores.",
    }


def train(train_path: Path, test_path: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError(
            "Output directory already exists. Choose a new directory to avoid overwriting."
        )
    if train_path.resolve() == test_path.resolve() or sha256(train_path) == sha256(test_path):
        raise ValueError("Training and test files must be separate partitions.")
    x_train, y_train, train_profile = load_partition(train_path)
    fitting, validation, split = split_training(x_train, y_train)
    models, validation_metrics, validation_scores = {}, {}, {}
    for name in MODEL_NAMES:
        print(
            f"Fitting {MODEL_NAMES[name]} on {len(fitting):,} records; validating on {len(validation):,}.",
            flush=True,
        )
        model = build_pipeline(name)
        model.fit(x_train.iloc[fitting], y_train.iloc[fitting])
        scores = model.predict_proba(x_train.iloc[validation])[:, list(model.classes_).index(1)]
        models[name] = model
        validation_scores[name] = scores
        validation_metrics[name] = evaluate(y_train.iloc[validation], scores)
    selected = select_model(validation_metrics)
    model = models[selected]
    print(
        f"Selected {MODEL_NAMES[selected]} using validation average precision. No refit.",
        flush=True,
    )
    # Test records are first loaded after model selection; they cannot influence fitting.
    x_test, y_test, test_profile = load_partition(test_path)
    test_scores = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    baseline = DummyClassifier(strategy="most_frequent").fit(
        x_train.iloc[fitting], y_train.iloc[fitting]
    )
    baseline_scores = baseline.predict_proba(x_test)[:, list(baseline.classes_).index(1)]
    original_hashes = pd.util.hash_pandas_object(x_train, index=False)
    test_hashes = pd.util.hash_pandas_object(x_test, index=False)
    nonoverlap = ~test_hashes.isin(set(original_hashes))

    def test_report(mask):
        count = int(np.count_nonzero(mask))
        return {
            "rows": count,
            "selected_model": evaluate(y_test.loc[mask], test_scores[mask]) if count else None,
            "majority_baseline": evaluate(y_test.loc[mask], baseline_scores[mask])
            if count
            else None,
        }

    metadata = {
        "bundle_version": BUNDLE_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "UNSW-NB15",
        "dataset_page": DATASET_PAGE,
        "provenance_note": (
            "Reference CSVs were downloaded from a third-party mirror because the official download "
            "redirected to Microsoft sign-in. Check matches_inspected_mirror for these actual inputs. "
            "Equality with the inaccessible official files remains unverified."
        ),
        "features": list(FEATURES),
        "numeric_features": list(NUMERIC),
        "categorical_features": list(CATEGORICAL),
        "excluded_columns": ["id", "attack_cat", "label"],
        "threshold": THRESHOLD,
        "selected_model": selected,
        "selection_rule": "Highest validation average precision; exact tie favors Random Forest; no refit",
        "model_parameters": {
            name: candidate.named_steps["forest"].get_params() for name, candidate in models.items()
        },
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "pandas", "scikit-learn", "joblib", "streamlit")
            },
        },
        "training": train_profile,
        "test": test_profile,
        "split": split,
        "validation": {
            "models": validation_metrics,
            "thresholds": [
                evaluate(y_train.iloc[validation], validation_scores[selected], t)
                for t in THRESHOLDS
            ],
            "curves": validation_curves(y_train.iloc[validation], validation_scores[selected]),
        },
        "global_feature_importance": global_importance(model),
        "test_records_with_feature_hash_in_training": int((~nonoverlap).sum()),
        "evaluation": {
            "full_test": test_report(np.ones(len(x_test), dtype=bool)),
            "nonoverlap_test": test_report(nonoverlap.to_numpy()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nids-training-", dir=output.parent) as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        joblib.dump(model, bundle / "model.joblib", compress=3)
        metadata["model_sha256"] = sha256(bundle / "model.joblib")
        (bundle / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        bundle.rename(output)
    print(
        json.dumps(
            {
                "selected_model": selected,
                "split": split,
                "validation_models": validation_metrics,
                "test": metadata["evaluation"],
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True, help="UNSW-NB15 training CSV")
    parser.add_argument("--test", type=Path, required=True, help="UNSW-NB15 testing CSV")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "artifacts")
    args = parser.parse_args()
    try:
        train(args.train, args.test, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
