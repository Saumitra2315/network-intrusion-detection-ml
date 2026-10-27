"""Shared UNSW-NB15 schema, strict CSV validation, and prediction pipeline."""

import csv
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MAX_BYTES = 10_000_000
MAX_ROWS = 10_000
THRESHOLD = 0.5
BUNDLE_VERSION = 2
THRESHOLDS = tuple(i / 20 for i in range(1, 20))
MODEL_NAMES = {"random_forest": "Random Forest", "extra_trees": "Extra Trees"}
CATEGORICAL = ("proto", "service", "state")
FEATURES = tuple(
    "dur proto service state spkts dpkts sbytes dbytes rate sttl dttl sload dload "
    "sloss dloss sinpkt dinpkt sjit djit swin stcpb dtcpb dwin tcprtt synack ackdat "
    "smean dmean trans_depth response_body_len ct_srv_src ct_state_ttl ct_dst_ltm "
    "ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm is_ftp_login ct_ftp_cmd "
    "ct_flw_http_mthd ct_src_ltm ct_srv_dst is_sm_ips_ports".split()
)
NUMERIC = tuple(name for name in FEATURES if name not in CATEGORICAL)
DATASET_PAGE = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
MIRROR_BASE = (
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/"
)
DATA_FILES = {
    "UNSW_NB15_training-set.csv": {
        "rows": 175341,
        "sha256": "bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa",
    },
    "UNSW_NB15_testing-set.csv": {
        "rows": 82332,
        "sha256": "734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559",
    },
}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_csv(
    data: bytes, *, max_rows: int | None = MAX_ROWS, max_bytes: int | None = MAX_BYTES
) -> pd.DataFrame:
    """Check raw headers and row widths before pandas can normalize them."""
    if max_bytes is not None and len(data) > max_bytes:
        raise ValueError("CSV exceeds the 10 MB upload limit.")
    try:
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("Use a UTF-8 encoded CSV file.") from None
    if "\x00" in content:
        raise ValueError("Malformed CSV: null bytes are not allowed.")
    try:
        reader = csv.reader(io.StringIO(content, newline=""), strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV is empty. Include a header and at least one record.")
        if any(not name.strip() for name in header):
            raise ValueError("Every CSV column needs a non-empty header.")
        if len(set(header)) != len(header):
            raise ValueError("Duplicate column headers. Each column name must be unique.")
        rows = 0
        for row in reader:
            if not row:
                continue
            rows += 1
            if max_rows is not None and rows > max_rows:
                raise ValueError("CSV exceeds the 10,000 record upload limit.")
            if len(row) != len(header):
                raise ValueError(
                    f"Malformed CSV: record {rows} has {len(row)} fields; "
                    f"expected {len(header)}. Check commas and quoting."
                )
        if rows == 0:
            raise ValueError("CSV has no records. Add at least one row below the header.")
        return pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False, na_values=[""])
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError):
        raise ValueError("Malformed CSV. Check field separators and quoting.") from None


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column headers. Each column name must be unique.")
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("CSV has no records.")
    # Only the allowlisted inputs reach the model; labels and identifiers never do.
    result = frame.loc[:, list(FEATURES)].copy()
    for name in NUMERIC:
        values = result[name].replace(r"^\s*$", np.nan, regex=True)
        try:
            converted = pd.to_numeric(values, errors="raise").astype(float)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"Column '{name}' must contain numbers or empty cells.") from None
        present = converted.dropna()
        if not np.isfinite(present).all() or (present.abs() > np.finfo(np.float32).max).any():
            raise ValueError(f"Column '{name}' contains an infinite or oversized number.")
        result[name] = converted
    for name in CATEGORICAL:
        values = result[name].astype("string").str.strip()
        result[name] = values.astype(object).where(values.notna() & values.ne(""), np.nan)
    return result


def build_pipeline(model_name: str = "random_forest") -> Pipeline:
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown model. Choose random_forest or extra_trees.")
    classifier = RandomForestClassifier if model_name == "random_forest" else ExtraTreesClassifier
    transform = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), list(NUMERIC)),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="constant", fill_value="missing", keep_empty_features=True
                            ),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                list(CATEGORICAL),
            ),
        ]
    )
    return Pipeline(
        [
            ("prepare", transform),
            (
                "forest",
                classifier(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=2,
                ),
            ),
        ]
    )


def apply_threshold(results: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a finite number between 0 and 1.")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number between 0 and 1.")
    updated = results.copy()
    updated["prediction"] = np.where(updated["attack_score"] >= threshold, "Attack", "Normal")
    return updated.loc[:, ["record", "prediction", "attack_score"]]


def predict(
    frame: pd.DataFrame, pipeline: Pipeline, *, threshold: float = THRESHOLD
) -> pd.DataFrame:
    features = prepare_features(frame)
    attack_index = list(pipeline.classes_).index(1)
    scores = pipeline.predict_proba(features)[:, attack_index]
    return apply_threshold(
        pd.DataFrame({"record": np.arange(1, len(frame) + 1), "attack_score": scores}),
        threshold,
    )


def filter_records(
    records: pd.DataFrame,
    *,
    prediction: str = "All",
    protocols: list[str] | None = None,
    services: list[str] | None = None,
    score_range: tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    mask = records.attack_score.between(*score_range)
    if prediction != "All":
        mask &= records.prediction.eq(prediction)
    if protocols:
        mask &= records.proto.isin(protocols)
    if services:
        mask &= records.service.isin(services)
    return records.loc[mask].copy()


def export_results(results: pd.DataFrame, threshold: float, model_name: str) -> bytes:
    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown model.")
    return (
        apply_threshold(results, threshold)
        .assign(threshold=threshold, model=MODEL_NAMES[model_name])
        .to_csv(index=False)
        .encode("utf-8")
    )


def load_bundle(directory: Path) -> tuple[Pipeline, dict]:
    """Only load application-owned artifacts, never user-supplied joblib files."""
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(
            "Model bundle needs upgrading. Run train.py to generate a version 2 bundle."
        )
    if metadata["features"] != list(FEATURES) or metadata["threshold"] != THRESHOLD:
        raise ValueError("Model schema does not match this application. Retrain the model.")
    if sha256(directory / "model.joblib") != metadata["model_sha256"]:
        raise ValueError("Model checksum does not match metadata. Restore the model bundle.")
    if metadata["selected_model"] not in MODEL_NAMES:
        raise ValueError("Unknown model in bundle.")
    return joblib.load(directory / "model.joblib"), metadata
