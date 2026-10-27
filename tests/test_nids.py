import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import download_data
from nids import (
    BUNDLE_VERSION,
    FEATURES,
    MAX_BYTES,
    MAX_ROWS,
    THRESHOLD,
    apply_threshold,
    build_pipeline,
    export_results,
    filter_records,
    load_bundle,
    predict,
    prepare_features,
    read_csv,
    sha256,
)
from train import evaluate, global_importance, load_partition, select_model, split_training, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frame():
    return read_csv((ROOT / "sample.csv").read_bytes())


@pytest.fixture
def model(frame):
    return build_pipeline().fit(prepare_features(frame), [0, 1, 0, 1, 0])


def test_schema_and_targets_are_excluded(frame, model):
    expected = predict(frame, model)
    frame["label"] = [1, 0, 1, 0, 1]
    frame["attack_cat"] = "Exploits"
    frame["id"] = np.arange(len(frame)) + 999
    frame["irrelevant"] = "=untrusted input"
    reordered = frame.loc[:, list(reversed(frame.columns))]
    pd.testing.assert_frame_equal(predict(reordered, model), expected)
    assert list(model.feature_names_in_) == list(FEATURES)
    assert list(expected.columns) == ["record", "prediction", "attack_score"]
    assert expected["record"].tolist() == list(range(1, len(frame) + 1))
    assert expected["attack_score"].between(0, 1).all()
    assert ((expected.attack_score >= THRESHOLD) == expected.prediction.eq("Attack")).all()


def test_missing_cells_and_unknown_categories(frame, model):
    frame.loc[0, "proto"] = "unseen-protocol"
    frame.loc[1, "dur"] = np.nan
    frame.loc[2, "service"] = "  "
    frame.loc[3, "state"] = None
    result = predict(frame, model)
    assert np.isfinite(result.attack_score).all()
    prepared = prepare_features(frame)
    assert pd.isna(prepared.loc[2, "service"])


@pytest.mark.parametrize("value", ["abc", "Infinity", "-inf", "1e40", "1e999"])
def test_invalid_numeric(frame, value):
    frame.loc[0, "dur"] = value
    with pytest.raises(ValueError, match="Column 'dur'"):
        prepare_features(frame)


def test_missing_column(frame):
    with pytest.raises(ValueError, match="Missing required columns: dur"):
        prepare_features(frame.drop(columns="dur"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "empty"),
        (b"dur,proto\n", "no records"),
        (b"dur,dur\n1,2\n", "Duplicate"),
        (b"dur,\n1,2\n", "non-empty header"),
        (b"dur,proto\n1,tcp,extra\n", "Malformed"),
        (b"dur,proto\n1\n", "Malformed"),
        (b'dur,proto\n1,"tcp\n', "Malformed"),
        (b"dur\n\xff", "UTF-8"),
        (b"dur\n1\x00", "null bytes"),
    ],
)
def test_invalid_csv(payload, message):
    with pytest.raises(ValueError, match=message):
        read_csv(payload)


def test_bom_quoted_values_and_limits():
    parsed = read_csv('\ufeffdur,proto\n1,"tcp,extra"\n'.encode())
    assert parsed.columns.tolist() == ["dur", "proto"]
    assert parsed.loc[0, "proto"] == "tcp,extra"
    assert len(read_csv(b"dur\n" + b"1\n" * MAX_ROWS)) == MAX_ROWS
    with pytest.raises(ValueError, match="10,000"):
        read_csv(b"dur\n" + b"1\n" * (MAX_ROWS + 1))
    with pytest.raises(ValueError, match="10 MB"):
        read_csv(b"x" * (MAX_BYTES + 1))


def test_bundle_roundtrip_and_checksum(tmp_path, frame, model):
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "features": list(FEATURES),
                "bundle_version": BUNDLE_VERSION,
                "selected_model": "random_forest",
                "threshold": THRESHOLD,
                "model_sha256": sha256(tmp_path / "model.joblib"),
            }
        )
    )
    restored, _ = load_bundle(tmp_path)
    pd.testing.assert_frame_equal(predict(frame, model), predict(frame, restored))
    with (tmp_path / "model.joblib").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(tmp_path)


def test_metric_definitions():
    result = evaluate(pd.Series([0, 0, 1, 1]), np.array([0.1, 0.8, 0.4, 0.9]))
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["false_positive_rate"] == 0.5
    assert result["attack_precision"] == result["attack_recall"] == result["attack_f1"] == 0.5
    assert result["roc_auc"] == 0.75


def test_training_and_baseline_end_to_end(tmp_path, frame, monkeypatch):
    import train as training_module

    training = pd.concat([frame] * 12, ignore_index=True)
    training["dur"] = np.arange(len(training)) + 0.25
    training["proto"] = [f"protocol-{i}" for i in range(len(training))]
    training["label"] = [0, 1, 0, 1, 0] * 12
    training["attack_cat"] = ["Normal", "Generic", "Normal", "DoS", "Normal"] * 12
    training["id"] = range(len(training))
    testing = training.copy()
    testing["dur"] = pd.to_numeric(testing["dur"]) + 0.123
    training.to_csv(tmp_path / "train.csv", index=False)
    testing.to_csv(tmp_path / "test.csv", index=False)
    output = tmp_path / "bundle"
    selected = []
    original_load = load_partition

    def guarded_load(path):
        if path.name == "test.csv":
            assert selected, "Test data was read before model selection"
        return original_load(path)

    def selection(reports):
        choice = select_model(reports)
        selected.append(choice)
        return choice

    monkeypatch.setattr(training_module, "load_partition", guarded_load)
    monkeypatch.setattr(training_module, "select_model", selection)
    report = train(tmp_path / "train.csv", tmp_path / "test.csv", output)
    assert report["training"]["rows"] == report["test"]["rows"] == 60
    assert report["evaluation"]["full_test"]["majority_baseline"]["confusion_matrix"] == [
        [36, 0],
        [24, 0],
    ]
    assert report["test_records_with_feature_hash_in_training"] == 0
    assert report["features"] == list(FEATURES)
    model, saved = load_bundle(output)
    assert saved == report
    assert len(predict(frame, model)) == 5
    features = prepare_features(training)
    fitting, validation, _ = split_training(features, training.label)
    imputer = model.named_steps["prepare"].named_transformers_["numeric"]
    assert imputer.statistics_[0] == features.iloc[fitting].dur.median()
    encoder = model.named_steps["prepare"].named_transformers_["categorical"].named_steps["encode"]
    assert set(encoder.categories_[0]) == set(features.iloc[fitting].proto)
    assert set(encoder.categories_[0]).isdisjoint(set(features.iloc[validation].proto))
    assert report["selected_model"] == selected[0]
    assert len(report["validation"]["thresholds"]) == 19
    assert set(report["global_feature_importance"]) == set(FEATURES)
    with pytest.raises(ValueError, match="already exists"):
        train(tmp_path / "train.csv", tmp_path / "test.csv", output)
    with pytest.raises(ValueError, match="separate partitions"):
        train(tmp_path / "train.csv", tmp_path / "train.csv", tmp_path / "other")


def test_grouped_split_is_reproducible_and_keeps_duplicates_together(frame):
    base = pd.concat([frame] * 6, ignore_index=True)
    base["dur"] = np.arange(len(base))
    features = prepare_features(pd.concat([base, base], ignore_index=True))
    labels = pd.Series(([0, 1, 0, 1, 0] * 6) * 2)
    fitting, validation, details = split_training(features, labels)
    other_fitting, other_validation, other_details = split_training(features, labels)
    np.testing.assert_array_equal(fitting, other_fitting)
    np.testing.assert_array_equal(validation, other_validation)
    assert details == other_details
    hashes = pd.util.hash_pandas_object(features, index=False)
    assert set(hashes.iloc[fitting]).isdisjoint(set(hashes.iloc[validation]))
    assert len(fitting) + len(validation) == len(features)
    assert details["shared_feature_hashes"] == 0


def test_model_selection_and_independent_preprocessing():
    reports = {
        "random_forest": {"average_precision": 0.9},
        "extra_trees": {"average_precision": 0.9},
    }
    assert select_model(reports) == "random_forest"
    reports["extra_trees"]["average_precision"] = 0.91
    assert select_model(reports) == "extra_trees"
    forest, extra = build_pipeline(), build_pipeline("extra_trees")
    assert forest.named_steps["prepare"] is not extra.named_steps["prepare"]
    assert forest.named_steps["forest"].bootstrap is True
    assert extra.named_steps["forest"].bootstrap is False
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("unknown")


def test_thresholds_filtering_and_export():
    scores = pd.DataFrame({"record": [7, 3, 9], "attack_score": [0.05, 0.5, 0.95]})
    original = scores.copy()
    result = apply_threshold(scores, 0.5)
    assert result.prediction.tolist() == ["Normal", "Attack", "Attack"]
    assert apply_threshold(scores, 0).prediction.eq("Attack").all()
    assert apply_threshold(scores, 1).prediction.eq("Normal").all()
    pd.testing.assert_frame_equal(scores, original)
    records = result.assign(proto=["tcp", "udp", "tcp"], service=["http", "dns", "ssh"])
    filtered = filter_records(
        records, prediction="Attack", protocols=["tcp"], services=["ssh"], score_range=(0.95, 0.95)
    )
    assert filtered.record.tolist() == [9]
    assert filter_records(records, protocols=["unknown"]).empty
    exported = pd.read_csv(io.BytesIO(export_results(filtered, 0.95, "random_forest")))
    assert exported.columns.tolist() == [
        "record",
        "prediction",
        "attack_score",
        "threshold",
        "model",
    ]
    assert exported.record.tolist() == [9]
    assert exported.prediction.tolist() == ["Attack"]
    assert exported.model.tolist() == ["Random Forest"]
    assert (
        len(pd.read_csv(io.BytesIO(export_results(filtered.iloc[:0], 0.5, "random_forest")))) == 0
    )


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf"), "0.5", True, None])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="Threshold"):
        apply_threshold(pd.DataFrame({"record": [1], "attack_score": [0.5]}), threshold)


def test_global_importance_aggregates_categories(model):
    importance = global_importance(model)
    assert len(importance) == 42
    assert sum(importance.values()) == pytest.approx(1)
    assert all(value >= 0 for value in importance.values())


def test_single_class_metrics_are_explicitly_unavailable():
    report = evaluate(pd.Series([0, 0]), np.array([0.1, 0.8]))
    assert report["roc_auc"] is report["average_precision"] is None
    assert report["false_positive_rate"] == 0.5


def test_empty_nonoverlap_test_subset(tmp_path, frame):
    training = pd.concat([frame] * 6, ignore_index=True)
    training["dur"] = np.arange(len(training))
    training["label"] = [0, 1, 0, 1, 0] * 6
    testing = training.assign(id=range(len(training)))
    training.to_csv(tmp_path / "train.csv", index=False)
    testing.to_csv(tmp_path / "test.csv", index=False)
    report = train(tmp_path / "train.csv", tmp_path / "test.csv", tmp_path / "bundle")
    assert report["test_records_with_feature_hash_in_training"] == len(training)
    assert report["evaluation"]["nonoverlap_test"] == {
        "rows": 0,
        "selected_model": None,
        "majority_baseline": None,
    }


def test_legacy_bundle_is_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps({"features": list(FEATURES), "threshold": 0.5})
    )
    with pytest.raises(ValueError, match="version 2"):
        load_bundle(tmp_path)


@pytest.mark.parametrize("labels", [[0, 0, 0, 0, 0], [0, 1, 2, 0, 1], [0, 1, "", 0, 1]])
def test_invalid_training_labels(tmp_path, frame, labels):
    frame["label"] = labels
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="labels must contain both"):
        load_partition(path)


def test_download_verifies_hash_and_preserves_existing_file(tmp_path, monkeypatch):
    payload = b"fake public dataset for isolated download test"
    monkeypatch.setattr(
        download_data, "DATA_FILES", {"test.csv": {"sha256": hashlib.sha256(payload).hexdigest()}}
    )
    calls = []

    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(download_data.urllib.request, "urlopen", fetch)
    download_data.download(tmp_path)
    assert (tmp_path / "test.csv").read_bytes() == payload
    download_data.download(tmp_path)
    assert len(calls) == 1
    (tmp_path / "test.csv").write_bytes(b"existing work")
    with pytest.raises(ValueError, match="left unchanged"):
        download_data.download(tmp_path)
    assert (tmp_path / "test.csv").read_bytes() == b"existing work"


def test_download_rejects_changed_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(download_data, "DATA_FILES", {"test.csv": {"sha256": "wrong"}})
    monkeypatch.setattr(
        download_data.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"changed")
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_data.download(tmp_path)
    assert list(tmp_path.iterdir()) == []
