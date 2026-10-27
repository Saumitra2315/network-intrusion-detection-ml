import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import nids

ROOT = Path(__file__).resolve().parents[1]


def start():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=20).run()
    assert not app.exception
    return app


def analyze(app):
    app.button(key="analyze").click().run()
    assert not app.exception
    return app


def investigation_rows(app):
    return next(view.value for view in app.dataframe if "record" in view.value.columns)


def test_example_results_download_and_session_isolation():
    app = analyze(start())
    scores = app.session_state["results"]
    assert len(scores) == 5
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Loaded records"] == "5"
    assert int(metrics["Flagged records"]) == scores.prediction.eq("Attack").sum()
    assert app.button(key="analyze").disabled
    assert "analysis_timestamp" in app.session_state
    assert app.selectbox(key="record_picker").value in scores.record.tolist()
    download = next(
        button for button in app.download_button if button.label == "Download all predictions"
    )
    assert download.proto.url.endswith(".csv")
    download.click().run()
    assert not app.exception
    assert len(app.session_state["results"]) == 5
    other_session = start()
    assert "results" not in other_session.session_state
    app.radio(key="traffic_source").set_value("Upload CSV").run()
    assert "results" not in app.session_state
    assert "input_features" not in app.session_state
    assert app.button(key="analyze").disabled


def test_threshold_changes_reuse_scores_without_inference(monkeypatch):
    calls = []
    original = nids.predict

    def counted_predict(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(nids, "predict", counted_predict)
    app = analyze(start())
    original_scores = app.session_state["results"].copy()
    timestamp = app.session_state["analysis_timestamp"]
    for threshold in (0.05, 0.95, 0.5):
        app.slider(key="threshold").set_value(threshold).run()
        assert not app.exception
        pd.testing.assert_frame_equal(original_scores, app.session_state["results"])
        assert app.session_state["analysis_timestamp"] == timestamp
        rows = investigation_rows(app)
        assert ((rows.attack_score >= threshold) == rows.prediction.eq("Attack")).all()
        assert (
            app.table[0].value.to_numpy().tolist()
            == json.loads((ROOT / "artifacts/metadata.json").read_text())["evaluation"][
                "full_test"
            ]["selected_model"]["confusion_matrix"]
        )
    assert len(calls) == 1


def test_filters_sorting_inspection_and_empty_results():
    app = analyze(start())
    app.slider(key="threshold").set_value(0.05).run()
    app.selectbox(key="prediction_filter").set_value("Attack").run()
    app.multiselect(key="protocol_filter").set_value(["tcp"]).run()
    app.multiselect(key="service_filter").set_value(["http"]).run()
    assert not app.exception
    assert investigation_rows(app).record.tolist() == [1]
    assert app.selectbox(key="record_picker").value == 1
    assert any("1 of 5 records match" in caption.value for caption in app.caption)
    app.multiselect(key="service_filter").set_value([]).run()
    app.multiselect(key="protocol_filter").set_value([]).run()
    app.selectbox(key="sort_order").set_value("Record order").run()
    assert investigation_rows(app).record.tolist() == [1, 2, 3, 4, 5]
    app.selectbox(key="sort_order").set_value("Lowest score").run()
    assert investigation_rows(app).attack_score.is_monotonic_increasing
    app.selectbox(key="record_picker").set_value(3).run()
    assert any(heading.value == "#### Record 3" for heading in app.markdown)
    app.slider(key="score_filter").set_value((0.0, 0.0)).run()
    assert not app.exception
    assert investigation_rows(app).empty
    assert not any(select.key == "record_picker" for select in app.selectbox)
    assert any("No records match" in item.value for item in app.info)
    empty_csv = nids.export_results(investigation_rows(app), 0.05, "random_forest")
    assert pd.read_csv(io.BytesIO(empty_csv)).empty


def test_valid_upload_missing_cells_and_replacement_clears_stale_predictions():
    app = start()
    app.radio(key="traffic_source").set_value("Upload CSV").run()
    frame = pd.read_csv(ROOT / "sample.csv", dtype=str)
    frame.loc[0, "dur"] = np.nan
    app.file_uploader[0].set_value(
        ("public.csv", frame.to_csv(index=False).encode(), "text/csv")
    ).run()
    analyze(app)
    assert len(app.session_state["results"]) == 5
    app.selectbox(key="record_picker").set_value(1).run()
    assert (
        next(metric.value for metric in app.metric if metric.label == "Missing input cells") == "1"
    )
    app.multiselect(key="protocol_filter").set_value(["tcp"]).run()
    app.file_uploader[0].set_value(("invalid.csv", b"dur\n1\n", "text/csv")).run()
    assert "results" not in app.session_state
    assert "input_features" not in app.session_state
    analyze(app)
    assert "Missing required columns" in app.error[0].value
    assert not any(button.label == "Download all predictions" for button in app.download_button)
    app.file_uploader[0].set_value(
        ("replacement.csv", (ROOT / "sample.csv").read_bytes(), "text/csv")
    ).run()
    analyze(app)
    assert app.multiselect(key="protocol_filter").value == []
    app.file_uploader[0].clear().run()
    assert "results" not in app.session_state


def test_test_population_switch_and_fixed_threshold_report():
    metadata = json.loads((ROOT / "artifacts/metadata.json").read_text())
    app = start()
    assert [tab.label for tab in app.tabs] == [
        "Overview",
        "Investigate",
        "Model evaluation",
        "Data guide",
    ]
    assert any(f"{metadata['split']['fitting_rows']:,}" in item.value for item in app.markdown)
    assert any(
        f"{metadata['test_records_with_feature_hash_in_training']:,}" in item.value
        for item in app.warning
    )
    assert len(app.dataframe[-1].value) == 42
    app.selectbox(key="test_population").set_value("No original-training feature hashes").run()
    assert not app.exception
    assert (
        app.table[0].value.to_numpy().tolist()
        == metadata["evaluation"]["nonoverlap_test"]["selected_model"]["confusion_matrix"]
    )
    app.slider(key="threshold").set_value(0.95).run()
    assert not app.exception
    assert (
        app.table[0].value.to_numpy().tolist()
        == metadata["evaluation"]["nonoverlap_test"]["selected_model"]["confusion_matrix"]
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [(b"dur,dur\n1,2\n", "Duplicate"), (b'dur,proto\n1,"tcp', "Malformed"), (b"", "empty")],
)
def test_upload_errors_are_visible(payload, message):
    app = start()
    app.radio(key="traffic_source").set_value("Upload CSV").run()
    app.file_uploader[0].set_value(("bad.csv", payload, "text/csv")).run()
    analyze(app)
    assert message in app.error[0].value


def test_numeric_upload_error():
    frame = pd.read_csv(ROOT / "sample.csv", dtype=str)
    frame.loc[0, "dur"] = "not-a-number"
    app = start()
    app.radio(key="traffic_source").set_value("Upload CSV").run()
    app.file_uploader[0].set_value(
        ("invalid.csv", frame.to_csv(index=False).encode(), "text/csv")
    ).run()
    analyze(app)
    assert "Column 'dur'" in app.error[0].value
