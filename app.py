"""Session-only batch investigation console for UNSW-NB15-compatible features."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from nids import (
    CATEGORICAL,
    DATASET_PAGE,
    FEATURES,
    MODEL_NAMES,
    NUMERIC,
    THRESHOLD,
    apply_threshold,
    export_results,
    filter_records,
    load_bundle,
    predict,
    prepare_features,
    read_csv,
)

ROOT = Path(__file__).resolve().parent
FILTER_KEYS = (
    "prediction_filter",
    "protocol_filter",
    "service_filter",
    "score_filter",
    "record_picker",
    "sort_order",
)
BATCH_KEYS = ("results", "input_features", "analysis_timestamp", "analysis_signature")
METRICS = {
    "accuracy": "Accuracy",
    "attack_precision": "Attack precision",
    "attack_recall": "Attack recall",
    "attack_f1": "Attack F1",
    "false_positive_rate": "False-positive rate",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
}

st.set_page_config(page_title="Network intrusion analysis", page_icon="🛡️", layout="wide")


@st.cache_resource(show_spinner="Loading the trained model…")
def get_model(bundle_stamp: int):
    return load_bundle(ROOT / "artifacts")


def clear_batch():
    for key in BATCH_KEYS + FILTER_KEYS:
        st.session_state.pop(key, None)


def comparison_table(reports: dict):
    table = pd.DataFrame(reports).reindex(METRICS).rename(index=METRICS)
    st.dataframe(table.astype(float).style.format("{:.3f}", na_rep="Unavailable"), width="stretch")


def score_chart(results, threshold):
    counts, edges = np.histogram(results.attack_score, bins=np.linspace(0, 1, 21))
    bins = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "Records": counts})
    bins["Decision region"] = np.where(
        (bins.start + bins.end) / 2 >= threshold, "Flagged", "Normal"
    )
    bars = (
        alt.Chart(bins)
        .mark_bar()
        .encode(
            x=alt.X("start:Q", title="Attack score", scale=alt.Scale(domain=[0, 1])),
            x2="end:Q",
            y=alt.Y("Records:Q"),
            color=alt.Color(
                "Decision region:N",
                scale=alt.Scale(domain=["Normal", "Flagged"], range=["#22d3ee", "#fb7185"]),
            ),
            tooltip=[
                alt.Tooltip("start:Q", title="From", format=".2f"),
                alt.Tooltip("end:Q", title="To", format=".2f"),
                "Records:Q",
            ],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(color="#fbbf24", strokeDash=[5, 4])
        .encode(x="threshold:Q")
    )
    st.altair_chart((bars + marker).properties(height=235), width="stretch")


def breakdown(features, column, label):
    counts = (
        features[column]
        .fillna("(missing)")
        .value_counts()
        .head(10)
        .rename_axis(label)
        .reset_index(name="Records")
    )
    chart = (
        alt.Chart(counts)
        .mark_bar(color="#22d3ee", cornerRadiusEnd=3)
        .encode(
            x=alt.X("Records:Q"),
            y=alt.Y(f"{label}:N", sort="-x"),
            tooltip=[f"{label}:N", "Records:Q"],
        )
        .properties(height=205)
    )
    st.altair_chart(chart, width="stretch")


def overview(results, features, metadata, threshold):
    st.subheader("Batch overview")
    if results is None:
        st.info("Choose an input above and analyze the batch to populate this console.")
        return
    # Overview statistics describe the complete batch, independent of investigation filters.
    flagged = int(results.prediction.eq("Attack").sum())
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ["Loaded records", "Flagged records", "Predicted normal", "Flagged share"],
        [
            f"{len(results):,}",
            f"{flagged:,}",
            f"{len(results) - flagged:,}",
            f"{flagged / len(results):.1%}",
        ],
    ):
        column.metric(label, value, border=True)
    st.caption(
        f"Analysis time (UTC): {st.session_state['analysis_timestamp']} · Active threshold: {threshold:.2f}"
    )
    st.markdown("#### Attack score distribution")
    score_chart(results, threshold)
    st.caption(
        "The amber line is the decision threshold. Scores are model outputs, not calibrated probabilities or proof of an intrusion."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Top protocols")
        breakdown(features, "proto", "Protocol")
    with right:
        st.markdown("#### Top services")
        breakdown(features, "service", "Service")
    st.download_button(
        "Download all predictions",
        export_results(results, threshold, metadata["selected_model"]),
        file_name="all_predictions.csv",
        mime="text/csv",
        key="download_all",
    )


def investigate(results, features, metadata, threshold):
    st.subheader("Record investigation")
    if results is None:
        st.info("Analyze a batch to inspect and filter individual records.")
        return
    records = results.assign(
        proto=features.proto.fillna("(missing)").astype(str).to_numpy(),
        service=features.service.fillna("(missing)").astype(str).to_numpy(),
    )
    first, second, third = st.columns(3)
    prediction = first.selectbox("Prediction", ["All", "Attack", "Normal"], key="prediction_filter")
    protocols = second.multiselect(
        "Protocols · empty means all", sorted(records.proto.unique()), key="protocol_filter"
    )
    services = third.multiselect(
        "Services · empty means all", sorted(records.service.unique()), key="service_filter"
    )
    score_range = st.slider("Score range", 0.0, 1.0, (0.0, 1.0), 0.05, key="score_filter")
    filtered = filter_records(
        records,
        prediction=prediction,
        protocols=protocols,
        services=services,
        score_range=score_range,
    )
    order = st.selectbox(
        "Sort records", ["Highest score", "Lowest score", "Record order"], key="sort_order"
    )
    if order == "Record order":
        filtered = filtered.sort_values("record", kind="stable")
    else:
        filtered = filtered.sort_values(
            ["attack_score", "record"], ascending=[order == "Lowest score", True]
        )
    st.caption(
        f"{len(filtered):,} of {len(records):,} records match. Original CSV record numbers are preserved."
    )
    st.dataframe(
        filtered.style.map(
            lambda value: (
                "color: #fb7185; font-weight: bold" if value == "Attack" else "color: #22d3ee"
            ),
            subset=["prediction"],
        ),
        hide_index=True,
        width="stretch",
        height=280,
        column_config={
            "attack_score": st.column_config.ProgressColumn(
                "Attack score", min_value=0, max_value=1, format="%.3f"
            ),
            "proto": "Protocol",
            "service": "Service",
        },
        key="investigation_table",
    )
    st.download_button(
        "Download filtered predictions",
        export_results(filtered, threshold, metadata["selected_model"]),
        file_name="filtered_predictions.csv",
        mime="text/csv",
        key="download_filtered",
    )
    if filtered.empty:
        st.info("No records match these filters. Widen the score range or clear a filter.")
        return
    choices = filtered.record.tolist()
    if st.session_state.get("record_picker") not in choices:
        st.session_state.pop("record_picker", None)
    record = st.selectbox(
        "Inspect original record",
        choices,
        format_func=lambda value: f"Record {value}",
        key="record_picker",
    )
    inputs = features.iloc[record - 1]
    decision = results.loc[results.record.eq(record)].iloc[0]
    st.markdown(f"#### Record {record}")
    left, middle, right = st.columns(3)
    left.metric("Attack score", f"{decision.attack_score:.3f}")
    middle.metric("Decision", decision.prediction)
    right.metric("Missing input cells", int(inputs.isna().sum()))
    st.caption(
        f"{MODEL_NAMES[metadata['selected_model']]} · Threshold {threshold:.2f} · A score equal to the threshold is flagged."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": FEATURES,
                "Input value": ["(missing)" if pd.isna(value) else str(value) for value in inputs],
                "Missing": inputs.isna().to_numpy(),
            }
        ),
        hide_index=True,
        width="stretch",
        height=340,
    )
    st.caption(
        "These are validated input values. Missing numbers use fitting-set medians; missing categories use the fitted missing-category representation."
    )


def model_evaluation(metadata, threshold):
    st.subheader("Validation and model selection")
    split = metadata["split"]
    st.write(
        f"{split['fitting_rows']:,} fitting records · {split['validation_rows']:,} validation records · {split['shared_feature_hashes']} shared feature hashes"
    )
    st.caption(
        "Both models use the same grouped split. Preprocessing is fitted only on fitting records. Selection uses validation average precision, with Random Forest winning an exact tie; the selected pipeline is not refitted."
    )
    comparison_table(
        {MODEL_NAMES[name]: report for name, report in metadata["validation"]["models"].items()}
    )
    st.markdown(f"**Selected model: {MODEL_NAMES[metadata['selected_model']]}**")
    st.markdown("#### Validation tradeoff at the active threshold")
    active = next(
        row
        for row in metadata["validation"]["thresholds"]
        if abs(row["threshold"] - threshold) < 1e-9
    )
    columns = st.columns(3)
    for column, name in zip(columns, ["attack_precision", "attack_recall", "false_positive_rate"]):
        column.metric(METRICS[name], f"{active[name]:.1%}")
    st.caption(
        f"Validation results at threshold {threshold:.2f}. Raising the threshold generally reduces flags and can miss more attacks. These are benchmark validation results, not estimates of uploaded-data accuracy."
    )
    left, right = st.columns(2)
    for container, curve, x_label, y_label, title in [
        (left, "precision_recall", "Recall", "Precision", "Validation precision–recall"),
        (right, "roc", "False-positive rate", "True-positive rate", "Validation ROC"),
    ]:
        with container:
            st.markdown(f"#### {title}")
            points = pd.DataFrame(metadata["validation"]["curves"][curve]).assign(
                point=lambda frame: np.arange(len(frame))
            )
            chart = (
                alt.Chart(points)
                .mark_line(color="#22d3ee")
                .encode(
                    x=alt.X("x:Q", title=x_label, scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("y:Q", title=y_label, scale=alt.Scale(domain=[0, 1])),
                    order="point:Q",
                )
                .properties(height=210)
            )
            st.altair_chart(chart, width="stretch")
    st.caption(metadata["validation"]["curves"]["note"])
    st.markdown("#### Global feature importance")
    importance = (
        pd.DataFrame(
            metadata["global_feature_importance"].items(), columns=["Feature", "Importance"]
        )
        .sort_values("Importance", ascending=False)
        .head(10)
    )
    chart = (
        alt.Chart(importance)
        .mark_bar(color="#22d3ee")
        .encode(
            x="Importance:Q",
            y=alt.Y("Feature:N", sort="-x"),
            tooltip=["Feature:N", alt.Tooltip("Importance:Q", format=".4f")],
        )
        .properties(height=260)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Top 10 training-wide, impurity-based importances, grouped to original features. These can be biased toward some feature types and are not causal explanations for individual predictions."
    )
    st.divider()
    st.subheader("Separate test evaluation · fixed threshold 0.50")
    st.info(
        "The session threshold does not change this saved test report. Test data was excluded from fitting and model selection."
    )
    population = st.selectbox(
        "Test population",
        ["Complete test file", "No original-training feature hashes"],
        key="test_population",
    )
    key = "full_test" if population == "Complete test file" else "nonoverlap_test"
    report = metadata["evaluation"][key]
    st.caption(f"{report['rows']:,} test records · Fixed threshold {THRESHOLD:.2f}")
    if report["selected_model"] is None:
        st.warning("No records are available in this test subset.")
    else:
        comparison_table(
            {
                MODEL_NAMES[metadata["selected_model"]]: report["selected_model"],
                "Fitting-majority baseline": report["majority_baseline"],
            }
        )
        st.table(
            pd.DataFrame(
                report["selected_model"]["confusion_matrix"],
                index=["Actual normal", "Actual attack"],
                columns=["Predicted normal", "Predicted attack"],
            )
        )
        if report["selected_model"]["roc_auc"] is None:
            st.caption(
                "Ranking metrics are unavailable because this subset contains only one class."
            )
    overlap = metadata["test_records_with_feature_hash_in_training"]
    st.warning(
        f"{overlap:,} test records have feature hashes found in the original training partition. The complete-file results include them. Hash separation is not proof of statistical independence."
    )
    st.caption(
        "Precision: share of flags that are attacks. Recall: share of attacks detected. F1 balances both. False-positive rate: share of normal records flagged. ROC-AUC summarizes ranking; average precision summarizes precision–recall performance. The baseline always predicts the fitting set's majority class."
    )
    st.warning(
        "Historical laboratory benchmark. Detection quality on a real network is unknown. This educational console provides no live capture, blocking, or operational certification."
    )
    with st.expander("Artifact details and provenance"):
        st.write(metadata["provenance_note"])
        st.json(metadata)
    with st.expander("Dataset attribution"):
        st.markdown((ROOT / "ATTRIBUTION.md").read_text())


def data_guide():
    st.subheader("CSV data contract")
    st.write(
        "Each row must contain UNSW-NB15-compatible connection features. Raw packets, PCAPs, and arbitrary firewall logs need a separate feature extractor. Renaming unrelated columns does not make their meanings compatible."
    )
    st.write(
        "All 42 columns are required in any order. Empty cells are accepted; unknown categories are encoded without a matching category indicator. Extra columns, including id, label, and attack_cat, never enter the model."
    )
    st.caption(
        "UTF-8 CSV · 10 MB maximum (10,000,000 bytes) · 10,000 records maximum · finite numeric values only"
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Column": FEATURES,
                "Type": ["Category" if name in CATEGORICAL else "Numeric" for name in FEATURES],
            }
        ),
        hide_index=True,
        width="stretch",
        height=360,
    )
    st.caption(
        f"Categories: {', '.join(CATEGORICAL)}; {len(NUMERIC)} numeric features. Output: record, prediction, attack_score, threshold, model. Record numbers start at 1 and follow CSV record order; filtered exports keep those numbers."
    )
    st.download_button(
        "Download synthetic example CSV",
        (ROOT / "sample.csv").read_bytes(),
        file_name="synthetic_network_features.csv",
        mime="text/csv",
        key="download_sample",
    )
    st.caption(
        "The five synthetic example records have no known attack labels. Their predictions demonstrate the workflow, not model quality."
    )
    st.link_button("Feature definitions and dataset papers", DATASET_PAGE)


def main():
    # This stylesheet is application-owned; uploaded values never enter HTML or CSS.
    st.markdown("<style>" + (ROOT / "style.css").read_text() + "</style>", unsafe_allow_html=True)
    st.markdown(
        '<div class="console-kicker">NETWORK DEFENSE / ANALYSIS CONSOLE</div>',
        unsafe_allow_html=True,
    )
    st.title("Network intrusion analysis")
    st.caption("BATCH ANALYSIS · EDUCATIONAL BENCHMARK · SESSION-ONLY DATA")
    try:
        model, metadata = get_model((ROOT / "artifacts/metadata.json").stat().st_mtime_ns)
    except (OSError, ValueError, KeyError, EOFError):
        st.error(
            "The model bundle is missing or incompatible. Restore a complete version 2 bundle or run train.py."
        )
        st.stop()
    with st.sidebar:
        st.markdown("### Analysis console")
        st.caption("MODEL ARTIFACT")
        st.markdown(f"**{MODEL_NAMES[metadata['selected_model']]}**")
        st.caption(
            f"{metadata['model_parameters'][metadata['selected_model']]['n_estimators']} trees · {len(FEATURES)} input features"
        )
        st.caption("Selected on validation data; no test-based selection.")
        st.divider()
        st.markdown("**Session handling**")
        st.caption(
            "Public or synthetic records only. Uploads are processed on this server, held in session memory, and excluded from shared caches, disk storage, and record logs."
        )
        st.caption("No live capture or automatic blocking.")
        st.link_button("Dataset documentation", DATASET_PAGE)
    st.markdown("#### Batch controls")
    left, right = st.columns([1, 2])
    with left:
        source = st.radio(
            "Traffic source",
            ["Synthetic example", "Upload CSV"],
            horizontal=True,
            key="traffic_source",
        )
    with right:
        if source == "Synthetic example":
            data = (ROOT / "sample.csv").read_bytes()
            st.caption(
                "Five hand-created records. No known attack labels. Select Analyze batch to explore."
            )
        else:
            uploaded = st.file_uploader(
                "Choose network feature records", type=["csv"], key="traffic_csv"
            )
            data = uploaded.getvalue() if uploaded is not None else None
    signature = (source, hashlib.sha256(data).hexdigest()) if data is not None else None
    if st.session_state.get("input_signature") != signature:
        clear_batch()
        st.session_state["input_signature"] = signature
    already_scored = (
        "results" in st.session_state and st.session_state.get("analysis_signature") == signature
    )
    clicked = st.button(
        "Analyze batch", type="primary", key="analyze", disabled=data is None or already_scored
    )
    if clicked and not already_scored:
        clear_batch()
        try:
            frame = read_csv(data)
            with st.spinner("Validating and scoring the batch…"):
                results = predict(frame, model)
                features = prepare_features(frame)
            st.session_state.update(
                results=results,
                input_features=features.reset_index(drop=True),
                analysis_signature=signature,
                analysis_timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            st.rerun()
        except ValueError as error:
            st.error(str(error))
    threshold = st.slider(
        "Decision threshold",
        0.05,
        0.95,
        THRESHOLD,
        0.05,
        key="threshold",
        help="Scores at or above this threshold are flagged. Changing it reuses stored model scores.",
    )
    st.caption("Threshold changes reuse scores. The saved test evaluation always uses 0.50.")
    results = (
        apply_threshold(st.session_state["results"], threshold)
        if "results" in st.session_state
        else None
    )
    features = st.session_state.get("input_features")
    summary, investigation, evaluation, guide = st.tabs(
        ["Overview", "Investigate", "Model evaluation", "Data guide"]
    )
    with summary:
        overview(results, features, metadata, threshold)
    with investigation:
        investigate(results, features, metadata, threshold)
    with evaluation:
        model_evaluation(metadata, threshold)
    with guide:
        data_guide()


if __name__ == "__main__":
    main()
