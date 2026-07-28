from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR
from peerread_review.data import read_peerread_rows
from peerread_review.model import load_model, predict
from peerread_review.xai import explain


st.set_page_config(page_title="PeerRead Review Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_report() -> dict:
    path = DEFAULT_REPORT_DIR / "peerread_decisions.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_peerread_model() -> dict:
    return load_model(DEFAULT_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_training_summary() -> dict:
    path = DEFAULT_REPORT_DIR / "training_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_dataset_rows() -> list[dict]:
    return read_peerread_rows(DEFAULT_DATASET_PATH)


def render_overview(payload: dict, model: dict) -> None:
    st.title("PeerRead Supervised Paper Review Dashboard")
    st.caption("Separate project using true PeerRead Accept/Reject labels. Default suggestions are XAI-based.")
    counts = payload.get("counts", {})
    actual_counts = payload.get("actual_counts", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Rows", model.get("training_count", 0))
    col2.metric("Actual Accept", actual_counts.get("Accept", 0))
    col3.metric("Actual Reject", actual_counts.get("Reject", 0))
    col4.metric("Predicted Modify", counts.get("Modify", 0))

    rows = payload.get("papers", [])
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No rows found in report.")
        return

    chart_df = df["predicted_decision"].value_counts().rename_axis("decision").reset_index(name="papers")
    st.subheader("Predicted Decision Distribution")
    st.bar_chart(chart_df, x="decision", y="papers", height=320)

    actual_df = df["actual_label"].value_counts().rename_axis("actual_label").reset_index(name="papers")
    st.subheader("Actual Label Distribution")
    st.bar_chart(actual_df, x="actual_label", y="papers", height=260)


def render_table(payload: dict) -> None:
    rows = payload.get("papers", [])
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No rows found in report.")
        return
    st.subheader("Paper-by-Paper XAI Table")
    decisions = sorted(df["predicted_decision"].unique())
    selected = st.multiselect("Decision filter", decisions, default=decisions)
    query = st.text_input("Search by title, paper id, or conference")
    filtered = df[df["predicted_decision"].isin(selected)]
    if query:
        needle = query.lower()
        filtered = filtered[
            filtered["paper_id"].astype(str).str.lower().str.contains(needle, na=False)
            | filtered["title"].astype(str).str.lower().str.contains(needle, na=False)
            | filtered["conference"].astype(str).str.lower().str.contains(needle, na=False)
        ]
    columns = [
        "paper_id",
        "conference",
        "split",
        "title",
        "actual_label",
        "predicted_decision",
        "accept_probability",
        "reject_probability",
        "xai_focus",
        "suggestion_1",
        "suggestion_2",
        "suggestion_3",
    ]
    st.caption(f"Showing {len(filtered)} of {len(df)} papers.")
    st.dataframe(filtered[columns], use_container_width=True, height=560)

    st.download_button(
        "Download Filtered CSV",
        filtered[columns].to_csv(index=False).encode("utf-8"),
        file_name="peerread_filtered_decisions.csv",
        mime="text/csv",
    )


def render_evaluation() -> None:
    st.title("Dataset and Evaluation")
    training = load_training_summary()
    if not training:
        st.warning("Run training first.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", training.get("total_rows", 0))
    col2.metric("Dev Accuracy", training.get("dev_evaluation", {}).get("accuracy", 0))
    col3.metric("Test Accuracy", training.get("test_evaluation", {}).get("accuracy", 0))
    st.subheader("Training Summary")
    st.json(training)
    summary_md = DEFAULT_REPORT_DIR / "dataset_summary.md"
    if summary_md.exists():
        st.subheader("Dataset Summary")
        st.markdown(summary_md.read_text(encoding="utf-8"))


def render_figures() -> None:
    st.title("PeerRead Poster Figures")
    figure_dir = DEFAULT_REPORT_DIR / "poster_figures"
    figures = [
        figure_dir / "01_peerread_predicted_decisions.svg",
        figure_dir / "02_peerread_actual_labels.svg",
        figure_dir / "SYSTEM_ARCHITECTURE.svg",
    ]
    existing = [path for path in figures if path.exists()]
    if not existing:
        st.warning("Run `peerread_review_project/poster_figures.py` first.")
        return
    tabs = st.tabs([path.stem for path in existing])
    for tab, path in zip(tabs, existing):
        with tab:
            st.image(str(path), use_container_width=True)


def render_single_review(model: dict) -> None:
    st.title("Review One PeerRead Paper")
    rows = load_dataset_rows()
    paper_ids = [str(row.get("paper_id")) for row in rows]
    selected = st.selectbox("Paper ID", paper_ids[:1000], help="Showing first 1000 IDs for faster selection. Use search below for title filtering.")
    query = st.text_input("Optional title search")
    if query:
        matches = [row for row in rows if query.lower() in str(row.get("title", "")).lower()][:25]
        if matches:
            selected = st.selectbox("Search matches", [str(row.get("paper_id")) for row in matches])
    row = next((item for item in rows if str(item.get("paper_id")) == selected), None)
    if not row:
        return
    prediction = predict(model, row)
    xai = explain(model, prediction)
    st.subheader(row.get("title", selected))
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", prediction["decision"])
    col2.metric("Accept Probability", prediction["accept_probability"])
    col3.metric("Reject Probability", prediction["reject_probability"])
    st.markdown("**XAI Risk Factors**")
    st.dataframe(pd.DataFrame(xai["risk_factors"]), use_container_width=True)
    st.markdown("**Suggestions**")
    for rec in xai["recommendations"]:
        st.write(f"- {rec}")


def main() -> None:
    payload = load_report()
    if not payload:
        st.warning("Run `.venv312/bin/python peerread_review_project/generate_peerread_reports.py` first.")
        return
    model = load_peerread_model()
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Paper Table", "Dataset/Evaluation", "Poster Figures", "Single Paper Review"],
    )
    if page == "Overview":
        render_overview(payload, model)
    elif page == "Paper Table":
        render_table(payload)
    elif page == "Dataset/Evaluation":
        render_evaluation()
    elif page == "Poster Figures":
        render_figures()
    else:
        render_single_review(model)


if __name__ == "__main__":
    main()
