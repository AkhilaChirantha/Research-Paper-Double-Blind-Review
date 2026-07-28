from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from peerread_review.config import DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR
from peerread_review.model import load_model


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


def main() -> None:
    st.title("PeerRead Supervised Paper Review Dashboard")
    st.caption("Separate project using true PeerRead Accept/Reject labels. Default suggestions are XAI-based.")
    payload = load_report()
    if not payload:
        st.warning("Run `.venv312/bin/python peerread_review_project/generate_peerread_reports.py` first.")
        return

    model = load_peerread_model()
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

    st.subheader("Files")
    st.write("Model:", Path(DEFAULT_MODEL_PATH))
    st.write("Reports:", Path(DEFAULT_REPORT_DIR))


if __name__ == "__main__":
    main()
