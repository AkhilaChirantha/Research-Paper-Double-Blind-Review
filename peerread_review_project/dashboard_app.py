from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peerread_review.agent import local_prediction_for_openai, review_new_paper
from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR
from peerread_review.data import read_peerread_rows
from peerread_review.model import load_model, predict
from peerread_review.xai import explain
from research_review.confidentiality import ConfidentialityMode, parse_mode, prepare_review_text
from research_review.io import read_document
from research_review.openai_reviewer import get_openai_recommendation


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
def load_openai_reviews() -> dict:
    path = DEFAULT_REPORT_DIR / "peerread_openai_reviews.json"
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
    st.dataframe(filtered[columns], width="stretch", height=560)

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


def render_advanced_metrics(payload: dict) -> None:
    st.title("Advanced Thesis Metrics")
    training = load_training_summary()
    if not training:
        st.warning("Run `.venv312/bin/python peerread_review_project/train_peerread_model.py` first.")
        return

    metrics = []
    for split_name, key in [("Dev", "dev_evaluation"), ("Test", "test_evaluation")]:
        eval_row = training.get(key, {})
        metrics.extend(
            [
                {"split": split_name, "metric": "Accuracy", "value": eval_row.get("accuracy", 0)},
                {"split": split_name, "metric": "Precision", "value": eval_row.get("accept_precision", 0)},
                {"split": split_name, "metric": "Recall", "value": eval_row.get("accept_recall", 0)},
                {"split": split_name, "metric": "F1 Score", "value": eval_row.get("accept_f1", 0)},
            ]
        )
    metric_df = pd.DataFrame(metrics)
    st.subheader("Classification Metrics")
    st.dataframe(metric_df, width="stretch", hide_index=True)
    st.bar_chart(metric_df, x="metric", y="value", color="split", height=320)

    test_confusion = training.get("test_evaluation", {}).get("confusion", {})
    heatmap_df = pd.DataFrame(
        [
            [test_confusion.get("reject_as_reject", 0), test_confusion.get("reject_as_accept", 0)],
            [test_confusion.get("accept_as_reject", 0), test_confusion.get("accept_as_accept", 0)],
        ],
        index=["Actual Reject", "Actual Accept"],
        columns=["Predicted Reject", "Predicted Accept"],
    )
    st.subheader("Test Confusion Matrix")
    render_confusion_heatmap(heatmap_df)

    rows = payload.get("papers", [])
    if rows:
        df = pd.DataFrame(rows)
        df["probability_bin"] = pd.cut(
            df["accept_probability"].astype(float),
            bins=[i / 10 for i in range(11)],
            include_lowest=True,
        ).astype(str)
        prob_df = df.groupby(["probability_bin", "actual_label"], observed=False).size().reset_index(name="papers")
        st.subheader("Accept Probability Distribution by Actual Label")
        st.bar_chart(prob_df, x="probability_bin", y="papers", color="actual_label", height=330)

        calibration = (
            df.groupby("probability_bin", observed=False)
            .agg(
                average_probability=("accept_probability", "mean"),
                observed_accept_rate=("actual_label", lambda values: (values == "Accept").mean()),
                papers=("paper_id", "count"),
            )
            .reset_index()
        )
        st.subheader("Regression-Style Calibration Table")
        st.caption("This checks whether predicted accept probability behaves like a calibrated numeric prediction.")
        st.dataframe(calibration, width="stretch", hide_index=True)

    figure_dir = DEFAULT_REPORT_DIR / "poster_figures"
    figures = [
        "03_classification_metrics.svg",
        "04_test_confusion_heatmap.svg",
        "05_probability_distribution.svg",
        "06_feature_importance.svg",
        "07_probability_calibration.svg",
    ]
    existing = [figure_dir / name for name in figures if (figure_dir / name).exists()]
    if existing:
        st.subheader("Poster/Thesis Figures")
        tabs = st.tabs([path.stem for path in existing])
        for tab, path in zip(tabs, existing):
            with tab:
                st.image(str(path), width="stretch")


def render_confusion_heatmap(matrix: pd.DataFrame) -> None:
    max_value = max([int(value) for value in matrix.to_numpy().flatten()] + [1])
    rows = []
    for actual_label, values in matrix.iterrows():
        cells = [f"<th style='padding: 10px; color: #344054; text-align: right;'>{actual_label}</th>"]
        for predicted_label, value in values.items():
            intensity = int(245 - (int(value) / max_value) * 120)
            cells.append(
                "<td style='"
                f"background: rgb({intensity}, {min(intensity + 20, 255)}, 255);"
                "border: 1px solid #d0d5dd; padding: 22px; text-align: center;"
                "font-size: 24px; font-weight: 800; color: #111827;"
                f"'><div>{int(value)}</div><small style='font-size: 13px; font-weight: 600; color: #475467;'>{predicted_label}</small></td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")
    html = (
        "<table style='border-collapse: collapse; width: 100%; max-width: 760px;'>"
        "<thead><tr><th></th>"
        + "".join(
            f"<th style='padding: 10px; color: #344054; text-align: center;'>{column}</th>"
            for column in matrix.columns
        )
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_xai_openai(payload: dict) -> None:
    st.title("Agentic AI")
    st.caption("Detailed Agentic AI paper-level reviews compared with the PeerRead local model and XAI suggestions.")
    rows = payload.get("papers", [])
    if rows:
        df = pd.DataFrame(rows)
        xai_columns = [
            "paper_id",
            "title",
            "actual_label",
            "predicted_decision",
            "accept_probability",
            "xai_focus",
            "suggestion_1",
            "suggestion_2",
            "suggestion_3",
        ]
        st.subheader("Local XAI Suggestions")
        st.dataframe(df[xai_columns], width="stretch", height=360)

    openai_payload = load_openai_reviews()
    if not openai_payload:
        st.subheader("Optional Agentic AI Reviews")
        st.info(
            "Agentic AI comparison file is not generated yet. Run "
            "`.venv312/bin/python peerread_review_project/top_peerread_openai.py --per-group 5 --confidentiality-mode section_summary_only` "
            "to generate the detailed Agentic AI review report."
        )
        return

    ai_rows = openai_payload.get("papers", [])
    comparison_rows = []
    for item in ai_rows:
        ai_review = item.get("ai_review") or {}
        comparison_rows.append(
            {
                "group": item.get("group"),
                "paper_id": item.get("paper_id"),
                "title": item.get("title"),
                "actual_label": item.get("actual_label"),
                "local_decision": item.get("predicted_decision") or item.get("local_decision"),
                "accept_probability": item.get("accept_probability"),
                "openai_decision": ai_review.get("ai_decision") or ai_review.get("verdict") or ai_review.get("final_verdict"),
                "openai_confidence": ai_review.get("confidence"),
                "openai_summary": ai_review.get("short_summary") or ai_review.get("summary") or ai_review.get("overall_summary"),
            }
        )
    st.subheader("XAI vs Agentic AI")
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, width="stretch", height=360)

    if comparison_df.empty:
        return
    selected_id = st.selectbox("Open detailed Agentic AI review", comparison_df["paper_id"].astype(str).tolist())
    item = next((row for row in ai_rows if str(row.get("paper_id")) == str(selected_id)), None)
    if not item:
        return
    ai = item.get("ai_review") or {}
    st.subheader(item.get("title", selected_id))
    cols = st.columns(4)
    cols[0].metric("PeerRead Label", item.get("actual_label", "unknown"))
    cols[1].metric("Local Decision", item.get("predicted_decision") or item.get("local_decision", "unknown"))
    cols[2].metric("Agentic AI Decision", ai.get("ai_decision") or ai.get("final_verdict") or "unknown")
    cols[3].metric("Agentic AI Confidence", ai.get("confidence", "unknown"))
    st.write(ai.get("short_summary") or ai.get("overall_summary") or "")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Good Points**")
        for point in ai.get("good_points", []) or ai.get("main_reasons", []):
            st.write(f"- {point}")
    with col2:
        st.markdown("**Weak Points**")
        for point in ai.get("weak_points", []):
            st.write(f"- {point}")

    st.markdown("**Must Modify**")
    for change in ai.get("must_modify", []) or ai.get("section_level_suggestions", []):
        section = change.get("section", "Section")
        problem = change.get("problem") or change.get("issue", "")
        suggestion = change.get("suggestion") or change.get("recommendation", "")
        priority = change.get("priority", "medium")
        st.write(f"- **{section}** ({priority}): {problem} Recommendation: {suggestion}")

    st.markdown("**Acceptance Plan**")
    for step in ai.get("acceptance_plan", []):
        st.write(f"- {step}")

    note = ai.get("supervisor_note")
    if note:
        st.markdown("**Supervisor Note**")
        st.info(note)


def render_figures() -> None:
    st.title("PeerRead Poster Figures")
    figure_dir = DEFAULT_REPORT_DIR / "poster_figures"
    figures = [
        figure_dir / "01_peerread_predicted_decisions.svg",
        figure_dir / "02_peerread_actual_labels.svg",
        figure_dir / "03_classification_metrics.svg",
        figure_dir / "04_test_confusion_heatmap.svg",
        figure_dir / "05_probability_distribution.svg",
        figure_dir / "06_feature_importance.svg",
        figure_dir / "07_probability_calibration.svg",
        figure_dir / "SYSTEM_ARCHITECTURE.svg",
    ]
    existing = [path for path in figures if path.exists()]
    if not existing:
        st.warning("Run `peerread_review_project/poster_figures.py` first.")
        return
    tabs = st.tabs([path.stem for path in existing])
    for tab, path in zip(tabs, existing):
        with tab:
            st.image(str(path), width="stretch")


def render_single_review(model: dict) -> None:
    st.title("Review Existing PeerRead Paper")
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
    st.dataframe(pd.DataFrame(xai["risk_factors"]), width="stretch")
    st.markdown("**Suggestions**")
    for rec in xai["recommendations"]:
        st.write(f"- {rec}")


def render_new_paper_agent(model: dict) -> None:
    st.title("AI Agent: Review a New Paper")
    st.caption("PeerRead-trained local model + XAI suggestions by default. Agentic AI is optional for deeper edit feedback.")

    source = st.radio("Paper input", ["Upload file", "Paste text"], horizontal=True)
    uploaded = None
    pasted_text = ""
    paper_name = "new_paper.txt"
    if source == "Upload file":
        uploaded = st.file_uploader("Upload paper", type=["md", "txt", "tex", "pdf"])
        if uploaded:
            paper_name = uploaded.name
    else:
        paper_name = st.text_input("Paper title or file name", value="new_paper.txt")
        pasted_text = st.text_area("Paste paper text", height=320)

    review_mode = st.radio(
        "Review model",
        ["XAI Local Review", "XAI + Agentic AI Detailed Review"],
        index=0,
        horizontal=True,
        help="XAI Local Review is the default. Agentic AI uses API credits only when selected.",
    )
    use_openai = review_mode == "XAI + Agentic AI Detailed Review"
    mode_value = st.selectbox(
        "Confidentiality mode",
        [mode.value for mode in ConfidentialityMode],
        index=0,
        help="Agentic AI is blocked in local_only mode. Use abstract/section summary/full paper only with consent.",
    )

    run_review = st.button("Run AI Review", type="primary")
    if not run_review:
        return

    try:
        if uploaded:
            text = read_uploaded_document(uploaded)
            paper_name = uploaded.name
        else:
            text = pasted_text.strip()
        if not text:
            st.warning("Add a paper file or paste paper text first.")
            return
        mode = parse_mode(mode_value)
        review_text, audit = prepare_review_text(text, paper_name, mode)
        result = review_new_paper(review_text, paper_name, model)
    except Exception as exc:
        st.error(f"Could not review paper: {exc}")
        return

    prediction = result["prediction"]
    agent = result["agent_review"]
    xai = result["xai"]

    st.subheader(result["title"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", agent["decision"])
    col2.metric("Quality Score", f"{agent['quality_score']}/100")
    col3.metric("Accept Probability", prediction["accept_probability"])

    probability_df = pd.DataFrame(
        [
            {"label": "Accept", "probability": prediction["accept_probability"]},
            {"label": "Modify", "probability": agent["probabilities"]["modify"]},
            {"label": "Reject", "probability": prediction["reject_probability"]},
        ]
    )
    st.bar_chart(probability_df, x="label", y="probability", height=260)

    st.markdown("**Overall AI Agent Summary**")
    st.write(agent["overall_summary"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Good Points**")
        for point in agent["good_points"]:
            st.write(f"- {point}")
    with col2:
        st.markdown("**Weak Points**")
        for point in agent["weak_points"]:
            st.write(f"- {point}")

    st.markdown("**Must Modify Before Submission**")
    for point in agent["must_modify"]:
        st.write(f"- {point}")

    st.markdown("**Plan to Reach Accept Level**")
    for step in agent["acceptance_plan"]:
        st.write(f"- {step}")

    st.subheader("XAI Evidence")
    factors = pd.DataFrame(xai["risk_factors"])
    if not factors.empty:
        st.dataframe(
            factors[["label", "value", "contribution", "direction", "recommendation"]],
            width="stretch",
            height=300,
        )

    export_result = {
        "title": result["title"],
        "prediction": export_prediction(prediction),
        "agent_review": agent,
        "xai": xai,
        "confidentiality_audit": audit,
    }
    st.download_button(
        "Download AI Agent Review JSON",
        json.dumps(export_result, indent=2, ensure_ascii=False).encode("utf-8"),
        file_name="peerread_new_paper_ai_review.json",
        mime="application/json",
    )

    with st.expander("Confidentiality Audit", expanded=False):
        st.json(audit)

    if use_openai:
        if not audit.get("api_allowed"):
            st.error("Agentic AI review is blocked in local_only mode. Select abstract_only, section_summary_only, or full_paper_with_consent.")
            return
        with st.spinner("Calling Agentic AI for detailed blind-review suggestions..."):
            try:
                ai_review = get_openai_recommendation(review_text, local_prediction_for_openai(prediction))
            except Exception as exc:
                st.error(f"Agentic AI review failed: {exc}")
                return
        render_openai_agent_review(ai_review)


def read_uploaded_document(uploaded) -> str:
    suffix = Path(uploaded.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded.getvalue())
        temp_path = Path(temp_file.name)
    try:
        return read_document(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def export_prediction(prediction: dict) -> dict:
    hidden = {"text", "scaled_features"}
    return {key: value for key, value in prediction.items() if key not in hidden}


def render_openai_agent_review(ai_review: dict) -> None:
    st.subheader("Agentic AI Detailed Blind Review")
    col1, col2 = st.columns(2)
    col1.metric("Agentic AI Verdict", ai_review.get("final_verdict", "unknown"))
    col2.metric("Agentic AI Confidence", ai_review.get("confidence", "unknown"))
    st.write(ai_review.get("overall_summary", ""))

    st.markdown("**Main Reasons**")
    for reason in ai_review.get("main_reasons", []):
        st.write(f"- {reason}")

    st.markdown("**Section-Level Edit Suggestions**")
    for item in ai_review.get("section_level_suggestions", []):
        st.write(
            f"- **{item.get('section')}** ({item.get('priority')}): "
            f"{item.get('issue')} Recommendation: {item.get('recommendation')}"
        )

    st.markdown("**Acceptance Plan**")
    for step in ai_review.get("acceptance_plan", []):
        st.write(f"- {step}")

    questions = ai_review.get("reviewer_questions", [])
    if questions:
        st.markdown("**Reviewer Questions to Answer**")
        for question in questions:
            st.write(f"- {question}")


def main() -> None:
    payload = load_report()
    if not payload:
        st.warning("Run `.venv312/bin/python peerread_review_project/generate_peerread_reports.py` first.")
        return
    model = load_peerread_model()
    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Paper Table",
            "Agentic AI",
            "Dataset/Evaluation",
            "Advanced Metrics",
            "Poster Figures",
            "AI Agent New Paper Review",
            "Single Paper Review",
        ],
    )
    if page == "Overview":
        render_overview(payload, model)
    elif page == "Paper Table":
        render_table(payload)
    elif page == "Agentic AI":
        render_xai_openai(payload)
    elif page == "Dataset/Evaluation":
        render_evaluation()
    elif page == "Advanced Metrics":
        render_advanced_metrics(payload)
    elif page == "Poster Figures":
        render_figures()
    elif page == "AI Agent New Paper Review":
        render_new_paper_agent(model)
    else:
        render_single_review(model)


if __name__ == "__main__":
    main()
