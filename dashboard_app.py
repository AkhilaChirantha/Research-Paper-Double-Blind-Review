from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from research_review.config import DEFAULT_MODEL_PATH, openai_model
from research_review.confidentiality import ConfidentialityMode, parse_mode, prepare_review_text
from research_review.io import read_document
from research_review.model import load_model, predict_with_model, training_summary
from research_review.openai_reviewer import get_openai_recommendation


ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
PAPER_DECISIONS_PATH = REPORTS_DIR / "paper_decisions.json"
ADVANCED_AI_PATH = REPORTS_DIR / "advanced_ai_reviews.json"
FIGURES_DIR = REPORTS_DIR / "poster_figures"


st.set_page_config(
    page_title="Research Review Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_review_model() -> dict:
    return load_model(ROOT / DEFAULT_MODEL_PATH)


def decision_dataframe() -> pd.DataFrame:
    payload = load_json(str(PAPER_DECISIONS_PATH))
    rows = payload.get("papers", [])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def advanced_reviews_dataframe() -> pd.DataFrame:
    payload = load_json(str(ADVANCED_AI_PATH))
    rows = []
    for item in payload.get("papers", []):
        ai = item.get("ai_review") or {}
        rows.append(
            {
                "paper_id": item.get("paper_id"),
                "title": item.get("title"),
                "local_decision": item.get("predicted_decision"),
                "ai_decision": ai.get("ai_decision") or ai.get("final_verdict"),
                "confidence": ai.get("confidence"),
                "group": item.get("group"),
                "summary": ai.get("short_summary") or ai.get("overall_summary"),
            }
        )
    return pd.DataFrame(rows)


def metric_card(label: str, value: object, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def render_overview() -> None:
    st.title("Double-Blind Research Review Dashboard")
    st.caption("Local ML screening + optional OpenAI feedback with confidentiality controls.")

    model = load_review_model()
    decisions = load_json(str(PAPER_DECISIONS_PATH))
    counts = decisions.get("counts", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Training Papers", model.get("training_count", 0))
    with col2:
        metric_card("Accept Candidates", counts.get("Accept", 0))
    with col3:
        metric_card("Modify Candidates", counts.get("Modify", 0))
    with col4:
        metric_card("Reject-Risk Candidates", counts.get("Reject", 0))

    st.info(
        "Current dataset has accepted OpenReview papers only. Reject and modify outputs are "
        "derived risk-screening labels from review scores and paper structure, not true final labels."
    )

    df = decision_dataframe()
    if not df.empty:
        st.subheader("Decision Distribution")
        chart_df = (
            df["predicted_decision"]
            .value_counts()
            .rename_axis("decision")
            .reset_index(name="papers")
            .sort_values("decision")
        )
        st.bar_chart(chart_df, x="decision", y="papers", height=320)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Quality Score Spread")
            bins = pd.cut(df["quality_score"], bins=[0, 25, 50, 75, 100], include_lowest=True)
            score_df = bins.value_counts().sort_index().rename_axis("score_range").reset_index(name="papers")
            score_df["score_range"] = score_df["score_range"].astype(str)
            st.bar_chart(score_df, x="score_range", y="papers", height=280)
        with col2:
            st.subheader("Model Summary")
            st.write(training_summary(model))
            st.write("OpenAI model:", f"`{openai_model()}`")
            st.write("Generated reports are available in `reports/`.")

    st.subheader("Poster Figures")
    figure_paths = [
        FIGURES_DIR / "01_local_decision_distribution.svg",
        FIGURES_DIR / "02_quality_score_distribution.svg",
        FIGURES_DIR / "03_decision_share.svg",
        FIGURES_DIR / "SYSTEM_ARCHITECTURE.svg",
    ]
    existing = [path for path in figure_paths if path.exists()]
    tabs = st.tabs([path.stem for path in existing])
    for tab, path in zip(tabs, existing):
        with tab:
            st.image(str(path), use_container_width=True)


def render_paper_table() -> None:
    st.title("Paper-by-Paper Screening Table")
    df = decision_dataframe()
    if df.empty:
        st.warning("Run `python paper_decisions.py` first to generate paper decisions.")
        return

    decisions = sorted(df["predicted_decision"].dropna().unique())
    selected = st.multiselect("Decision filter", decisions, default=decisions)
    min_score, max_score = st.slider("Quality score range", 0.0, 100.0, (0.0, 100.0), 1.0)
    query = st.text_input("Search by paper id or title")

    filtered = df[df["predicted_decision"].isin(selected)]
    filtered = filtered[(filtered["quality_score"] >= min_score) & (filtered["quality_score"] <= max_score)]
    if query:
        needle = query.lower()
        filtered = filtered[
            filtered["paper_id"].str.lower().str.contains(needle, na=False)
            | filtered["title"].str.lower().str.contains(needle, na=False)
        ]

    st.caption(f"Showing {len(filtered)} of {len(df)} papers.")
    columns = [
        "paper_id",
        "title",
        "actual_decision",
        "predicted_decision",
        "quality_score",
        "accept_probability",
        "modify_probability",
        "reject_probability",
        "suggestions",
    ]
    st.dataframe(filtered[columns], use_container_width=True, height=560)
    st.download_button(
        "Download Filtered CSV",
        filtered[columns].to_csv(index=False).encode("utf-8"),
        file_name="filtered_paper_decisions.csv",
        mime="text/csv",
    )


def render_openai_comparison() -> None:
    st.title("OpenAI Feedback Comparison")
    payload = load_json(str(ADVANCED_AI_PATH))
    df = advanced_reviews_dataframe()
    if df.empty:
        st.warning("Run `python top_papers_openai.py --confidentiality-mode section_summary_only` first.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("OpenAI Model", payload.get("model", "unknown"))
    with col2:
        metric_card("Papers Reviewed", len(df))
    with col3:
        metric_card("Per Group", payload.get("per_group", "unknown"))

    st.dataframe(df, use_container_width=True, height=360)

    selected_id = st.selectbox("Open detailed AI review", df["paper_id"].tolist())
    item = next((row for row in payload.get("papers", []) if row.get("paper_id") == selected_id), None)
    if not item:
        return
    ai = item.get("ai_review") or {}
    st.subheader(item.get("title", selected_id))
    st.write("Local:", item.get("predicted_decision"), "| AI:", ai.get("ai_decision") or ai.get("final_verdict"))
    st.write(ai.get("short_summary") or ai.get("overall_summary") or "")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Good Points**")
        for point in ai.get("good_points", []) or ai.get("main_reasons", []):
            st.write(f"- {point}")
    with col2:
        st.markdown("**Weak / Modify Points**")
        for point in ai.get("weak_points", []):
            st.write(f"- {point}")

    st.markdown("**Must Modify**")
    for change in ai.get("must_modify", []) or ai.get("section_level_suggestions", []):
        section = change.get("section", "Section")
        problem = change.get("problem") or change.get("issue", "")
        suggestion = change.get("suggestion") or change.get("recommendation", "")
        priority = change.get("priority", "medium")
        st.write(f"- **{section}** ({priority}): {problem} {suggestion}")


def render_single_review() -> None:
    st.title("Review a New Paper")
    st.caption("Default mode is local-only. OpenAI is optional and controlled below.")

    uploaded = st.file_uploader("Upload paper", type=["md", "txt", "tex", "pdf"])
    use_openai = st.toggle("Use OpenAI for detailed suggestions", value=False)
    mode_value = st.selectbox(
        "Confidentiality mode",
        [mode.value for mode in ConfidentialityMode],
        index=0,
        help="OpenAI is blocked when local_only is selected.",
    )

    if not uploaded:
        return

    suffix = Path(uploaded.name).suffix or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded.getvalue())
        temp_path = Path(temp_file.name)

    try:
        text = read_document(temp_path)
        mode = parse_mode(mode_value)
        review_text, audit = prepare_review_text(text, uploaded.name, mode)
        model = load_review_model()
        prediction = predict_with_model(model, review_text)
    except Exception as exc:
        st.error(f"Could not review paper: {exc}")
        return
    finally:
        temp_path.unlink(missing_ok=True)

    st.subheader("Local Screening Result")
    col1, col2 = st.columns(2)
    with col1:
        metric_card("Verdict", prediction["verdict"])
    with col2:
        metric_card("Quality Score", f"{prediction['quality_score']}/100")

    probabilities = pd.DataFrame(
        [{"label": key, "probability": value} for key, value in prediction["probabilities"].items()]
    )
    st.bar_chart(probabilities, x="label", y="probability", height=260)

    st.markdown("**Structural Gaps**")
    gaps = prediction.get("feature_gaps") or ["No major structural gaps detected by the local model."]
    for gap in gaps:
        st.write(f"- {gap}")

    with st.expander("Confidentiality Audit", expanded=False):
        st.json(audit)

    if use_openai:
        if not audit.get("api_allowed"):
            st.error("OpenAI review is blocked in local_only mode. Select another mode if you have consent.")
            return
        with st.spinner("Calling OpenAI for detailed blind-review suggestions..."):
            try:
                ai_review = get_openai_recommendation(review_text, prediction)
            except Exception as exc:
                st.error(f"OpenAI review failed: {exc}")
                return

        st.subheader("OpenAI Blind Review")
        st.write("Final verdict:", ai_review.get("final_verdict"))
        st.write("Confidence:", ai_review.get("confidence"))
        st.write(ai_review.get("overall_summary", ""))

        st.markdown("**Section-Level Suggestions**")
        for item in ai_review.get("section_level_suggestions", []):
            st.write(
                f"- **{item.get('section')}** ({item.get('priority')}): "
                f"{item.get('issue')} Recommendation: {item.get('recommendation')}"
            )

        st.markdown("**Acceptance Plan**")
        for step in ai_review.get("acceptance_plan", []):
            st.write(f"- {step}")


def main() -> None:
    st.sidebar.title("Research Review")
    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Paper Table",
            "OpenAI Comparison",
            "Review New Paper",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Prototype for double-blind pre-submission research screening.")

    if page == "Overview":
        render_overview()
    elif page == "Paper Table":
        render_paper_table()
    elif page == "OpenAI Comparison":
        render_openai_comparison()
    else:
        render_single_review()


if __name__ == "__main__":
    main()
