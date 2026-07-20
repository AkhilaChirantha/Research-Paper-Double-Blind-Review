from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from research_review.io import read_json


def read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def file_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def load_counts() -> dict:
    cleaning_report = {}
    clean_papers_path = Path("data/processed/clean_papers.json")
    if clean_papers_path.exists():
        cleaning_report = read_json(clean_papers_path).get("report", {}).get("counts", {})

    decisions = {}
    paper_decisions_path = Path("reports/paper_decisions.json")
    if paper_decisions_path.exists():
        paper_decisions = read_json(paper_decisions_path)
        decisions = paper_decisions.get("counts", {})

    advanced = {}
    advanced_path = Path("reports/advanced_ai_reviews.json")
    if advanced_path.exists():
        payload = read_json(advanced_path)
        advanced = {
            "model": payload.get("model"),
            "paper_count": len(payload.get("papers", [])),
            "groups": dict(Counter(paper.get("group", "Unknown") for paper in payload.get("papers", []))),
        }

    return {
        "cleaning": cleaning_report,
        "paper_decisions": decisions,
        "sft_total": read_jsonl_count(Path("data/sft/sft_double_blind_reviews.jsonl")),
        "sft_train": read_jsonl_count(Path("data/sft/train.jsonl")),
        "sft_validation": read_jsonl_count(Path("data/sft/validation.jsonl")),
        "section_summaries": read_jsonl_count(Path("data/processed/section_summaries.jsonl")),
        "advanced_ai": advanced,
    }


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def render_html(counts: dict) -> str:
    cleaning = counts["cleaning"]
    decisions = counts["paper_decisions"]
    advanced = counts["advanced_ai"]

    metrics = [
        ("Cleaned papers", f"{cleaning.get('cleaned_papers', 0):,}"),
        ("SFT examples", f"{counts['sft_total']:,}"),
        ("Section summaries", f"{counts['section_summaries']:,}"),
        ("AI reviewed papers", f"{advanced.get('paper_count', 0):,}"),
    ]
    metric_cards = "".join(
        f'<div class="metric"><strong>{value}</strong><span>{html.escape(label)}</span></div>'
        for label, value in metrics
    )

    cleaning_rows = [
        ["Metadata papers", f"{cleaning.get('metadata_papers', 0):,}"],
        ["Missing markdown files", f"{cleaning.get('missing_markdown', 0):,}"],
        ["No real reviewer review", f"{cleaning.get('no_real_review', 0):,}"],
        ["Cleaned papers kept", f"{cleaning.get('cleaned_papers', 0):,}"],
        ["Good paper derived", f"{cleaning.get('derived_good_paper', 0):,}"],
        ["Needs modification derived", f"{cleaning.get('derived_needs_modification', 0):,}"],
        ["Reject-risk derived", f"{cleaning.get('derived_reject_risk', 0):,}"],
    ]
    decision_rows = [[key, f"{value:,}"] for key, value in decisions.items()]
    sft_rows = [
        ["Total examples", f"{counts['sft_total']:,}"],
        ["Training examples", f"{counts['sft_train']:,}"],
        ["Validation examples", f"{counts['sft_validation']:,}"],
    ]
    file_rows = [
        ["Clean papers", "data/processed/clean_papers.json", file_size(Path("data/processed/clean_papers.json"))],
        ["SFT train", "data/sft/train.jsonl", file_size(Path("data/sft/train.jsonl"))],
        ["Paper decisions", "reports/paper_decisions.html", file_size(Path("reports/paper_decisions.html"))],
        ["Poster figures", "reports/poster_figures/", "folder"],
        ["System architecture", "reports/poster_figures/SYSTEM_ARCHITECTURE.svg", file_size(Path("reports/poster_figures/SYSTEM_ARCHITECTURE.svg"))],
        ["Thesis methodology", "THESIS_METHODOLOGY_SECTION.md", file_size(Path("THESIS_METHODOLOGY_SECTION.md"))],
        ["Presentation outline", "PRESENTATION_OUTLINE.md", file_size(Path("PRESENTATION_OUTLINE.md"))],
        ["Confidentiality guide", "CONFIDENTIALITY_GUIDE.md", file_size(Path("CONFIDENTIALITY_GUIDE.md"))],
    ]

    css = """
body { margin: 0; background: #f6f7f9; color: #111827; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }
main { max-width: 1160px; margin: 0 auto; padding: 34px 22px 60px; }
h1 { font-size: 38px; margin: 0 0 8px; }
h2 { margin-top: 34px; font-size: 24px; }
.muted { color: #667085; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin: 18px 0; }
.metric, .panel { background: #fff; border: 1px solid #d0d5dd; border-radius: 10px; padding: 16px; }
.metric strong { display: block; font-size: 31px; margin-bottom: 4px; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; border-bottom: 1px solid #eaecf0; padding: 10px 12px; }
th { background: #eef2f6; }
li { margin: 7px 0; line-height: 1.45; }
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Final Evaluation Report</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Final Evaluation Report</h1>
  <p class="muted">Research Review prototype aligned with the proposal: dataset cleaning, double-blind preparation, SFT dataset, local screening, confidentiality modes, and AI review support.</p>
  <section class="grid">{metric_cards}</section>

  <h2>Dataset Cleaning</h2>
  <div class="panel">{table(["Metric", "Value"], cleaning_rows)}</div>

  <h2>Local Screening Results</h2>
  <div class="panel">{table(["Decision", "Papers"], decision_rows)}</div>

  <h2>SFT Dataset</h2>
  <div class="panel">{table(["Split", "Examples"], sft_rows)}</div>

  <h2>System Capabilities</h2>
  <div class="panel"><ul>
    <li>Cleaned OpenReview-style dataset by removing unusable/non-review rows.</li>
    <li>Prepared double-blind paper/review data with identity masking.</li>
    <li>Created SFT-ready chat JSONL for fine-tuning a double-blind reviewer model.</li>
    <li>Implemented local Accept/Modify/Reject-risk screening.</li>
    <li>Added OpenAI-assisted explanation and advanced AI review scripts.</li>
    <li>Added confidentiality modes: local only, abstract only, section summary only, and full paper with consent.</li>
    <li>Generated poster-ready charts and supervisor-facing documentation.</li>
  </ul></div>

  <h2>Core Outputs</h2>
  <div class="panel">{table(["Output", "Path", "Size"], file_rows)}</div>

  <h2>Limitations</h2>
  <div class="panel"><ul>
    <li>The current dataset contains accepted papers only; reject labels are derived as risk labels from reviewer scores.</li>
    <li>Many markdown files contain abstract/review text rather than full manuscript sections.</li>
    <li>Fine-tuning pipeline is ready, but full LoRA training may require GPU resources.</li>
    <li>OpenAI review is optional and must respect confidentiality mode selection.</li>
  </ul></div>
</main>
</body>
</html>
"""


def render_markdown(counts: dict) -> str:
    cleaning = counts["cleaning"]
    decisions = counts["paper_decisions"]
    lines = [
        "# Final Evaluation Summary",
        "",
        "## Completed Work",
        "",
        "- Dataset cleaning and review filtering",
        "- Double-blind data preparation",
        "- SFT dataset creation",
        "- VS Code LoRA fine-tuning setup and dry-run validation",
        "- Section summary screening pipeline",
        "- Confidentiality modes",
        "- Poster figures and supervisor documentation",
        "",
        "## Key Counts",
        "",
        f"- Cleaned papers: {cleaning.get('cleaned_papers', 0):,}",
        f"- Good paper derived: {cleaning.get('derived_good_paper', 0):,}",
        f"- Needs modification derived: {cleaning.get('derived_needs_modification', 0):,}",
        f"- Reject-risk derived: {cleaning.get('derived_reject_risk', 0):,}",
        f"- SFT examples: {counts['sft_total']:,}",
        f"- Section summaries: {counts['section_summaries']:,}",
        "",
        "## Paper Decision Counts",
        "",
    ]
    for key, value in decisions.items():
        lines.append(f"- {key}: {value:,}")
    lines.extend(
        [
            "",
            "## Thesis Framing",
            "",
            "This project should be framed as an AI-assisted double-blind paper screening framework that estimates acceptance readiness and rejection risk using reviewer-derived signals, while generating interpretable feedback through an LLM-assisted explanation layer.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_presentation_points(counts: dict) -> str:
    cleaning = counts["cleaning"]
    return "\n".join(
        [
            "# Presentation Points",
            "",
            "1. Problem: peer review is overloaded, inconsistent, and difficult for authors to anticipate.",
            "2. Goal: build a pre-submission AI screening framework for research papers.",
            f"3. Dataset: {cleaning.get('cleaned_papers', 0):,} cleaned papers with real reviewer evidence.",
            "4. Method: local reviewer-derived risk model plus LLM-based feedback generation.",
            f"5. SFT dataset: {counts['sft_total']:,} double-blind chat-format examples.",
            "6. Confidentiality: local-only, abstract-only, section-summary-only, and full-paper-with-consent modes.",
            "7. Output: Accept / Modify / Reject-risk decision, strengths, weaknesses, and modification plan.",
            "8. Limitation: current dataset has accepted papers only, so reject is treated as reviewer-derived risk.",
            "9. Future work: add true rejected papers and complete PEFT fine-tuning on GPU.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final evaluation report files.")
    parser.add_argument("--html-output", default="reports/final_evaluation_report.html")
    parser.add_argument("--md-output", default="reports/final_evaluation_summary.md")
    parser.add_argument("--presentation-output", default="reports/final_presentation_points.md")
    args = parser.parse_args()

    counts = load_counts()
    outputs = [
        (Path(args.html_output), render_html(counts)),
        (Path(args.md_output), render_markdown(counts)),
        (Path(args.presentation_output), render_presentation_points(counts)),
    ]
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
