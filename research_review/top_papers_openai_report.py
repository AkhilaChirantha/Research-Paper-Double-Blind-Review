from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from openai import OpenAI

from research_review.confidentiality import ConfidentialityMode, parse_mode, prepare_review_text
from research_review.config import DEFAULT_PAPERS_DIR, load_env, openai_model
from research_review.io import read_document, read_json, write_json
from research_review.openai_reviewer import trim_for_review


TOP_REVIEW_SCHEMA = {
    "type": "json_schema",
    "name": "paper_level_ai_review",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ai_decision": {"type": "string", "enum": ["ACCEPT", "MODIFY", "REJECT"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "short_summary": {"type": "string"},
            "good_points": {"type": "array", "items": {"type": "string"}},
            "weak_points": {"type": "array", "items": {"type": "string"}},
            "must_modify": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "section": {"type": "string"},
                        "problem": {"type": "string"},
                        "suggestion": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["section", "problem", "suggestion", "priority"],
                },
            },
            "acceptance_plan": {"type": "array", "items": {"type": "string"}},
            "supervisor_note": {"type": "string"},
        },
        "required": [
            "ai_decision",
            "confidence",
            "short_summary",
            "good_points",
            "weak_points",
            "must_modify",
            "acceptance_plan",
            "supervisor_note",
        ],
    },
}


def select_top_papers(rows: list[dict], per_group: int) -> list[dict]:
    accept = sorted(
        [row for row in rows if row["predicted_decision"] == "Accept"],
        key=lambda row: float(row["quality_score"]),
        reverse=True,
    )[:per_group]
    modify = sorted(
        [row for row in rows if row["predicted_decision"] == "Modify"],
        key=lambda row: float(row["quality_score"]),
        reverse=True,
    )[:per_group]
    reject = sorted(
        [row for row in rows if row["predicted_decision"] == "Reject"],
        key=lambda row: float(row["reject_probability"]),
        reverse=True,
    )[:per_group]
    return accept + modify + reject


def group_name(local_decision: str) -> str:
    return {
        "Accept": "Top Accept Candidates",
        "Modify": "Top Modify Candidates",
        "Reject": "Top Reject-Risk Papers",
    }.get(local_decision, local_decision)


def call_openai_review(client: OpenAI, row: dict, paper_text: str, max_chars: int) -> dict:
    prompt = {
        "task": (
            "Review this research paper before submission. Explain clearly whether it is accept-ready, "
            "needs modification, or reject-risk. Give paper-specific strengths, weaknesses, and exact modifications."
        ),
        "local_model_output": {
            "decision": row["predicted_decision"],
            "quality_score": row["quality_score"],
            "accept_probability": row["accept_probability"],
            "modify_probability": row["modify_probability"],
            "reject_probability": row["reject_probability"],
            "local_suggestions": row["suggestions"],
        },
        "paper": {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "text": trim_for_review(paper_text, max_chars=max_chars),
        },
        "review_criteria": [
            "technical soundness",
            "novelty and contribution",
            "clarity and structure",
            "experimental evidence",
            "baseline comparison",
            "ablation and analysis",
            "limitations and ethics",
            "reproducibility",
        ],
    }
    response = client.responses.create(
        model=openai_model(),
        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict but constructive academic reviewer. "
                    "Do not give generic advice. Every suggestion must be specific to the paper."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        text={"format": TOP_REVIEW_SCHEMA},
    )
    return json.loads(response.output_text)


def build_report(
    local_json: Path,
    papers_dir: Path,
    per_group: int,
    max_chars: int,
    mode: ConfidentialityMode,
) -> dict:
    load_env(Path(".env"))
    client = OpenAI()
    local_rows = read_json(local_json)["papers"]
    selected = select_top_papers(local_rows, per_group)
    reviewed = []
    for index, row in enumerate(selected, start=1):
        paper_path = papers_dir / f"{row['paper_id']}.md"
        text = read_document(paper_path)
        text, audit = prepare_review_text(text, str(paper_path), mode)
        if not audit.get("api_allowed"):
            raise RuntimeError("Advanced AI review cannot run in local_only mode.")
        ai_review = call_openai_review(client, row, text, max_chars)
        reviewed.append({**row, "group": group_name(row["predicted_decision"]), "ai_review": ai_review, "confidentiality": audit})
        print(f"[{index}/{len(selected)}] {row['paper_id']} {row['predicted_decision']} -> {ai_review['ai_decision']}")
    return {
        "model": openai_model(),
        "per_group": per_group,
        "max_chars": max_chars,
        "confidentiality_mode": mode.value,
        "papers": reviewed,
    }


def join_list(items: list[str]) -> str:
    return " ".join(items)


def modifications_text(items: list[dict]) -> str:
    return " ".join(
        f"{item['section']} ({item['priority']}): {item['suggestion']}" for item in items
    )


def write_csv_report(payload: dict, output_path: Path) -> None:
    rows = []
    for item in payload["papers"]:
        review = item["ai_review"]
        rows.append(
            {
                "group": item["group"],
                "paper_id": item["paper_id"],
                "title": item["title"],
                "local_decision": item["predicted_decision"],
                "ai_decision": review["ai_decision"],
                "ai_confidence": review["confidence"],
                "quality_score": item["quality_score"],
                "good_points": join_list(review["good_points"]),
                "weak_points": join_list(review["weak_points"]),
                "must_modify": modifications_text(review["must_modify"]),
                "acceptance_plan": join_list(review["acceptance_plan"]),
                "supervisor_note": review["supervisor_note"],
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def pill(label: str) -> str:
    css = label.lower().replace("_", "-")
    return f'<span class="pill {css}">{html.escape(label)}</span>'


def write_html_report(payload: dict, output_path: Path) -> None:
    sections = []
    for group in ("Top Accept Candidates", "Top Modify Candidates", "Top Reject-Risk Papers"):
        cards = []
        for item in [paper for paper in payload["papers"] if paper["group"] == group]:
            review = item["ai_review"]
            mods = "".join(
                "<li>"
                f"<strong>{html.escape(mod['section'])}</strong> "
                f"({html.escape(mod['priority'])}): "
                f"{html.escape(mod['problem'])} "
                f"<em>{html.escape(mod['suggestion'])}</em>"
                "</li>"
                for mod in review["must_modify"]
            )
            good = "".join(f"<li>{html.escape(x)}</li>" for x in review["good_points"])
            weak = "".join(f"<li>{html.escape(x)}</li>" for x in review["weak_points"])
            plan = "".join(f"<li>{html.escape(x)}</li>" for x in review["acceptance_plan"])
            cards.append(
                f"""
                <article class="card">
                  <div class="card-head">
                    <div>
                      <h3>{html.escape(item['title'])}</h3>
                      <p><code>{html.escape(item['paper_id'])}</code></p>
                    </div>
                    <div class="decision-box">
                      <div>Local: {pill(item['predicted_decision'])}</div>
                      <div>AI: {pill(review['ai_decision'])}</div>
                      <div class="score">Score {html.escape(str(item['quality_score']))} | AI confidence {review['confidence']:.2f}</div>
                    </div>
                  </div>
                  <p class="summary">{html.escape(review['short_summary'])}</p>
                  <div class="cols">
                    <section><h4>Good Points</h4><ul>{good}</ul></section>
                    <section><h4>Weak Points</h4><ul>{weak}</ul></section>
                  </div>
                  <section><h4>Must Modify</h4><ul>{mods}</ul></section>
                  <section><h4>Acceptance Plan</h4><ol>{plan}</ol></section>
                  <p class="note"><strong>Supervisor note:</strong> {html.escape(review['supervisor_note'])}</p>
                </article>
                """
            )
        sections.append(f"<h2>{group}</h2>{''.join(cards)}")

    css = """
body { margin: 0; background: #f5f7fa; color: #111827; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding: 34px 22px 60px; }
h1 { font-size: 38px; margin: 0 0 8px; }
h2 { font-size: 27px; margin: 34px 0 14px; }
h3 { font-size: 20px; margin: 0 0 4px; }
h4 { font-size: 15px; margin: 12px 0 8px; color: #344054; text-transform: uppercase; letter-spacing: .04em; }
.muted { color: #667085; }
.card { background: white; border: 1px solid #d0d5dd; border-radius: 10px; padding: 18px; margin: 14px 0; box-shadow: 0 1px 2px rgba(16,24,40,.05); }
.card-head { display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; border-bottom: 1px solid #eaecf0; padding-bottom: 12px; }
.decision-box { min-width: 250px; line-height: 1.9; }
.score { color: #667085; font-size: 13px; }
.summary { font-size: 15px; line-height: 1.55; }
.cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; }
li { margin: 6px 0; line-height: 1.45; }
.pill { display: inline-block; min-width: 64px; text-align: center; border-radius: 999px; padding: 3px 9px; color: white; font-weight: 700; font-size: 12px; }
.accept, .good-paper { background: #157347; }
.modify, .needs-modification { background: #c77700; }
.reject, .reject-risk { background: #b42318; }
.note { background: #f8fafc; border-left: 4px solid #2477b3; padding: 10px 12px; }
"""
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Advanced AI Paper Review Report</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Advanced AI Paper Review Report</h1>
  <p class="muted">Best {payload['per_group']} papers from each local class were reviewed with OpenAI model <code>{html.escape(payload['model'])}</code>. Confidentiality mode: <code>{html.escape(str(payload.get('confidentiality_mode')))}</code>. Each paper includes paper-specific good points, weak points, and required modifications.</p>
  {''.join(sections)}
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OpenAI reviews for best Accept/Modify/Reject papers.")
    parser.add_argument("--local-json", default="reports/paper_decisions.json")
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=14000)
    parser.add_argument(
        "--confidentiality-mode",
        default="section_summary_only",
        choices=["abstract_only", "section_summary_only", "full_paper_with_consent"],
    )
    parser.add_argument("--abstract-only", action="store_true", help="Shortcut for --confidentiality-mode abstract_only")
    parser.add_argument("--json-output", default="reports/advanced_ai_reviews.json")
    parser.add_argument("--csv-output", default="reports/advanced_ai_reviews.csv")
    parser.add_argument("--html-output", default="reports/advanced_ai_reviews.html")
    args = parser.parse_args()

    mode = ConfidentialityMode.ABSTRACT_ONLY if args.abstract_only else parse_mode(args.confidentiality_mode)
    payload = build_report(
        Path(args.local_json),
        Path(args.papers_dir),
        args.per_group,
        args.max_chars,
        mode,
    )
    write_json(Path(args.json_output), payload)
    write_csv_report(payload, Path(args.csv_output))
    write_html_report(payload, Path(args.html_output))
    print(f"Saved {args.json_output}")
    print(f"Saved {args.csv_output}")
    print(f"Saved {args.html_output}")


if __name__ == "__main__":
    main()
