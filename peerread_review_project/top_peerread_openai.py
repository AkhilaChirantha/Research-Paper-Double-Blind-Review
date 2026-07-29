from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR
from peerread_review.data import paper_text, parse_bool, read_peerread_rows
from peerread_review.model import load_model, predict
from peerread_review.xai import explain
from research_review.confidentiality import ConfidentialityMode, parse_mode, prepare_review_text
from research_review.config import load_env, openai_model
from research_review.openai_reviewer import trim_for_review


ADVANCED_PEERREAD_SCHEMA = {
    "type": "json_schema",
    "name": "peerread_paper_level_ai_review",
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


def modify_probability(accept_probability: float) -> float:
    return round(1.0 - abs(float(accept_probability) - 0.5) * 2, 4)


def build_scored_rows() -> list[dict]:
    model = load_model(DEFAULT_MODEL_PATH)
    rows = []
    for row in read_peerread_rows(DEFAULT_DATASET_PATH):
        prediction = predict(model, row)
        xai = explain(model, prediction)
        rows.append(
            {
                "source_row": row,
                "paper_id": row.get("paper_id"),
                "conference": row.get("conference"),
                "split": row.get("split"),
                "title": row.get("title"),
                "actual_label": "Accept" if parse_bool(row.get("accepted")) else "Reject",
                "predicted_decision": prediction["decision"],
                "quality_score": round(100 * prediction["accept_probability"], 1),
                "accept_probability": prediction["accept_probability"],
                "modify_probability": modify_probability(prediction["accept_probability"]),
                "reject_probability": prediction["reject_probability"],
                "suggestions": " ".join(xai.get("recommendations", [])),
                "xai_focus": "; ".join(
                    f"{item['label']}: {item['value']}" for item in xai.get("risk_factors", [])[:3]
                ),
            }
        )
    return rows


def select_top_papers(rows: list[dict], per_group: int) -> list[dict]:
    accept = sorted(
        [row for row in rows if row["predicted_decision"] == "Accept"],
        key=lambda row: float(row["accept_probability"]),
        reverse=True,
    )[:per_group]
    modify = sorted(
        [row for row in rows if row["predicted_decision"] == "Modify"],
        key=lambda row: abs(float(row["accept_probability"]) - 0.5),
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


def call_openai_review(client: OpenAI, row: dict, review_text: str, max_chars: int) -> dict:
    prompt = {
        "task": (
            "Review this research paper before submission using a double-blind academic reviewer style. "
            "Give paper-specific good points, weak points, and exact modification suggestions with priority. "
            "The goal is to help the author improve the paper until it reaches acceptance level."
        ),
        "local_model_output": {
            "decision": row["predicted_decision"],
            "actual_peerread_label": row["actual_label"],
            "quality_score": row["quality_score"],
            "accept_probability": row["accept_probability"],
            "modify_probability": row["modify_probability"],
            "reject_probability": row["reject_probability"],
            "xai_focus": row["xai_focus"],
            "local_xai_suggestions": row["suggestions"],
        },
        "paper": {
            "paper_id": row["paper_id"],
            "title": row["title"],
            "conference": row["conference"],
            "text": trim_for_review(review_text, max_chars=max_chars),
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
                    "You are a strict but constructive double-blind academic reviewer. "
                    "Do not give generic advice. Every good point, weak point, and modification "
                    "must be specific to the paper content."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        text={"format": ADVANCED_PEERREAD_SCHEMA},
    )
    return json.loads(response.output_text)


def build_report(per_group: int, max_chars: int, mode: ConfidentialityMode) -> dict:
    load_env(Path(".env"))
    client = OpenAI()
    selected = select_top_papers(build_scored_rows(), per_group)
    reviewed = []
    for index, row in enumerate(selected, start=1):
        text = paper_text(row["source_row"])
        review_text, audit = prepare_review_text(text, str(row["paper_id"]), mode)
        if not audit.get("api_allowed"):
            raise RuntimeError("OpenAI review cannot run in local_only mode.")
        ai_review = call_openai_review(client, row, review_text, max_chars)
        item = {key: value for key, value in row.items() if key != "source_row"}
        item.update(
            {
                "group": group_name(row["predicted_decision"]),
                "ai_review": ai_review,
                "confidentiality": audit,
            }
        )
        reviewed.append(item)
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
    return " ".join(f"{item['section']} ({item['priority']}): {item['suggestion']}" for item in items)


def write_csv_report(payload: dict, output_path: Path) -> None:
    rows = []
    for item in payload["papers"]:
        review = item["ai_review"]
        rows.append(
            {
                "group": item["group"],
                "paper_id": item["paper_id"],
                "title": item["title"],
                "actual_label": item["actual_label"],
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
    css = str(label).lower().replace("_", "-")
    return f'<span class="pill {css}">{html.escape(str(label))}</span>'


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
            good = "".join(f"<li>{html.escape(point)}</li>" for point in review["good_points"])
            weak = "".join(f"<li>{html.escape(point)}</li>" for point in review["weak_points"])
            plan = "".join(f"<li>{html.escape(point)}</li>" for point in review["acceptance_plan"])
            cards.append(
                f"""
                <article class="card">
                  <div class="card-head">
                    <div>
                      <h3>{html.escape(str(item['title']))}</h3>
                      <p><code>{html.escape(str(item['paper_id']))}</code> | PeerRead label: {pill(item['actual_label'])}</p>
                    </div>
                    <div class="decision-box">
                      <div>Local: {pill(item['predicted_decision'])}</div>
                      <div>OpenAI: {pill(review['ai_decision'])}</div>
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
.pill { display: inline-block; min-width: 64px; text-align: center; border-radius: 999px; padding: 3px 9px; color: white; font-weight: 700; font-size: 12px; background: #475467; }
.accept, .accepted { background: #157347; }
.modify { background: #c77700; }
.reject { background: #b42318; }
.note { background: #f8fafc; border-left: 4px solid #2477b3; padding: 10px 12px; }
"""
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PeerRead Advanced OpenAI Paper Review Report</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>PeerRead Advanced OpenAI Paper Review Report</h1>
  <p class="muted">Model: {html.escape(payload.get("model", "unknown"))} | Per group: {payload.get("per_group")} | Confidentiality: {html.escape(payload.get("confidentiality_mode", ""))}</p>
  {''.join(sections)}
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detailed OpenAI feedback for selected PeerRead papers.")
    parser.add_argument("--per-group", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=14000)
    parser.add_argument(
        "--confidentiality-mode",
        default=ConfidentialityMode.SECTION_SUMMARY_ONLY.value,
        choices=[mode.value for mode in ConfidentialityMode],
    )
    args = parser.parse_args()
    payload = build_report(args.per_group, args.max_chars, parse_mode(args.confidentiality_mode))
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DEFAULT_REPORT_DIR / "peerread_openai_reviews.json"
    csv_path = DEFAULT_REPORT_DIR / "peerread_openai_reviews.csv"
    html_path = DEFAULT_REPORT_DIR / "peerread_openai_reviews.html"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv_report(payload, csv_path)
    write_html_report(payload, html_path)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")
    print(f"Saved {html_path}")


if __name__ == "__main__":
    main()
