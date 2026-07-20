from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

from research_review.confidentiality import ConfidentialityMode, parse_mode, prepare_review_text
from research_review.config import DEFAULT_PAPERS_DIR
from research_review.io import read_document, read_json, write_json
from research_review.openai_reviewer import get_openai_recommendation


LOCAL_TO_OPENAI = {
    "Accept": "GOOD_PAPER",
    "Modify": "NEEDS_MODIFICATION",
    "Reject": "REJECT_RISK",
}

OPENAI_TO_LABEL = {
    "GOOD_PAPER": "Accept",
    "NEEDS_MODIFICATION": "Modify",
    "REJECT_RISK": "Reject",
}


def select_rows(rows: list[dict], per_class: int, limit: int | None) -> list[dict]:
    selected = []
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        decision = row.get("predicted_decision")
        if decision in {"Accept", "Modify", "Reject"}:
            by_class[decision].append(row)

    for decision in ("Accept", "Modify", "Reject"):
        selected.extend(by_class[decision][:per_class])

    if limit is not None:
        selected = selected[:limit]
    return selected


def local_prediction_from_row(row: dict) -> dict:
    return {
        "verdict": LOCAL_TO_OPENAI.get(row["predicted_decision"], "NEEDS_MODIFICATION"),
        "quality_score": row.get("quality_score", 0),
        "probabilities": {
            "good_paper": float(row.get("accept_probability") or 0),
            "needs_modification": float(row.get("modify_probability") or 0),
            "reject_risk": float(row.get("reject_probability") or 0),
        },
        "feature_gaps": [part.strip() for part in str(row.get("suggestions", "")).split(".") if part.strip()],
    }


def flatten_openai_suggestions(review: dict) -> str:
    items = []
    for item in review.get("section_level_suggestions", []):
        section = item.get("section", "Section")
        recommendation = item.get("recommendation", "")
        priority = item.get("priority", "medium")
        items.append(f"{section} ({priority}): {recommendation}")
    if not items:
        items = review.get("acceptance_plan", [])
    return " ".join(items)


def compare_rows(local_rows: list[dict], papers_dir: Path, max_chars: int, mode: ConfidentialityMode) -> list[dict]:
    compared = []
    for index, row in enumerate(local_rows, start=1):
        paper_id = row["paper_id"]
        paper_path = papers_dir / f"{paper_id}.md"
        text = read_document(paper_path)
        text, audit = prepare_review_text(text, str(paper_path), mode)
        if not audit.get("api_allowed"):
            raise RuntimeError("OpenAI comparison cannot run in local_only mode.")
        local_prediction = local_prediction_from_row(row)
        review = get_openai_recommendation(text, local_prediction, max_chars=max_chars)
        compared.append(
            {
                "paper_id": paper_id,
                "title": row["title"],
                "actual_decision": row["actual_decision"],
                "local_decision": row["predicted_decision"],
                "local_quality_score": row["quality_score"],
                "local_suggestions": row["suggestions"],
                "openai_decision": OPENAI_TO_LABEL.get(review["final_verdict"], review["final_verdict"]),
                "openai_confidence": round(float(review.get("confidence", 0)), 3),
                "openai_summary": review.get("overall_summary", ""),
                "openai_reasons": " ".join(review.get("main_reasons", [])),
                "openai_suggestions": flatten_openai_suggestions(review),
                "openai_questions": " ".join(review.get("reviewer_questions", [])),
                "agreement": row["predicted_decision"] == OPENAI_TO_LABEL.get(review["final_verdict"], review["final_verdict"]),
                "confidentiality": audit,
            }
        )
        print(f"[{index}/{len(local_rows)}] {paper_id}: local={row['predicted_decision']} openai={compared[-1]['openai_decision']}")
    return compared


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def html_table(rows: list[dict]) -> str:
    headers = [
        "Paper",
        "Title",
        "Local",
        "OpenAI",
        "Agreement",
        "Local Suggestions",
        "OpenAI Suggestions",
    ]
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        agree = "Yes" if row["agreement"] else "No"
        body.append(
            "<tr>"
            f"<td><code>{html.escape(row['paper_id'])}</code></td>"
            f"<td>{html.escape(row['title'])}</td>"
            f"<td>{html.escape(row['local_decision'])}</td>"
            f"<td>{html.escape(row['openai_decision'])}</td>"
            f"<td>{agree}</td>"
            f"<td>{html.escape(row['local_suggestions'])}</td>"
            f"<td>{html.escape(row['openai_suggestions'])}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(rows: list[dict], output_path: Path) -> None:
    local_counts = Counter(row["local_decision"] for row in rows)
    openai_counts = Counter(row["openai_decision"] for row in rows)
    agreement = sum(1 for row in rows if row["agreement"])
    total = len(rows)
    css = """
body { margin: 0; background: #f6f7f9; color: #172026; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 30px 22px 56px; }
h1 { margin: 0 0 8px; font-size: 32px; }
.muted { color: #5c6772; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin: 18px 0; }
.metric, .panel { background: white; border: 1px solid #dfe5eb; border-radius: 8px; padding: 16px; }
.metric strong { display: block; font-size: 30px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; border-bottom: 1px solid #e7ebef; padding: 10px 12px; vertical-align: top; }
th { background: #eef2f5; }
td:nth-child(2) { min-width: 260px; }
td:nth-child(6), td:nth-child(7) { min-width: 360px; }
.table-wrap { overflow-x: auto; background: white; border: 1px solid #dfe5eb; border-radius: 8px; }
"""
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local vs OpenAI Paper Review Comparison</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Local vs OpenAI Paper Review Comparison</h1>
  <p class="muted">OpenAI review is run on a cost-controlled sample selected from the local paper decision report.</p>
  <section class="grid">
    <div class="metric"><strong>{total}</strong><span>Papers compared</span></div>
    <div class="metric"><strong>{agreement}</strong><span>Same decision</span></div>
    <div class="metric"><strong>{100 * agreement / total if total else 0:.1f}%</strong><span>Agreement rate</span></div>
  </section>
  <section class="grid">
    <div class="panel"><h2>Local Counts</h2><pre>{html.escape(str(dict(local_counts)))}</pre></div>
    <div class="panel"><h2>OpenAI Counts</h2><pre>{html.escape(str(dict(openai_counts)))}</pre></div>
  </section>
  <div class="table-wrap">{html_table(rows)}</div>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenAI reviews for a sample and compare with local decisions.")
    parser.add_argument("--local-json", default="reports/paper_decisions.json")
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--per-class", type=int, default=10, help="Sample this many Accept/Modify/Reject rows")
    parser.add_argument("--limit", type=int, help="Optional hard cap after class sampling")
    parser.add_argument("--max-chars", type=int, default=12000, help="Max paper characters sent per OpenAI call")
    parser.add_argument(
        "--confidentiality-mode",
        default="section_summary_only",
        choices=["abstract_only", "section_summary_only", "full_paper_with_consent"],
    )
    parser.add_argument("--abstract-only", action="store_true", help="Shortcut for --confidentiality-mode abstract_only")
    parser.add_argument("--json-output", default="reports/openai_comparison.json")
    parser.add_argument("--csv-output", default="reports/openai_comparison.csv")
    parser.add_argument("--html-output", default="reports/openai_comparison.html")
    args = parser.parse_args()

    local = read_json(Path(args.local_json))
    selected = select_rows(local["papers"], args.per_class, args.limit)
    if not selected:
        raise RuntimeError("No local paper rows selected. Run python3 paper_decisions.py first.")
    mode = ConfidentialityMode.ABSTRACT_ONLY if args.abstract_only else parse_mode(args.confidentiality_mode)
    compared = compare_rows(selected, Path(args.papers_dir), args.max_chars, mode)
    payload = {
        "sampling": {
            "per_class": args.per_class,
            "limit": args.limit,
            "max_chars": args.max_chars,
            "confidentiality_mode": mode.value,
        },
        "counts": {
            "local": dict(Counter(row["local_decision"] for row in compared)),
            "openai": dict(Counter(row["openai_decision"] for row in compared)),
            "agreement": sum(1 for row in compared if row["agreement"]),
            "total": len(compared),
        },
        "papers": compared,
    }
    write_json(Path(args.json_output), payload)
    write_csv(compared, Path(args.csv_output))
    write_html(compared, Path(args.html_output))
    print(f"Saved {args.json_output}")
    print(f"Saved {args.csv_output}")
    print(f"Saved {args.html_output}")


if __name__ == "__main__":
    main()
