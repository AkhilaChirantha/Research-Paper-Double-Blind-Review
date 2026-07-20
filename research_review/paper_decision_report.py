from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

from research_review.config import DEFAULT_CRITERIA_PATH, DEFAULT_MODEL_PATH, DEFAULT_PAPERS_DIR
from research_review.io import read_document, read_json, write_json
from research_review.model import load_model, predict_with_model, train
from research_review.xai import explain_prediction


VERDICT_LABELS = {
    "GOOD_PAPER": "Accept",
    "NEEDS_MODIFICATION": "Modify",
    "REJECT_RISK": "Reject",
}


def ensure_model(model_path: Path) -> dict:
    if model_path.exists():
        return load_model(model_path)
    return train(DEFAULT_CRITERIA_PATH, DEFAULT_PAPERS_DIR, model_path)


def xai_focus_text(xai_review: dict) -> str:
    factors = xai_review.get("risk_factors") or xai_review.get("key_factors") or []
    if not factors:
        return "No major XAI risk factor detected."
    parts = []
    for item in factors[:3]:
        value = item.get("value", "")
        parts.append(f"{item.get('label', item.get('feature'))}: {value}")
    return "; ".join(parts)


def short_suggestions(prediction: dict, xai_review: dict) -> list[str]:
    recommendations = xai_review.get("recommendations") or prediction.get("feature_gaps") or []
    verdict = prediction["verdict"]
    suggestions = list(recommendations[:4])
    if verdict == "GOOD_PAPER":
        suggestions.insert(0, "XAI indicates this paper is close to submission-ready; polish the listed risk factors.")
    elif verdict == "NEEDS_MODIFICATION":
        suggestions.insert(0, "XAI recommends revision before submission based on the listed risk factors.")
    else:
        suggestions.insert(0, "XAI flags high rejection risk; address the listed weaknesses before submission.")
    return suggestions[:5]


def build_rows(
    criteria_path: Path,
    papers_dir: Path,
    model_path: Path,
    limit: int | None = None,
    include_missing: bool = False,
) -> list[dict]:
    model = ensure_model(model_path)
    metadata = read_json(criteria_path)
    rows = []
    for index, item in enumerate(metadata):
        if limit is not None and index >= limit:
            break
        paper_id = item.get("paper_id", "")
        paper_path = papers_dir / f"{paper_id}.md"
        if not paper_path.exists():
            if not include_missing:
                continue
            rows.append(
                {
                    "paper_id": paper_id,
                    "title": item.get("title", ""),
                    "actual_decision": item.get("decision", ""),
                    "predicted_decision": "Missing file",
                    "quality_score": "",
                    "accept_probability": "",
                    "modify_probability": "",
                    "reject_probability": "",
                    "xai_focus": "",
                    "suggestion_1": "Paper markdown file not found in data/Research Papers.",
                    "suggestion_2": "",
                    "suggestion_3": "",
                    "suggestions": "Paper markdown file not found in data/Research Papers.",
                }
            )
            continue

        text = read_document(paper_path)
        prediction = predict_with_model(model, text)
        xai_review = explain_prediction(model, prediction, top_n=5)
        probabilities = prediction["probabilities"]
        suggestions = short_suggestions(prediction, xai_review)
        rows.append(
            {
                "paper_id": paper_id,
                "title": item.get("title", ""),
                "actual_decision": item.get("decision", ""),
                "predicted_decision": VERDICT_LABELS[prediction["verdict"]],
                "quality_score": prediction["quality_score"],
                "accept_probability": round(probabilities.get("good_paper", 0.0), 4),
                "modify_probability": round(probabilities.get("needs_modification", 0.0), 4),
                "reject_probability": round(probabilities.get("reject_risk", 0.0), 4),
                "xai_focus": xai_focus_text(xai_review),
                "suggestion_1": suggestions[0] if len(suggestions) > 0 else "",
                "suggestion_2": suggestions[1] if len(suggestions) > 1 else "",
                "suggestion_3": suggestions[2] if len(suggestions) > 2 else "",
                "suggestions": " ".join(suggestions),
            }
        )
    return rows


def decision_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"Accept": 0, "Modify": 0, "Reject": 0, "Missing file": 0}
    for row in rows:
        decision = row["predicted_decision"]
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def svg_bar_chart(counts: dict[str, int]) -> str:
    items = [("Accept", counts.get("Accept", 0)), ("Modify", counts.get("Modify", 0)), ("Reject", counts.get("Reject", 0))]
    width = 780
    height = 340
    margin_left = 70
    margin_bottom = 52
    margin_top = 48
    usable_w = width - margin_left - 36
    usable_h = height - margin_top - margin_bottom
    max_value = max([value for _, value in items] + [1])
    col_w = usable_w / len(items)
    colors = {"Accept": "#238855", "Modify": "#d08a1d", "Reject": "#c43c39"}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="Paper decisions bar chart">',
        '<text x="20" y="30" class="chart-title">Predicted Paper Decisions</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 24}" y2="{height - margin_bottom}" class="axis"></line>',
    ]
    for i, (label, value) in enumerate(items):
        bar_h = usable_h * value / max_value
        x = margin_left + i * col_w + col_w * 0.22
        y = height - margin_bottom - bar_h
        w = col_w * 0.56
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="6" fill="{colors[label]}"></rect>'
        )
        parts.append(f'<text x="{x + w / 2:.1f}" y="{y - 10:.1f}" class="value center">{value:,}</text>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{height - 24}" class="label center">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_donut_chart(counts: dict[str, int]) -> str:
    total = counts.get("Accept", 0) + counts.get("Modify", 0) + counts.get("Reject", 0)
    if total == 0:
        total = 1
    width = 420
    height = 300
    radius = 92
    circumference = 2 * 3.141592653589793 * radius
    colors = [("#238855", "Accept"), ("#d08a1d", "Modify"), ("#c43c39", "Reject")]
    offset = 0.0
    circles = []
    legend = []
    for index, (color, label) in enumerate(colors):
        value = counts.get(label, 0)
        dash = circumference * value / total
        circles.append(
            f'<circle cx="145" cy="155" r="{radius}" class="donut" stroke="{color}" '
            f'stroke-dasharray="{dash:.2f} {circumference - dash:.2f}" stroke-dashoffset="{-offset:.2f}"></circle>'
        )
        offset += dash
        pct = 100 * value / total
        y = 104 + index * 38
        legend.append(f'<rect x="285" y="{y - 13}" width="14" height="14" rx="3" fill="{color}"></rect>')
        legend.append(f'<text x="308" y="{y}" class="label">{label}: {value:,} ({pct:.1f}%)</text>')
    return "\n".join(
        [
            f'<svg viewBox="0 0 {width} {height}" class="chart donut-chart" role="img" aria-label="Decision share chart">',
            '<text x="20" y="30" class="chart-title">Decision Share</text>',
            '<circle cx="145" cy="155" r="92" class="donut-bg"></circle>',
            *circles,
            '<text x="145" y="150" class="value center">Total</text>',
            f'<text x="145" y="175" class="big center">{total:,}</text>',
            *legend,
            "</svg>",
        ]
    )


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def html_table(rows: list[dict]) -> str:
    headers = [
        "Paper ID",
        "Paper Name",
        "Actual Dataset Decision",
        "Predicted",
        "Score",
        "Accept P",
        "Modify P",
        "Reject P",
        "XAI Focus",
        "Suggestion 1",
        "Suggestion 2",
        "Suggestion 3",
        "Suggestions",
    ]
    body = []
    for row in rows:
        css = str(row["predicted_decision"]).lower().replace(" ", "-")
        body.append(
            "<tr>"
            f"<td><code>{html.escape(str(row['paper_id']))}</code></td>"
            f"<td>{html.escape(str(row['title']))}</td>"
            f"<td>{html.escape(str(row['actual_decision']))}</td>"
            f'<td><span class="pill {css}">{html.escape(str(row["predicted_decision"]))}</span></td>'
            f"<td>{html.escape(str(row['quality_score']))}</td>"
            f"<td>{html.escape(str(row['accept_probability']))}</td>"
            f"<td>{html.escape(str(row['modify_probability']))}</td>"
            f"<td>{html.escape(str(row['reject_probability']))}</td>"
            f"<td>{html.escape(str(row.get('xai_focus', '')))}</td>"
            f"<td>{html.escape(str(row.get('suggestion_1', '')))}</td>"
            f"<td>{html.escape(str(row.get('suggestion_2', '')))}</td>"
            f"<td>{html.escape(str(row.get('suggestion_3', '')))}</td>"
            f"<td>{html.escape(str(row['suggestions']))}</td>"
            "</tr>"
        )
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def write_html(rows: list[dict], output_path: Path) -> None:
    counts = decision_counts(rows)
    total = sum(counts.values())
    css = """
body { margin: 0; background: #f6f7f9; color: #172026; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 30px 22px 56px; }
h1 { margin: 0 0 6px; font-size: 32px; }
.muted { color: #5c6772; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin: 18px 0; }
.metric, .panel { background: white; border: 1px solid #dfe5eb; border-radius: 8px; padding: 16px; }
.metric strong { display: block; font-size: 30px; }
.table-wrap { overflow-x: auto; background: white; border: 1px solid #dfe5eb; border-radius: 8px; margin-top: 18px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; border-bottom: 1px solid #e7ebef; padding: 10px 12px; vertical-align: top; }
th { background: #eef2f5; position: sticky; top: 0; z-index: 1; }
td:nth-child(2) { min-width: 280px; }
td:nth-child(9) { min-width: 420px; }
.pill { display: inline-block; min-width: 58px; text-align: center; border-radius: 999px; padding: 4px 9px; color: white; font-weight: 700; }
.accept { background: #238855; }
.modify { background: #d08a1d; }
.reject { background: #c43c39; }
.missing-file { background: #69737d; }
.chart { width: 100%; height: auto; display: block; }
.chart-title { font-size: 18px; font-weight: 750; fill: #172026; }
.label { font-size: 13px; fill: #3d4954; }
.value { font-size: 13px; fill: #172026; font-weight: 700; }
.big { font-size: 22px; fill: #172026; font-weight: 800; }
.center { text-anchor: middle; }
.axis { stroke: #aeb8c2; stroke-width: 1; }
.donut-bg { fill: none; stroke: #e8edf1; stroke-width: 30; }
.donut { fill: none; stroke-width: 30; transform: rotate(-90deg); transform-origin: 145px 155px; }
"""
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper-by-Paper Review Decisions</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Paper-by-Paper Review Decisions</h1>
  <p class="muted">Each row is one paper. Predicted decisions come from the local model: Accept, Modify, or Reject-risk.</p>
  <section class="grid">
    <div class="metric"><strong>{total:,}</strong><span>Total papers checked</span></div>
    <div class="metric"><strong>{counts.get("Accept", 0):,}</strong><span>Predicted accepts</span></div>
    <div class="metric"><strong>{counts.get("Modify", 0):,}</strong><span>Need modification</span></div>
    <div class="metric"><strong>{counts.get("Reject", 0):,}</strong><span>Reject-risk papers</span></div>
  </section>
  <section class="grid">
    <div class="panel">{svg_bar_chart(counts)}</div>
    <div class="panel">{svg_donut_chart(counts)}</div>
  </section>
  <div class="table-wrap">{html_table(rows)}</div>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-by-paper accept/modify/reject report.")
    parser.add_argument("--criteria", default=str(DEFAULT_CRITERIA_PATH))
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--html-output", default="reports/paper_decisions.html")
    parser.add_argument("--csv-output", default="reports/paper_decisions.csv")
    parser.add_argument("--json-output", default="reports/paper_decisions.json")
    parser.add_argument("--limit", type=int, help="Optional limit for quick testing")
    parser.add_argument("--include-missing", action="store_true", help="Include metadata rows with no markdown file")
    args = parser.parse_args()

    rows = build_rows(
        Path(args.criteria),
        Path(args.papers_dir),
        Path(args.model_path),
        args.limit,
        args.include_missing,
    )
    write_csv(rows, Path(args.csv_output))
    write_json(Path(args.json_output), {"counts": decision_counts(rows), "papers": rows})
    write_html(rows, Path(args.html_output))
    print(f"Saved {args.html_output}")
    print(f"Saved {args.csv_output}")
    print(f"Saved {args.json_output}")


if __name__ == "__main__":
    main()
