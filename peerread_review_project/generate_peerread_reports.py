from __future__ import annotations

import csv
import html
import json
from collections import Counter

from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR
from peerread_review.data import parse_bool, read_peerread_rows
from peerread_review.model import load_model, predict
from peerread_review.xai import explain


def xai_focus_text(xai: dict) -> str:
    factors = xai.get("risk_factors") or xai.get("key_factors") or []
    return "; ".join(f"{item['label']}: {item['value']}" for item in factors[:3])


def build_report_rows() -> list[dict]:
    model = load_model(DEFAULT_MODEL_PATH)
    rows = []
    for row in read_peerread_rows(DEFAULT_DATASET_PATH):
        prediction = predict(model, row)
        xai = explain(model, prediction)
        recommendations = xai.get("recommendations") or ["Polish the paper before submission."]
        rows.append(
            {
                "paper_id": row.get("paper_id", ""),
                "conference": row.get("conference", ""),
                "split": row.get("split", ""),
                "title": row.get("title", ""),
                "actual_label": "Accept" if parse_bool(row.get("accepted")) else "Reject",
                "predicted_decision": prediction["decision"],
                "accept_probability": prediction["accept_probability"],
                "reject_probability": prediction["reject_probability"],
                "xai_focus": xai_focus_text(xai),
                "suggestion_1": recommendations[0] if len(recommendations) > 0 else "",
                "suggestion_2": recommendations[1] if len(recommendations) > 1 else "",
                "suggestion_3": recommendations[2] if len(recommendations) > 2 else "",
                "suggestions": " ".join(recommendations),
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    path = DEFAULT_REPORT_DIR / "peerread_decisions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict]) -> None:
    counts = Counter(row["predicted_decision"] for row in rows)
    actual = Counter(row["actual_label"] for row in rows)
    payload = {
        "counts": dict(counts),
        "actual_counts": dict(actual),
        "papers": rows,
    }
    (DEFAULT_REPORT_DIR / "peerread_decisions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def bar_svg(counts: Counter) -> str:
    labels = ["Accept", "Modify", "Reject"]
    colors = {"Accept": "#157347", "Modify": "#c77700", "Reject": "#b42318"}
    max_value = max([counts.get(label, 0) for label in labels] + [1])
    parts = [
        '<svg viewBox="0 0 760 300" role="img" aria-label="PeerRead decision distribution">',
        '<rect width="760" height="300" fill="white"/>',
        '<text x="24" y="36" style="font:700 24px Arial">PeerRead Predicted Decisions</text>',
    ]
    for index, label in enumerate(labels):
        value = counts.get(label, 0)
        x = 80 + index * 210
        h = 190 * value / max_value
        y = 245 - h
        parts.append(f'<rect x="{x}" y="{y}" width="110" height="{h}" rx="8" fill="{colors[label]}"/>')
        parts.append(f'<text x="{x + 55}" y="{y - 10}" text-anchor="middle" style="font:700 18px Arial">{value:,}</text>')
        parts.append(f'<text x="{x + 55}" y="275" text-anchor="middle" style="font:600 16px Arial">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write_html(rows: list[dict]) -> None:
    counts = Counter(row["predicted_decision"] for row in rows)
    headers = list(rows[0])
    table_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(header, '')))}</td>" for header in headers)
        table_rows.append(f"<tr>{cells}</tr>")
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PeerRead Decisions</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 28px; color: #172026; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 14px; margin: 18px 0; }}
.metric {{ border: 1px solid #d0d5dd; border-radius: 8px; padding: 14px; }}
.metric strong {{ display: block; font-size: 28px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
th {{ background: #f2f4f7; position: sticky; top: 0; }}
</style>
</head>
<body>
<h1>PeerRead Supervised Paper Screening</h1>
<p>This report is generated from the separate PeerRead dataset with true Accept/Reject labels.</p>
<div class="grid">
  <div class="metric"><strong>{len(rows):,}</strong>Total papers</div>
  <div class="metric"><strong>{counts.get("Accept", 0):,}</strong>Predicted Accept</div>
  <div class="metric"><strong>{counts.get("Reject", 0):,}</strong>Predicted Reject</div>
</div>
{bar_svg(counts)}
<table><thead><tr>{head}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</body>
</html>
"""
    (DEFAULT_REPORT_DIR / "peerread_decisions.html").write_text(doc, encoding="utf-8")


def main() -> None:
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_report_rows()
    write_csv(rows)
    write_json(rows)
    write_html(rows)
    print(f"Saved reports to {DEFAULT_REPORT_DIR}")


if __name__ == "__main__":
    main()
