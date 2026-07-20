from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from research_review.config import DEFAULT_CRITERIA_PATH, DEFAULT_PAPERS_DIR
from research_review.model import paper_training_label, parse_score


CONCERN_CATEGORIES = {
    "limited experiments": [
        "experiment",
        "evaluation",
        "dataset",
        "benchmark",
        "ablation",
        "baseline",
        "compare",
        "comparison",
    ],
    "clarity / writing": [
        "clarity",
        "presentation",
        "writing",
        "explain",
        "unclear",
        "readability",
        "organization",
    ],
    "novelty / contribution": [
        "novel",
        "novelty",
        "contribution",
        "incremental",
        "impact",
        "significance",
    ],
    "technical soundness": [
        "proof",
        "theorem",
        "theory",
        "assumption",
        "soundness",
        "correctness",
        "validity",
    ],
    "reproducibility": [
        "code",
        "reproduc",
        "implementation",
        "hyperparameter",
        "details",
        "release",
    ],
    "related work": [
        "related work",
        "prior work",
        "literature",
        "citation",
        "cite",
    ],
    "limitations / ethics": [
        "limitation",
        "failure",
        "ethic",
        "broader impact",
        "risk",
    ],
}


STOPWORDS = {
    "about",
    "after",
    "also",
    "although",
    "and",
    "are",
    "because",
    "been",
    "being",
    "can",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "most",
    "not",
    "only",
    "other",
    "paper",
    "papers",
    "proposed",
    "results",
    "show",
    "some",
    "such",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "using",
    "were",
    "with",
    "work",
    "would",
}


def load_data(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: int | float, total: int | float) -> str:
    if not total:
        return "0.0%"
    return f"{100 * value / total:.1f}%"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    items = sorted(values)
    middle = len(items) // 2
    if len(items) % 2:
        return items[middle]
    return (items[middle - 1] + items[middle]) / 2


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z\-]{3,}", text.lower())


def review_rows(data: list[dict]) -> list[dict]:
    rows = []
    for paper in data:
        for review in paper.get("reviews", []):
            if review.get("reviewer_id") == "Authors":
                continue
            rating = parse_score(review.get("rating"))
            if rating is None:
                continue
            rows.append(
                {
                    "paper_id": paper.get("paper_id", ""),
                    "decision": paper.get("decision", ""),
                    "rating": rating,
                    "confidence": parse_score(review.get("confidence")),
                    "soundness": parse_score(review.get("soundness")),
                    "presentation": parse_score(review.get("presentation")),
                    "contribution": parse_score(review.get("contribution")),
                    "summary": review.get("summary") or "",
                    "strengths": review.get("strengths") or "",
                    "weaknesses": review.get("weaknesses") or "",
                    "questions": review.get("questions") or "",
                }
            )
    return rows


def summarize(data: list[dict]) -> dict:
    reviews = review_rows(data)
    decision_counts = Counter(paper.get("decision", "Unknown") for paper in data)
    derived_counts = Counter(paper_training_label(paper) for paper in data)
    rating_counts = Counter(int(row["rating"]) for row in reviews)

    decision_rating: dict[str, list[float]] = defaultdict(list)
    subscore_values: dict[str, list[float]] = defaultdict(list)
    for row in reviews:
        decision_rating[row["decision"]].append(row["rating"])
        for key in ("soundness", "presentation", "contribution", "confidence"):
            if row[key] is not None:
                subscore_values[key].append(row[key])

    concern_counts = Counter()
    weakness_word_counts = Counter()
    question_word_counts = Counter()
    for row in reviews:
        weakness_text = row["weaknesses"].lower()
        question_text = row["questions"].lower()
        combined = weakness_text + "\n" + question_text
        for category, terms in CONCERN_CATEGORIES.items():
            if any(term in combined for term in terms):
                concern_counts[category] += 1
        weakness_word_counts.update(
            word for word in tokenize(row["weaknesses"]) if word not in STOPWORDS
        )
        question_word_counts.update(
            word for word in tokenize(row["questions"]) if word not in STOPWORDS
        )

    abstract_lengths = [float(paper.get("abstract", "") and len(tokenize(paper.get("abstract", "")))) for paper in data]
    review_lengths = [
        len(tokenize(row["summary"] + " " + row["strengths"] + " " + row["weaknesses"] + " " + row["questions"]))
        for row in reviews
    ]

    return {
        "paper_count": len(data),
        "review_count": len(reviews),
        "decision_counts": decision_counts,
        "derived_counts": derived_counts,
        "rating_counts": rating_counts,
        "decision_rating": decision_rating,
        "subscore_values": subscore_values,
        "concern_counts": concern_counts,
        "weakness_words": weakness_word_counts,
        "question_words": question_word_counts,
        "abstract_lengths": abstract_lengths,
        "review_lengths": review_lengths,
    }


def svg_bar_chart(items: list[tuple[str, float]], title: str, width: int = 760, height: int = 320) -> str:
    margin_left = 170
    margin_right = 40
    margin_top = 42
    row_h = 34
    chart_h = max(height, margin_top + len(items) * row_h + 30)
    max_value = max((value for _, value in items), default=1)
    parts = [
        f'<svg viewBox="0 0 {width} {chart_h}" class="chart" role="img" aria-label="{html.escape(title)}">',
        f'<text x="20" y="26" class="chart-title">{html.escape(title)}</text>',
    ]
    usable = width - margin_left - margin_right
    for i, (label, value) in enumerate(items):
        y = margin_top + i * row_h
        bar_w = 0 if max_value == 0 else usable * value / max_value
        parts.append(f'<text x="20" y="{y + 20}" class="axis-label">{html.escape(label)}</text>')
        parts.append(f'<rect x="{margin_left}" y="{y + 5}" width="{bar_w:.1f}" height="20" rx="4" class="bar"></rect>')
        parts.append(f'<text x="{margin_left + bar_w + 8:.1f}" y="{y + 20}" class="value-label">{value:.1f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_column_chart(items: list[tuple[str, float]], title: str, width: int = 760, height: int = 320) -> str:
    margin_left = 48
    margin_bottom = 54
    margin_top = 42
    usable_w = width - margin_left - 30
    usable_h = height - margin_top - margin_bottom
    max_value = max((value for _, value in items), default=1)
    col_w = usable_w / max(len(items), 1)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{html.escape(title)}">',
        f'<text x="20" y="26" class="chart-title">{html.escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 20}" y2="{height - margin_bottom}" class="axis-line"></line>',
    ]
    for i, (label, value) in enumerate(items):
        bar_h = 0 if max_value == 0 else usable_h * value / max_value
        x = margin_left + i * col_w + col_w * 0.18
        y = height - margin_bottom - bar_h
        w = col_w * 0.64
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="4" class="bar"></rect>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{y - 7:.1f}" class="value-label center">{value:.0f}</text>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{height - 28}" class="axis-label center">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def recommendations(summary: dict) -> list[str]:
    concerns = summary["concern_counts"]
    recs = []
    if concerns["limited experiments"]:
        recs.append("Prioritize stronger experiments: add stronger baselines, ablations, dataset diversity, and clearer metric reporting.")
    if concerns["clarity / writing"]:
        recs.append("Improve presentation before submission: make contribution bullets explicit, define notation early, and rewrite unclear method sections.")
    if concerns["novelty / contribution"]:
        recs.append("Sharpen novelty claims: say exactly what is new compared with the closest related work, not only that the method performs well.")
    if concerns["technical soundness"]:
        recs.append("Reduce technical-risk comments: state assumptions, proof scope, failure cases, and why the experimental evidence supports the claims.")
    if concerns["reproducibility"]:
        recs.append("Add reproducibility details: code link, hyperparameters, compute budget, seeds, implementation details, and data preprocessing.")
    if concerns["related work"]:
        recs.append("Strengthen related work: compare directly against the closest papers reviewers are likely to cite.")
    recs.append("Because this dataset has accepted papers only, add rejected papers later if you want a stronger true accept/reject classifier.")
    return recs


def render_html(summary: dict, output_path: Path) -> str:
    paper_count = summary["paper_count"]
    review_count = summary["review_count"]

    decision_rows = [
        [label, count, pct(count, paper_count)]
        for label, count in summary["decision_counts"].most_common()
    ]
    derived_labels = {
        "good_paper": "Good paper",
        "needs_modification": "Needs modification",
        "reject_risk": "Reject-risk",
    }
    derived_rows = [
        [derived_labels.get(label, label), count, pct(count, paper_count)]
        for label, count in summary["derived_counts"].most_common()
    ]
    rating_rows = [
        [rating, count, pct(count, review_count)]
        for rating, count in sorted(summary["rating_counts"].items())
    ]
    decision_rating_rows = [
        [decision, len(values), f"{mean(values):.2f}", f"{median(values):.1f}"]
        for decision, values in sorted(summary["decision_rating"].items())
    ]
    subscore_rows = [
        [name.title(), len(values), f"{mean(values):.2f}", f"{median(values):.1f}"]
        for name, values in summary["subscore_values"].items()
    ]
    concern_rows = [
        [label, count, pct(count, review_count)]
        for label, count in summary["concern_counts"].most_common()
    ]
    weakness_rows = [[word, count] for word, count in summary["weakness_words"].most_common(20)]
    question_rows = [[word, count] for word, count in summary["question_words"].most_common(20)]

    decision_chart = svg_bar_chart(
        [(label, count) for label, count in summary["decision_counts"].most_common()],
        "Decision Distribution",
    )
    derived_chart = svg_bar_chart(
        [(derived_labels.get(label, label), count) for label, count in summary["derived_counts"].most_common()],
        "Derived Paper Quality Classes",
    )
    rating_chart = svg_column_chart(
        [(str(label), count) for label, count in sorted(summary["rating_counts"].items())],
        "Reviewer Rating Distribution",
    )
    concern_chart = svg_bar_chart(
        [(label, count) for label, count in summary["concern_counts"].most_common()],
        "Most Common Reviewer Concern Areas",
        height=360,
    )

    abstract_lengths = summary["abstract_lengths"]
    review_lengths = summary["review_lengths"]
    length_rows = [
        ["Abstract words", len(abstract_lengths), f"{mean(abstract_lengths):.1f}", f"{median(abstract_lengths):.1f}"],
        ["Review words", len(review_lengths), f"{mean(review_lengths):.1f}", f"{median(review_lengths):.1f}"],
    ]

    rec_items = "".join(f"<li>{html.escape(item)}</li>" for item in recommendations(summary))

    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #172026; background: #f6f7f9; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 22px 56px; }
h1 { margin: 0 0 8px; font-size: 34px; }
h2 { margin-top: 34px; font-size: 22px; }
.muted { color: #5d6975; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }
.metric { background: white; border: 1px solid #dfe5eb; border-radius: 8px; padding: 18px; }
.metric strong { display: block; font-size: 30px; margin-bottom: 4px; }
.panel { background: white; border: 1px solid #dfe5eb; border-radius: 8px; padding: 18px; margin-top: 16px; overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th, td { text-align: left; border-bottom: 1px solid #e7ebef; padding: 10px 12px; vertical-align: top; }
th { background: #f0f3f6; color: #34404c; font-weight: 650; }
.chart { width: 100%; height: auto; display: block; }
.chart-title { font-size: 18px; font-weight: 700; fill: #172026; }
.axis-label { font-size: 13px; fill: #46515c; }
.value-label { font-size: 12px; fill: #172026; }
.center { text-anchor: middle; }
.bar { fill: #2477b3; }
.axis-line { stroke: #aeb8c2; stroke-width: 1; }
li { margin: 8px 0; }
"""
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research Review Dataset Report</title>
  <style>{css}</style>
</head>
<body>
<main>
  <h1>Research Review Dataset Report</h1>
  <p class="muted">Generated from <code>data/Analyse Critaria/metadata.json</code>. Charts are based only on the current local dataset.</p>

  <section class="grid">
    <div class="metric"><strong>{paper_count:,}</strong><span>Papers</span></div>
    <div class="metric"><strong>{review_count:,}</strong><span>Reviewer reviews</span></div>
    <div class="metric"><strong>{mean([row["rating"] for row in review_rows(load_data(DEFAULT_CRITERIA_PATH))]):.2f}</strong><span>Average reviewer rating</span></div>
  </section>

  <h2>Charts</h2>
  <div class="grid">
    <div class="panel">{decision_chart}</div>
    <div class="panel">{derived_chart}</div>
    <div class="panel">{rating_chart}</div>
    <div class="panel">{concern_chart}</div>
  </div>

  <h2>Decision Tables</h2>
  <div class="grid">
    <div class="panel">{table(["Decision", "Papers", "Share"], decision_rows)}</div>
    <div class="panel">{table(["Derived class", "Papers", "Share"], derived_rows)}</div>
  </div>

  <h2>Reviewer Scores</h2>
  <div class="grid">
    <div class="panel">{table(["Rating", "Reviews", "Share"], rating_rows)}</div>
    <div class="panel">{table(["Decision", "Reviews", "Avg rating", "Median"], decision_rating_rows)}</div>
    <div class="panel">{table(["Score type", "Count", "Average", "Median"], subscore_rows)}</div>
    <div class="panel">{table(["Text length", "Count", "Average", "Median"], length_rows)}</div>
  </div>

  <h2>Common Review Concerns</h2>
  <div class="panel">{table(["Concern area", "Reviews mentioning it", "Share of reviews"], concern_rows)}</div>

  <h2>Frequent Weakness / Question Terms</h2>
  <div class="grid">
    <div class="panel">{table(["Weakness term", "Count"], weakness_rows)}</div>
    <div class="panel">{table(["Question term", "Count"], question_rows)}</div>
  </div>

  <h2>Suggestions</h2>
  <div class="panel"><ul>{rec_items}</ul></div>
</main>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_doc, encoding="utf-8")
    return html_doc


def render_markdown(summary: dict, output_path: Path) -> None:
    paper_count = summary["paper_count"]
    review_count = summary["review_count"]
    lines = [
        "# Research Review Dataset Summary",
        "",
        f"- Papers: {paper_count:,}",
        f"- Reviewer reviews: {review_count:,}",
        "",
        "## Decisions",
        "",
        "| Decision | Papers | Share |",
        "|---|---:|---:|",
    ]
    for label, count in summary["decision_counts"].most_common():
        lines.append(f"| {label} | {count:,} | {pct(count, paper_count)} |")

    lines.extend(["", "## Derived Quality Classes", "", "| Class | Papers | Share |", "|---|---:|---:|"])
    labels = {"good_paper": "Good paper", "needs_modification": "Needs modification", "reject_risk": "Reject-risk"}
    for label, count in summary["derived_counts"].most_common():
        lines.append(f"| {labels.get(label, label)} | {count:,} | {pct(count, paper_count)} |")

    lines.extend(["", "## Suggestions", ""])
    for item in recommendations(summary):
        lines.append(f"- {item}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate charts and tables for the research review dataset.")
    parser.add_argument("--criteria", default=str(DEFAULT_CRITERIA_PATH))
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--html-output", default="reports/dataset_report.html")
    parser.add_argument("--md-output", default="reports/dataset_summary.md")
    args = parser.parse_args()

    data = load_data(Path(args.criteria))
    summary = summarize(data)
    render_html(summary, Path(args.html_output))
    render_markdown(summary, Path(args.md_output))
    print(f"Saved {args.html_output}")
    print(f"Saved {args.md_output}")


if __name__ == "__main__":
    main()
