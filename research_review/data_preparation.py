from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from research_review.config import DEFAULT_CRITERIA_PATH, DEFAULT_PAPERS_DIR
from research_review.features import extract_abstract
from research_review.io import read_document, read_json, write_json
from research_review.model import paper_training_label, parse_score


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+")
OPENREVIEW_RE = re.compile(r"openreview\.net/\S+", re.IGNORECASE)
REVIEWER_RE = re.compile(r"Reviewer_[A-Za-z0-9]+|Reviewer: Reviewer_[A-Za-z0-9]+")


def clean_text(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = OPENREVIEW_RE.sub("[URL]", text)
    text = REVIEWER_RE.sub("Anonymous Reviewer", text)
    text = re.sub(r"\s+\n", "\n", text)
    return text.strip()


def is_real_review(review: dict) -> bool:
    if review.get("reviewer_id") == "Authors":
        return False
    rating = parse_score(review.get("rating"))
    text_fields = [
        review.get("summary") or "",
        review.get("strengths") or "",
        review.get("weaknesses") or "",
        review.get("questions") or "",
    ]
    has_text = any(len(field.strip()) >= 40 for field in text_fields)
    return rating is not None and has_text


def anonymize_review(review: dict, index: int) -> dict:
    return {
        "reviewer": f"Anonymous Reviewer {index}",
        "rating": parse_score(review.get("rating")),
        "confidence": parse_score(review.get("confidence")),
        "soundness": parse_score(review.get("soundness")),
        "presentation": parse_score(review.get("presentation")),
        "contribution": parse_score(review.get("contribution")),
        "summary": clean_text(review.get("summary") or ""),
        "strengths": clean_text(review.get("strengths") or ""),
        "weaknesses": clean_text(review.get("weaknesses") or ""),
        "questions": clean_text(review.get("questions") or ""),
    }


def strip_review_blocks(markdown: str) -> str:
    review_start = re.search(r"(?m)^## Review\s+\d+", markdown)
    if review_start:
        return markdown[: review_start.start()].strip()
    return markdown.strip()


def anonymize_paper_text(markdown: str) -> str:
    text = strip_review_blocks(markdown)
    text = re.sub(r"\*\*PDF:\*\*.*", "**PDF:** [URL]", text)
    text = re.sub(r"\*\*Forum:\*\*.*", "**Forum:** [URL]", text)
    text = clean_text(text)
    # A conservative acknowledgement mask; exact author names are not available in this dataset.
    text = re.sub(
        r"(?is)\n#{1,3}\s*acknowledg(e)?ments?.*?(?=\n#{1,3}\s+\w|\Z)",
        "\n## Acknowledgements\n[REMOVED FOR DOUBLE-BLIND REVIEW]",
        text,
    )
    return text


def aggregate_review_stats(reviews: list[dict]) -> dict:
    keys = ["rating", "confidence", "soundness", "presentation", "contribution"]
    stats = {}
    for key in keys:
        values = [review[key] for review in reviews if review.get(key) is not None]
        stats[f"avg_{key}"] = round(sum(values) / len(values), 3) if values else None
        stats[f"min_{key}"] = min(values) if values else None
        stats[f"max_{key}"] = max(values) if values else None
    return stats


def prepare_dataset(criteria_path: Path, papers_dir: Path) -> tuple[list[dict], list[dict], dict]:
    metadata = read_json(criteria_path)
    cleaned_papers = []
    review_rows = []
    counters = Counter()

    for paper in metadata:
        counters["metadata_papers"] += 1
        paper_id = paper.get("paper_id", "")
        paper_path = papers_dir / f"{paper_id}.md"
        if not paper_path.exists():
            counters["missing_markdown"] += 1
            continue

        real_reviews = [review for review in paper.get("reviews", []) if is_real_review(review)]
        if not real_reviews:
            counters["no_real_review"] += 1
            continue

        markdown = read_document(paper_path)
        anonymized_reviews = [anonymize_review(review, i + 1) for i, review in enumerate(real_reviews)]
        paper_text = anonymize_paper_text(markdown)
        abstract = clean_text(paper.get("abstract") or extract_abstract(markdown))
        derived_label = paper_training_label({**paper, "reviews": real_reviews})
        review_stats = aggregate_review_stats(anonymized_reviews)

        clean_paper = {
            "paper_id": paper_id,
            "source": paper.get("source"),
            "year": paper.get("year"),
            "title": clean_text(paper.get("title") or ""),
            "abstract": abstract,
            "keywords": paper.get("keywords", ""),
            "original_decision": paper.get("decision", ""),
            "derived_label": derived_label,
            "review_count": len(anonymized_reviews),
            "review_stats": review_stats,
            "paper_text": paper_text,
            "reviews": anonymized_reviews,
        }
        cleaned_papers.append(clean_paper)
        counters["cleaned_papers"] += 1
        counters[f"derived_{derived_label}"] += 1

        for review in anonymized_reviews:
            review_rows.append(
                {
                    "paper_id": paper_id,
                    "title": clean_paper["title"],
                    "original_decision": clean_paper["original_decision"],
                    "derived_label": derived_label,
                    "reviewer": review["reviewer"],
                    "rating": review["rating"],
                    "confidence": review["confidence"],
                    "soundness": review["soundness"],
                    "presentation": review["presentation"],
                    "contribution": review["contribution"],
                    "summary": review["summary"],
                    "strengths": review["strengths"],
                    "weaknesses": review["weaknesses"],
                    "questions": review["questions"],
                }
            )

    report = {
        "criteria_path": str(criteria_path),
        "papers_dir": str(papers_dir),
        "counts": dict(counters),
    }
    return cleaned_papers, review_rows, report


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def render_report(report: dict, path: Path) -> None:
    counts = report["counts"]
    lines = [
        "# Dataset Cleaning Report",
        "",
        f"- Source metadata: `{report['criteria_path']}`",
        f"- Papers directory: `{report['papers_dir']}`",
        "",
        "## Counts",
        "",
        f"- Metadata papers: {counts.get('metadata_papers', 0):,}",
        f"- Missing markdown files: {counts.get('missing_markdown', 0):,}",
        f"- Papers with no real reviewer review: {counts.get('no_real_review', 0):,}",
        f"- Cleaned papers kept: {counts.get('cleaned_papers', 0):,}",
        "",
        "## Derived Labels",
        "",
        f"- Good paper: {counts.get('derived_good_paper', 0):,}",
        f"- Needs modification: {counts.get('derived_needs_modification', 0):,}",
        f"- Reject-risk: {counts.get('derived_reject_risk', 0):,}",
        "",
        "## Cleaning Rules",
        "",
        "- Removed author-only rows.",
        "- Kept only reviews with numeric reviewer rating and useful text.",
        "- Masked emails, URLs, OpenReview links, and reviewer IDs.",
        "- Removed review blocks from paper text to preserve manuscript-only input.",
        "- Masked acknowledgement sections when detected.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean OpenReview data and create double-blind dataset files.")
    parser.add_argument("--criteria", default=str(DEFAULT_CRITERIA_PATH))
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--report", default="reports/dataset_cleaning_report.md")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cleaned_papers, review_rows, report = prepare_dataset(Path(args.criteria), Path(args.papers_dir))

    write_json(output_dir / "clean_papers.json", {"papers": cleaned_papers, "report": report})
    write_jsonl(output_dir / "double_blind_papers.jsonl", cleaned_papers)
    write_csv(output_dir / "clean_reviews.csv", review_rows)
    render_report(report, Path(args.report))

    print(f"Saved {output_dir / 'clean_papers.json'}")
    print(f"Saved {output_dir / 'double_blind_papers.jsonl'}")
    print(f"Saved {output_dir / 'clean_reviews.csv'}")
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
