from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


LABEL_TO_DECISION = {
    "good_paper": "ACCEPT",
    "needs_modification": "MODIFY",
    "reject_risk": "REJECT_RISK",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_text(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + " ...[truncated]"


def join_review_field(reviews: list[dict], field: str, max_items: int = 6) -> list[str]:
    values = []
    for review in reviews:
        text = compact_text(review.get(field, ""), 900)
        if text:
            values.append(text)
    return values[:max_items]


def make_user_prompt(paper: dict, max_paper_chars: int) -> str:
    stats = paper.get("review_stats", {})
    return "\n".join(
        [
            "Review this anonymized research paper in a double-blind style.",
            "",
            "Use the criteria: technical soundness, novelty, clarity, contribution, experiments, reproducibility, limitations, and ethics.",
            "",
            f"Paper ID: {paper['paper_id']}",
            f"Title: {paper['title']}",
            f"Abstract: {compact_text(paper.get('abstract', ''), 1800)}",
            f"Review statistics: average rating={stats.get('avg_rating')}, average soundness={stats.get('avg_soundness')}, average presentation={stats.get('avg_presentation')}, average contribution={stats.get('avg_contribution')}",
            "",
            "Anonymized paper text:",
            compact_text(paper.get("paper_text", ""), max_paper_chars),
        ]
    )


def make_assistant_feedback(paper: dict) -> str:
    reviews = paper.get("reviews", [])
    decision = LABEL_TO_DECISION.get(paper.get("derived_label"), "MODIFY")
    stats = paper.get("review_stats", {})

    summaries = join_review_field(reviews, "summary")
    strengths = join_review_field(reviews, "strengths")
    weaknesses = join_review_field(reviews, "weaknesses")
    questions = join_review_field(reviews, "questions")

    feedback = {
        "decision": decision,
        "confidence_basis": {
            "review_count": paper.get("review_count"),
            "average_rating": stats.get("avg_rating"),
            "average_soundness": stats.get("avg_soundness"),
            "average_presentation": stats.get("avg_presentation"),
            "average_contribution": stats.get("avg_contribution"),
        },
        "double_blind_summary": summaries,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "required_modifications": [
            {
                "area": "Technical clarity and positioning",
                "recommendation": "Clarify the central contribution, assumptions, and relationship to the closest prior work.",
            },
            {
                "area": "Evaluation and evidence",
                "recommendation": "Address reviewer concerns about baselines, ablations, robustness, and quantitative evidence.",
            },
            {
                "area": "Presentation and reproducibility",
                "recommendation": "Improve explanation quality, add missing implementation details, and make limitations explicit.",
            },
        ],
        "reviewer_questions_to_answer": questions,
        "natural_language_explanation": (
            "The decision is derived from the anonymized reviewer evidence and score pattern. "
            "High ratings with strong soundness/contribution indicate accept readiness, while repeated weaknesses, "
            "low soundness, limited evaluation, or unclear contribution indicate modification or reject-risk."
        ),
    }
    return json.dumps(feedback, indent=2, ensure_ascii=False)


def make_sft_example(paper: dict, max_paper_chars: int) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a double-blind academic research paper reviewer. "
                    "Give fair, constructive, paper-specific feedback without revealing author or reviewer identity."
                ),
            },
            {"role": "user", "content": make_user_prompt(paper, max_paper_chars)},
            {"role": "assistant", "content": make_assistant_feedback(paper)},
        ],
        "metadata": {
            "paper_id": paper["paper_id"],
            "original_decision": paper.get("original_decision"),
            "derived_label": paper.get("derived_label"),
            "review_count": paper.get("review_count"),
        },
    }


def split_rows(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    items = rows[:]
    random.Random(seed).shuffle(items)
    val_size = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
    return items[val_size:], items[:val_size]


def render_report(total: int, train: int, val: int, output_path: Path, max_paper_chars: int) -> None:
    lines = [
        "# SFT Dataset Creation Report",
        "",
        "This dataset is prepared for supervised fine-tuning of a double-blind research-paper reviewer model.",
        "",
        "## Output Files",
        "",
        "- `data/sft/sft_double_blind_reviews.jsonl`",
        "- `data/sft/train.jsonl`",
        "- `data/sft/validation.jsonl`",
        "",
        "## Counts",
        "",
        f"- Total examples: {total:,}",
        f"- Training examples: {train:,}",
        f"- Validation examples: {val:,}",
        f"- Max paper characters per prompt: {max_paper_chars:,}",
        "",
        "## Format",
        "",
        "Each line is a JSON object with `messages` in chat fine-tuning format.",
        "",
        "## Note",
        "",
        "The assistant target is created by aggregating all anonymized reviewer reviews of a paper into one double-blind feedback object.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SFT JSONL dataset from double-blind cleaned papers.")
    parser.add_argument("--input", default="data/processed/double_blind_papers.jsonl")
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument("--report", default="reports/sft_dataset_report.md")
    parser.add_argument("--max-paper-chars", type=int, default=8000)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    papers = read_jsonl(Path(args.input))
    if args.limit:
        papers = papers[: args.limit]

    examples = [make_sft_example(paper, args.max_paper_chars) for paper in papers]
    output_dir = Path(args.output_dir)
    train, val = split_rows(examples, args.val_ratio, args.seed)

    write_jsonl(output_dir / "sft_double_blind_reviews.jsonl", examples)
    write_jsonl(output_dir / "train.jsonl", train)
    write_jsonl(output_dir / "validation.jsonl", val)
    render_report(len(examples), len(train), len(val), Path(args.report), args.max_paper_chars)

    print(f"Saved {output_dir / 'sft_double_blind_reviews.jsonl'}")
    print(f"Saved {output_dir / 'train.jsonl'}")
    print(f"Saved {output_dir / 'validation.jsonl'}")
    print(f"Saved {args.report}")


if __name__ == "__main__":
    main()
