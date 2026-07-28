from __future__ import annotations

import json
from pathlib import Path

from peerread_review.config import DEFAULT_DATASET_PATH
from peerread_review.data import parse_bool, read_peerread_rows


def target_feedback(row: dict) -> dict:
    accepted = parse_bool(row.get("accepted"))
    return {
        "decision": "Accept" if accepted else "Reject",
        "summary": row.get("meta_review") or row.get("combined_reviews") or "No review text available.",
        "strengths_or_evidence": "Use paper sections and reviewer evidence to justify the decision.",
        "required_modifications": [
            "Clarify contribution and novelty.",
            "Strengthen methodology and experiments.",
            "Improve limitations and reproducibility discussion.",
        ],
    }


def main() -> None:
    rows = read_peerread_rows(DEFAULT_DATASET_PATH)
    project_dir = Path(__file__).resolve().parent
    sft_dir = project_dir / "data" / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for row in rows:
        paper = {
            "title": row.get("title"),
            "abstract": row.get("abstract"),
            "introduction": row.get("introduction", "")[:2500],
            "methodology": row.get("methodology", "")[:2500],
            "experiments": row.get("experiments", "")[:2500],
            "reviews": row.get("combined_reviews", "")[:2500],
        }
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": "You are a double-blind research paper review assistant."},
                    {"role": "user", "content": json.dumps(paper, ensure_ascii=False)},
                    {"role": "assistant", "content": json.dumps(target_feedback(row), ensure_ascii=False)},
                ],
                "metadata": {
                    "paper_id": row.get("paper_id"),
                    "split": row.get("split"),
                    "accepted": parse_bool(row.get("accepted")),
                },
            }
        )
    train = [ex for ex in examples if ex["metadata"]["split"] == "train"]
    validation = [ex for ex in examples if ex["metadata"]["split"] in {"dev", "test"}]
    for name, data in [("sft_peerread_reviews.jsonl", examples), ("train.jsonl", train), ("validation.jsonl", validation)]:
        with (sft_dir / name).open("w", encoding="utf-8") as handle:
            for item in data:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    report = f"# PeerRead SFT Dataset\n\n- Total examples: {len(examples):,}\n- Train: {len(train):,}\n- Validation: {len(validation):,}\n"
    (sft_dir / "sft_dataset_report.md").write_text(report, encoding="utf-8")
    print(f"Saved SFT dataset to {sft_dir}")


if __name__ == "__main__":
    main()
