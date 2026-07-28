from __future__ import annotations

import json

from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_REPORT_DIR
from peerread_review.data import read_peerread_rows


SECTION_COLUMNS = ["abstract", "introduction", "related_work", "methodology", "experiments", "results", "discussion", "conclusion"]


def summarize_text(text: str, max_words: int = 90) -> str:
    words = " ".join((text or "").split()).split()
    if not words:
        return ""
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def main() -> None:
    rows = []
    for row in read_peerread_rows(DEFAULT_DATASET_PATH):
        summaries = {column: summarize_text(row.get(column, "")) for column in SECTION_COLUMNS if row.get(column)}
        rows.append(
            {
                "paper_id": row.get("paper_id"),
                "title": row.get("title"),
                "accepted": row.get("accepted"),
                "sections_found": list(summaries),
                "section_summaries": summaries,
            }
        )
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = DEFAULT_REPORT_DIR / "section_summaries.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
