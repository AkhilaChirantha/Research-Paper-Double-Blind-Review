from __future__ import annotations

import csv
import sys
from pathlib import Path


TEXT_COLUMNS = [
    "title",
    "abstract",
    "introduction",
    "related_work",
    "background",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "meta_review",
    "combined_reviews",
]


def read_peerread_rows(path: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


def paper_text(row: dict) -> str:
    parts = []
    for column in TEXT_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            heading = column.replace("_", " ").title()
            parts.append(f"# {heading}\n\n{value}")
    return "\n\n".join(parts)


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "accepted", "accept"}
