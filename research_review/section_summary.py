from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from research_review.config import DEFAULT_PAPERS_DIR
from research_review.io import read_document


SECTION_ALIASES = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "background"],
    "related_work": ["related work", "literature review", "prior work", "background"],
    "method": ["method", "methods", "methodology", "approach", "model", "framework"],
    "experiments": ["experiment", "experiments", "evaluation", "experimental setup"],
    "results": ["results", "analysis", "discussion"],
    "limitations": ["limitation", "limitations", "failure cases", "broader impact", "ethics"],
    "conclusion": ["conclusion", "conclusions", "future work"],
}


def normalize_heading(text: str) -> str:
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text.strip().lower())
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def section_key_for_heading(heading: str) -> str | None:
    normalized = normalize_heading(heading)
    for key, aliases in SECTION_ALIASES.items():
        if any(alias == normalized or normalized.startswith(alias + " ") for alias in aliases):
            return key
    return None


def extract_markdown_sections(text: str) -> dict[str, str]:
    heading_re = re.compile(r"(?m)^(#{1,3})\s+(.+?)\s*$")
    matches = list(heading_re.finditer(text))
    sections: dict[str, str] = {}

    if not matches:
        sections["full_text"] = text
        return sections

    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        key = section_key_for_heading(heading)
        if key:
            content = text[start:end].strip()
            if content:
                sections.setdefault(key, "")
                sections[key] = (sections[key] + "\n\n" + content).strip()

    if "abstract" not in sections:
        abstract_match = re.search(r"(?is)\babstract\b\s*\n+(.*?)(?=\n\s*(?:#{1,3}\s+)?(?:introduction|1\s+introduction)\b|\Z)", text)
        if abstract_match:
            sections["abstract"] = abstract_match.group(1).strip()
    return sections


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [sentence.strip() for sentence in sentences if len(sentence.split()) >= 5]


def summarize_text(text: str, max_sentences: int = 4, max_chars: int = 1200) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text[:max_chars].strip()

    selected = []
    for sentence in sentences:
        selected.append(sentence)
        if len(selected) >= max_sentences or len(" ".join(selected)) >= max_chars:
            break
    summary = " ".join(selected)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 15].rstrip() + " ...[shortened]"
    return summary


def summarize_sections(text: str) -> dict[str, str]:
    sections = extract_markdown_sections(text)
    summaries = {}
    for key in SECTION_ALIASES:
        content = sections.get(key, "")
        if content:
            summaries[key] = summarize_text(content)
    if not summaries:
        summaries["full_text"] = summarize_text(sections.get("full_text", text), max_sentences=8, max_chars=2400)
    return summaries


def render_summary_document(path: str, summaries: dict[str, str]) -> str:
    lines = [
        "# Section Summary for Screening",
        "",
        f"Paper: `{path}`",
        "",
    ]
    for key, summary in summaries.items():
        title = key.replace("_", " ").title()
        lines.extend([f"## {title}", "", summary, ""])
    return "\n".join(lines).strip() + "\n"


def summaries_to_screening_text(path: str, summaries: dict[str, str]) -> str:
    return render_summary_document(path, summaries)


def iter_paper_paths(papers_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(papers_dir.glob("*.md"))
    return paths[:limit] if limit else paths


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_section_summary_dataset(papers_dir: Path, limit: int | None = None) -> list[dict]:
    rows = []
    for paper_path in iter_paper_paths(papers_dir, limit):
        text = read_document(paper_path)
        summaries = summarize_sections(text)
        rows.append(
            {
                "paper_id": paper_path.stem,
                "path": str(paper_path),
                "sections_found": list(summaries.keys()),
                "section_summaries": summaries,
                "screening_text": summaries_to_screening_text(str(paper_path), summaries),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate section summaries before paper screening.")
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--output", default="data/processed/section_summaries.jsonl")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = build_section_summary_dataset(Path(args.papers_dir), args.limit)
    write_jsonl(Path(args.output), rows)
    print(f"Saved {args.output}")
    print(f"Papers summarized: {len(rows)}")


if __name__ == "__main__":
    main()
