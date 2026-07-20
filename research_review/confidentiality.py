from __future__ import annotations

import re
from enum import Enum

from research_review.section_summary import summaries_to_screening_text, summarize_sections


class ConfidentialityMode(str, Enum):
    LOCAL_ONLY = "local_only"
    ABSTRACT_ONLY = "abstract_only"
    SECTION_SUMMARY_ONLY = "section_summary_only"
    FULL_PAPER_WITH_CONSENT = "full_paper_with_consent"


EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+")
ORCID_RE = re.compile(r"\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b")
AFFILIATION_RE = re.compile(
    r"(?im)^\s*(affiliation|affiliations|department|university|institute|institution|email|correspondence)\s*:?.*$"
)
ACK_RE = re.compile(
    r"(?is)\n#{1,3}\s*acknowledg(e)?ments?.*?(?=\n#{1,3}\s+\w|\Z)"
)


def parse_mode(value: str) -> ConfidentialityMode:
    try:
        return ConfidentialityMode(value)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in ConfidentialityMode)
        raise ValueError(f"Unknown confidentiality mode: {value}. Allowed: {allowed}") from exc


def mask_sensitive_text(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = URL_RE.sub("[URL]", text)
    text = ORCID_RE.sub("[ORCID]", text)
    text = AFFILIATION_RE.sub("[AFFILIATION REMOVED]", text)
    text = ACK_RE.sub("\n## Acknowledgements\n[REMOVED FOR CONFIDENTIAL REVIEW]", text)
    text = re.sub(r"(?im)^\s*\*\*(PDF|Forum):\*\*.*$", r"**\1:** [URL]", text)
    return text.strip()


def extract_abstract_text(text: str) -> str:
    match = re.search(
        r"(?is)(?:^|\n)\s*#{0,3}\s*abstract\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
        text,
    )
    if match:
        return "# Abstract\n\n" + match.group(1).strip()
    # Fallback for plain text or PDFs with weak heading extraction.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return "# Abstract / Opening Text\n\n" + "\n\n".join(paragraphs[:3])


def prepare_review_text(
    full_text: str,
    paper_path: str,
    mode: ConfidentialityMode,
) -> tuple[str, dict]:
    """Return text allowed for screening/API plus an audit record."""
    audit = {
        "mode": mode.value,
        "sent_full_paper": False,
        "masked_sensitive_text": True,
        "section_summaries_used": False,
        "abstract_only": False,
    }

    if mode == ConfidentialityMode.LOCAL_ONLY:
        audit["api_allowed"] = False
        return mask_sensitive_text(full_text), audit

    if mode == ConfidentialityMode.ABSTRACT_ONLY:
        audit["api_allowed"] = True
        audit["abstract_only"] = True
        return mask_sensitive_text(extract_abstract_text(full_text)), audit

    if mode == ConfidentialityMode.SECTION_SUMMARY_ONLY:
        audit["api_allowed"] = True
        audit["section_summaries_used"] = True
        summaries = summarize_sections(mask_sensitive_text(full_text))
        return summaries_to_screening_text(paper_path, summaries), {**audit, "sections": list(summaries.keys())}

    if mode == ConfidentialityMode.FULL_PAPER_WITH_CONSENT:
        audit["api_allowed"] = True
        audit["sent_full_paper"] = True
        return mask_sensitive_text(full_text), audit

    raise ValueError(f"Unhandled confidentiality mode: {mode}")


def mode_help() -> str:
    return (
        "Confidentiality modes: local_only=no external API; abstract_only=send only masked abstract; "
        "section_summary_only=send masked section summaries; full_paper_with_consent=send masked full paper."
    )
