from __future__ import annotations

import math
import re
from pathlib import Path

from peerread_review.features import BOOLEAN_FEATURES, NUMERIC_FEATURES, extract_features, tokenize
from peerread_review.model import predict
from peerread_review.xai import explain


SECTION_PATTERNS = {
    "abstract": r"(?is)(?:^|\n)\s*#{0,3}\s*abstract\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
    "introduction": r"(?is)(?:^|\n)\s*#{0,3}\s*(?:introduction|intro)\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
    "related_work": r"(?is)(?:^|\n)\s*#{0,3}\s*(?:related work|literature review|background)\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
    "methodology": r"(?is)(?:^|\n)\s*#{0,3}\s*(?:method|methods|methodology|approach|model)\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
    "experiments": r"(?is)(?:^|\n)\s*#{0,3}\s*(?:experiment|experiments|evaluation|experimental setup)\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
    "results": r"(?is)(?:^|\n)\s*#{0,3}\s*(?:results|analysis|discussion)\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
    "conclusion": r"(?is)(?:^|\n)\s*#{0,3}\s*(?:conclusion|conclusions|future work)\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
}

REVIEW_ONLY_FEATURES = {
    "review_count",
    "average_recommendation",
    "average_confidence",
    "recommendation_std",
    "confidence_std",
    "minimum_recommendation",
    "maximum_recommendation",
}


def row_from_text(text: str, title: str = "New Paper", model: dict | None = None) -> dict:
    sections = extract_sections(text)
    full_tokens = tokenize(text)
    row = {
        "paper_id": Path(title).stem or "new_paper",
        "title": clean_title(title, text),
        "abstract": sections.get("abstract", ""),
        "introduction": sections.get("introduction", ""),
        "related_work": sections.get("related_work", ""),
        "background": sections.get("related_work", ""),
        "methodology": sections.get("methodology", ""),
        "experiments": sections.get("experiments", ""),
        "results": sections.get("results", ""),
        "discussion": sections.get("results", ""),
        "conclusion": sections.get("conclusion", ""),
    }
    row.update(
        {
            "author_count": neutral_raw_value(model, "author_count", 1.0),
            "title_word_count": len(tokenize(row["title"])),
            "abstract_word_count": len(tokenize(row["abstract"])),
            "introduction_word_count": len(tokenize(row["introduction"])),
            "methodology_word_count": len(tokenize(row["methodology"])),
            "experiments_word_count": len(tokenize(row["experiments"])),
            "results_word_count": len(tokenize(row["results"])),
            "conclusion_word_count": len(tokenize(row["conclusion"])),
            "section_count": sum(1 for value in sections.values() if value.strip()),
            "total_word_count": len(full_tokens),
            "title_character_count": len(row["title"]),
            "average_title_word_length": average_word_length(row["title"]),
        }
    )
    for feature in REVIEW_ONLY_FEATURES:
        row[feature] = neutral_raw_value(model, feature, 0.0)
    row.update(
        {
            "single_author": "true",
            "multi_author": "false",
            "large_collaboration": "false",
            "contains_colon": "true" if ":" in row["title"] else "false",
            "contains_question": "true" if "?" in row["title"] else "false",
            "contains_dash": "true" if "-" in row["title"] else "false",
            "has_introduction": bool_text(row["introduction"]),
            "has_related_work": bool_text(row["related_work"]),
            "has_background": bool_text(row["related_work"]),
            "has_methodology": bool_text(row["methodology"]),
            "has_experiments": bool_text(row["experiments"]),
            "has_results": bool_text(row["results"]),
            "has_discussion": bool_text(row["results"]),
            "has_conclusion": bool_text(row["conclusion"]),
            "has_appendix": bool_text(re.search(r"(?im)^\s*#{0,3}\s*appendix\b", text)),
        }
    )
    for feature in NUMERIC_FEATURES + BOOLEAN_FEATURES:
        row[feature] = str(row.get(feature, 0))
    return row


def review_new_paper(text: str, title: str, model: dict) -> dict:
    row = row_from_text(text, title, model)
    prediction = predict(model, row)
    xai = explain(model, prediction)
    return {
        "title": row["title"],
        "prediction": prediction,
        "xai": xai,
        "agent_review": build_agent_review(prediction, xai),
        "screening_row": row,
    }


def build_agent_review(prediction: dict, xai: dict) -> dict:
    probabilities = {
        "accept": prediction["accept_probability"],
        "modify": round(1.0 - abs(prediction["accept_probability"] - 0.5) * 2, 4),
        "reject": prediction["reject_probability"],
    }
    positive = [
        factor
        for factor in xai.get("key_factors", [])
        if factor.get("direction") == "supports_accept" and abs(float(factor.get("contribution", 0))) > 0.01
    ][:5]
    risk = xai.get("risk_factors", [])[:6]
    good_points = [good_point_text(item) for item in positive] or [
        "The paper has enough structural evidence for an initial automated screening pass."
    ]
    weak_points = [weak_point_text(item) for item in risk]
    modify_points = xai.get("recommendations", [])[:6]
    verdict = prediction["decision"]
    if verdict == "Accept":
        summary = "The paper looks comparatively strong, but final acceptance still depends on polishing the reviewer-risk factors."
    elif verdict == "Modify":
        summary = "The paper is borderline and should be revised before submission; the highest-impact changes are listed below."
    else:
        summary = "The paper has high rejection risk in the current form; address the core weaknesses before submitting."
    return {
        "decision": verdict,
        "quality_score": round(prediction["accept_probability"] * 100, 1),
        "probabilities": probabilities,
        "overall_summary": summary,
        "good_points": unique(good_points),
        "weak_points": unique(weak_points),
        "must_modify": unique(modify_points),
        "acceptance_plan": acceptance_plan(verdict, modify_points),
    }


def local_prediction_for_openai(prediction: dict) -> dict:
    return {
        "verdict": {"Accept": "GOOD_PAPER", "Modify": "NEEDS_MODIFICATION", "Reject": "REJECT_RISK"}[
            prediction["decision"]
        ],
        "quality_score": round(prediction["accept_probability"] * 100, 1),
        "probabilities": {
            "good_paper": prediction["accept_probability"],
            "needs_modification": round(1.0 - abs(prediction["accept_probability"] - 0.5) * 2, 4),
            "reject_risk": prediction["reject_probability"],
        },
        "features": prediction.get("features", {}),
    }


def extract_sections(text: str) -> dict[str, str]:
    sections = {}
    for name, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, text)
        sections[name] = match.group(1).strip() if match else ""
    if not sections.get("abstract"):
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        sections["abstract"] = "\n\n".join(paragraphs[:2])
    return sections


def clean_title(title: str, text: str) -> str:
    fallback = Path(title).stem.replace("_", " ").strip()
    if fallback and fallback.lower() not in {"new paper", "paper"}:
        return fallback[:180]
    for line in text.splitlines()[:20]:
        line = line.strip(" #\t")
        if 8 <= len(line) <= 180 and not line.lower().startswith("abstract"):
            return line
    return "New Paper"


def neutral_raw_value(model: dict | None, feature: str, fallback: float) -> float:
    if not model:
        return fallback
    try:
        index = model["feature_names"].index(feature)
        scaled_mean = float(model["scaler"]["mean"][index])
        return max(math.exp(scaled_mean) - 1.0, 0.0)
    except (KeyError, ValueError, IndexError, TypeError):
        return fallback


def average_word_length(text: str) -> float:
    words = tokenize(text)
    return sum(len(word) for word in words) / max(len(words), 1)


def bool_text(value: object) -> str:
    return "true" if value else "false"


def good_point_text(item: dict) -> str:
    label = item.get("label", "A paper feature")
    return f"{label} currently supports the acceptance probability."


def weak_point_text(item: dict) -> str:
    label = item.get("label", "A paper feature")
    rec = item.get("recommendation", "Improve this area before submission.")
    return f"{label}: {rec}"


def acceptance_plan(verdict: str, recommendations: list[str]) -> list[str]:
    first = "Keep the current strengths visible in the abstract, introduction, and conclusion."
    if verdict == "Accept":
        first = "Polish the strongest sections and remove small reviewer doubts before submission."
    elif verdict == "Reject":
        first = "Do a major revision before submission, focusing first on evidence, novelty, and clarity."
    return unique([first, *recommendations, "Re-run the review after revision and compare the probability shift."])[:7]


def unique(items: list[str]) -> list[str]:
    seen = set()
    output = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output
