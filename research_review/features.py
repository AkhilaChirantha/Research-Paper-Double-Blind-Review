from __future__ import annotations

import math
import re
from collections import Counter


FEATURE_NAMES = [
    "word_count",
    "abstract_word_count",
    "section_count",
    "citation_count",
    "figure_count",
    "table_count",
    "equation_count",
    "numeric_result_count",
    "has_abstract",
    "has_introduction",
    "has_related_work",
    "has_method",
    "has_experiments",
    "has_results",
    "has_limitations",
    "has_ethics",
    "has_conclusion",
    "baseline_terms",
    "ablation_terms",
    "reproducibility_terms",
    "novelty_terms",
    "limitation_terms",
    "evaluation_terms",
    "avg_sentence_words",
]

SECTION_PATTERNS = {
    "has_abstract": r"(?im)^\s#{0,3}\s*abstract\b",
    "has_introduction": r"(?im)^\s#{1,3}\s*introduction\b",
    "has_related_work": r"(?im)^\s#{1,3}\s*(related work|background)\b",
    "has_method": r"(?im)^\s#{1,3}\s*(method|methodology|approach|model)\b",
    "has_experiments": r"(?im)^\s#{1,3}\s*(experiment|experiments|evaluation)\b",
    "has_results": r"(?im)^\s#{1,3}\s*(results|analysis)\b",
    "has_limitations": r"(?im)^\s#{1,3}\s*(limitation|limitations)\b",
    "has_ethics": r"(?im)^\s#{1,3}\s*(ethic|ethics|broader impact|societal impact)\b",
    "has_conclusion": r"(?im)^\s#{1,3}\s*(conclusion|discussion)\b",
}

TERM_GROUPS = {
    "baseline_terms": ["baseline", "compare", "comparison", "state-of-the-art", "sota"],
    "ablation_terms": ["ablation", "sensitivity", "component analysis"],
    "reproducibility_terms": ["code", "dataset", "implementation", "hyperparameter", "reproduc"],
    "novelty_terms": ["novel", "new", "first", "propose", "contribution"],
    "limitation_terms": ["limitation", "failure", "threat", "caveat", "future work"],
    "evaluation_terms": ["experiment", "evaluate", "benchmark", "metric", "accuracy", "performance"],
}


def extract_abstract(text: str) -> str:
    match = re.search(
        r"(?is)(?:^|\n)\s*#{0,3}\s*abstract\s*\n+(.*?)(?=\n\s*#{1,3}\s+\w|\Z)",
        text,
    )
    if match:
        return match.group(1).strip()
    return ""


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_\-']+", text.lower())


def count_terms(tokens: list[str], terms: list[str]) -> int:
    text = " ".join(tokens)
    return sum(text.count(term.lower()) for term in terms)


def extract_features(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    abstract = extract_abstract(text)
    abstract_tokens = tokenize(abstract)
    sentences = [s for s in re.split(r"[.!?]+", text) if len(tokenize(s)) > 3]
    sentence_lengths = [len(tokenize(s)) for s in sentences[:300]]

    features: dict[str, float] = {
        "word_count": float(len(tokens)),
        "abstract_word_count": float(len(abstract_tokens)),
        "section_count": float(len(re.findall(r"(?m)^\s*#{1,3}\s+\S", text))),
        "citation_count": float(
            len(re.findall(r"\[[0-9,\-\s]+\]|\([A-Z][A-Za-z\-]+ et al\.,? \d{4}\)", text))
        ),
        "figure_count": float(len(re.findall(r"(?i)\bfig(?:ure)?\.?\s*\d+", text))),
        "table_count": float(len(re.findall(r"(?i)\btable\s*\d+", text))),
        "equation_count": float(len(re.findall(r"\$\$|\\begin\{equation\}|\\\(|\\\[", text))),
        "numeric_result_count": float(len(re.findall(r"\b\d+(?:\.\d+)?\s*%|\b\d+\.\d+\b", text))),
        "avg_sentence_words": float(sum(sentence_lengths) / max(len(sentence_lengths), 1)),
    }

    for name, pattern in SECTION_PATTERNS.items():
        features[name] = 1.0 if re.search(pattern, text) else 0.0

    for name, terms in TERM_GROUPS.items():
        features[name] = float(count_terms(tokens, terms))

    return {name: features.get(name, 0.0) for name in FEATURE_NAMES}


def feature_vector(features: dict[str, float]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def summarize_feature_gaps(features: dict[str, float]) -> list[str]:
    gaps: list[str] = []
    required_sections = [
        ("has_abstract", "Add a clear Abstract section."),
        ("has_introduction", "Add a focused Introduction with problem, gap, and contribution."),
        ("has_method", "Clarify the method/approach enough for a reviewer to reproduce the idea."),
        ("has_experiments", "Add or strengthen the Experiments/Evaluation section."),
        ("has_results", "Show concrete results and analysis, not only claims."),
        ("has_conclusion", "Add a concise conclusion/discussion."),
    ]
    for key, message in required_sections:
        if features.get(key, 0.0) < 1:
            gaps.append(message)
    if features.get("baseline_terms", 0.0) < 2:
        gaps.append("Compare against stronger and more explicit baselines.")
    if features.get("ablation_terms", 0.0) < 1:
        gaps.append("Include ablations or sensitivity analysis for key design choices.")
    if features.get("citation_count", 0.0) < 8:
        gaps.append("Strengthen related-work coverage with more precise citations.")
    if features.get("numeric_result_count", 0.0) < 5:
        gaps.append("Add more quantitative evidence with metrics and uncertainty where possible.")
    if features.get("has_limitations", 0.0) < 1:
        gaps.append("Add limitations/failure cases to reduce reviewer risk.")
    return gaps[:8]


def top_words(text: str, n: int = 30) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "paper", "method", "model", "results", "using", "use", "can", "our", "we",
    }
    counts = Counter(t for t in tokenize(text) if t not in stop and len(t) > 3)
    return [word for word, _ in counts.most_common(n)]
