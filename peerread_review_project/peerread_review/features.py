from __future__ import annotations

import math
import re


NUMERIC_FEATURES = [
    "author_count",
    "title_word_count",
    "abstract_word_count",
    "introduction_word_count",
    "methodology_word_count",
    "experiments_word_count",
    "results_word_count",
    "conclusion_word_count",
    "section_count",
    "total_word_count",
    "review_count",
    "average_recommendation",
    "average_confidence",
    "recommendation_std",
    "confidence_std",
    "minimum_recommendation",
    "maximum_recommendation",
    "title_character_count",
    "average_title_word_length",
]

BOOLEAN_FEATURES = [
    "single_author",
    "multi_author",
    "large_collaboration",
    "contains_colon",
    "contains_question",
    "contains_dash",
    "has_introduction",
    "has_related_work",
    "has_background",
    "has_methodology",
    "has_experiments",
    "has_results",
    "has_discussion",
    "has_conclusion",
    "has_appendix",
]

TEXT_FEATURES = [
    "citation_like_count",
    "numeric_result_count",
    "baseline_terms",
    "ablation_terms",
    "reproducibility_terms",
    "limitation_terms",
    "novelty_terms",
    "readability_sentence_words",
]

FEATURE_NAMES = NUMERIC_FEATURES + BOOLEAN_FEATURES + TEXT_FEATURES


TERM_GROUPS = {
    "baseline_terms": ["baseline", "compare", "comparison", "state-of-the-art", "sota"],
    "ablation_terms": ["ablation", "sensitivity", "component analysis"],
    "reproducibility_terms": ["code", "dataset", "implementation", "hyperparameter", "reproduc"],
    "limitation_terms": ["limitation", "failure", "threat", "caveat", "future work"],
    "novelty_terms": ["novel", "new", "first", "propose", "contribution"],
}


def parse_float(value: object) -> float:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def parse_bool_float(value: object) -> float:
    return 1.0 if str(value).strip().lower() in {"true", "1", "yes"} else 0.0


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9_\-']+", text.lower())


def extract_text_features(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    joined = " ".join(tokens)
    sentences = [s for s in re.split(r"[.!?]+", text) if len(tokenize(s)) > 3]
    sentence_lengths = [len(tokenize(s)) for s in sentences[:300]]
    features = {
        "citation_like_count": float(len(re.findall(r"\[[0-9,\-\s]+\]|\([A-Z][A-Za-z\-]+ et al\.,? \d{4}\)", text))),
        "numeric_result_count": float(len(re.findall(r"\b\d+(?:\.\d+)?\s*%|\b\d+\.\d+\b", text))),
        "readability_sentence_words": float(sum(sentence_lengths) / max(len(sentence_lengths), 1)),
    }
    for name, terms in TERM_GROUPS.items():
        features[name] = float(sum(joined.count(term) for term in terms))
    return features


def extract_features(row: dict, text: str) -> dict[str, float]:
    features = {name: parse_float(row.get(name)) for name in NUMERIC_FEATURES}
    features.update({name: parse_bool_float(row.get(name)) for name in BOOLEAN_FEATURES})
    features.update(extract_text_features(text))
    return {name: features.get(name, 0.0) for name in FEATURE_NAMES}


def feature_vector(features: dict[str, float]) -> list[float]:
    return [math.log1p(max(float(features.get(name, 0.0)), 0.0)) for name in FEATURE_NAMES]
