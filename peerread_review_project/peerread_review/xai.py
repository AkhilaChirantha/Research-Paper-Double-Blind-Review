from __future__ import annotations

from peerread_review.features import FEATURE_NAMES


FEATURE_LABELS = {
    "average_recommendation": "Reviewer recommendation score",
    "average_confidence": "Reviewer confidence",
    "minimum_recommendation": "Lowest recommendation",
    "maximum_recommendation": "Highest recommendation",
    "total_word_count": "Paper length",
    "abstract_word_count": "Abstract detail",
    "introduction_word_count": "Introduction detail",
    "methodology_word_count": "Methodology detail",
    "experiments_word_count": "Experiment detail",
    "results_word_count": "Result detail",
    "citation_like_count": "Citation coverage",
    "numeric_result_count": "Quantitative evidence",
    "baseline_terms": "Baseline comparison evidence",
    "ablation_terms": "Ablation evidence",
    "reproducibility_terms": "Reproducibility evidence",
    "limitation_terms": "Limitation discussion",
    "novelty_terms": "Novelty framing",
    "readability_sentence_words": "Sentence complexity",
}


RECOMMENDATIONS = {
    "average_recommendation": "Address reviewer concerns that lowered recommendation scores.",
    "average_confidence": "Improve clarity so reviewers can judge the contribution with higher confidence.",
    "minimum_recommendation": "Find and fix the strongest negative reviewer concern.",
    "total_word_count": "Expand or tighten the manuscript so the contribution, method, evidence, and limitations are clear.",
    "abstract_word_count": "Improve the abstract with problem, method, key result, and contribution.",
    "methodology_word_count": "Clarify the methodology with reproducible details.",
    "experiments_word_count": "Strengthen experiments with datasets, metrics, and comparisons.",
    "results_word_count": "Add deeper result analysis and explain what the numbers mean.",
    "citation_like_count": "Improve related-work coverage with precise citations.",
    "numeric_result_count": "Add quantitative evidence, metrics, and uncertainty where possible.",
    "baseline_terms": "Compare against stronger and more explicit baselines.",
    "ablation_terms": "Add ablation or sensitivity analysis.",
    "reproducibility_terms": "Add implementation details, code/data notes, and hyperparameters.",
    "limitation_terms": "Add limitations, failure cases, and future work.",
    "novelty_terms": "Make the novelty and contribution claims more explicit.",
    "readability_sentence_words": "Improve readability by simplifying long sentences.",
}


def explain(model: dict, prediction: dict, top_n: int = 6) -> dict:
    features = prediction["features"]
    scaled = prediction.get("scaled_features") or []
    weights = model.get("weights") or [0.0 for _ in FEATURE_NAMES]
    rows = []
    for index, feature in enumerate(FEATURE_NAMES):
        contribution = float(weights[index]) * float(scaled[index]) if index < len(scaled) else 0.0
        direction = "supports_accept" if contribution >= 0 else "supports_reject"
        rows.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
                "value": round(float(features.get(feature, 0.0)), 3),
                "contribution": round(contribution, 4),
                "direction": direction,
                "recommendation": RECOMMENDATIONS.get(feature, "Improve this paper aspect before submission."),
            }
        )
    rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    if prediction.get("decision") == "Accept":
        risk_rows = [item for item in rows if item["direction"] == "supports_reject"][:top_n]
    else:
        risk_rows = [item for item in rows if item["direction"] == "supports_reject"][:top_n]
    if not risk_rows:
        risk_rows = rows[:top_n]
    recommendations = []
    for item in risk_rows:
        rec = item["recommendation"]
        if rec not in recommendations:
            recommendations.append(rec)
    return {
        "method": "PeerRead local XAI feature-distance explanation",
        "key_factors": rows[:top_n],
        "risk_factors": risk_rows,
        "recommendations": recommendations[:5],
    }
