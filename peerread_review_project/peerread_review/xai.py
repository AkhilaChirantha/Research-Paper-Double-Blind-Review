from __future__ import annotations

from peerread_review.features import FEATURE_NAMES


FEATURE_LABELS = {
    "average_recommendation": "Reviewer recommendation score",
    "average_confidence": "Reviewer confidence",
    "minimum_recommendation": "Lowest recommendation",
    "maximum_recommendation": "Highest recommendation",
    "total_word_count": "Paper length",
    "title_word_count": "Title specificity",
    "title_character_count": "Title length",
    "average_title_word_length": "Title readability",
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
    "has_introduction": "Introduction section",
    "has_related_work": "Related-work section",
    "has_background": "Background section",
    "has_methodology": "Methodology section",
    "has_experiments": "Experiment section",
    "has_results": "Results section",
    "has_discussion": "Discussion section",
    "has_conclusion": "Conclusion section",
    "has_appendix": "Appendix / supplementary material",
}


RECOMMENDATIONS = {
    "average_recommendation": "Address reviewer concerns that lowered recommendation scores.",
    "average_confidence": "Improve clarity so reviewers can judge the contribution with higher confidence.",
    "minimum_recommendation": "Find and fix the strongest negative reviewer concern.",
    "total_word_count": "Expand or tighten the manuscript so the contribution, method, evidence, and limitations are clear.",
    "title_word_count": "Revise the title so it clearly signals the task, method, and contribution.",
    "title_character_count": "Keep the title concise while preserving the main technical contribution.",
    "average_title_word_length": "Improve title readability with clearer, reviewer-friendly wording.",
    "abstract_word_count": "Improve the abstract with problem, method, key result, and contribution.",
    "introduction_word_count": "Expand the introduction to explain the research problem, gap, contribution, and motivation.",
    "methodology_word_count": "Clarify the methodology with reproducible details.",
    "experiments_word_count": "Strengthen experiments with datasets, metrics, and comparisons.",
    "results_word_count": "Add deeper result analysis and explain what the numbers mean.",
    "conclusion_word_count": "Add a concise conclusion that summarizes contribution, evidence, limitations, and future work.",
    "citation_like_count": "Improve related-work coverage with precise citations.",
    "numeric_result_count": "Add quantitative evidence, metrics, and uncertainty where possible.",
    "baseline_terms": "Compare against stronger and more explicit baselines.",
    "ablation_terms": "Add ablation or sensitivity analysis.",
    "reproducibility_terms": "Add implementation details, code/data notes, and hyperparameters.",
    "limitation_terms": "Add limitations, failure cases, and future work.",
    "novelty_terms": "Make the novelty and contribution claims more explicit.",
    "readability_sentence_words": "Improve readability by simplifying long sentences.",
    "has_introduction": "Add a clear introduction that states the problem, gap, contribution, and paper structure.",
    "has_related_work": "Add a related-work section comparing the closest prior studies.",
    "has_background": "Add background needed for reviewers to understand the technical setting.",
    "has_methodology": "Add a dedicated methodology section with enough detail to reproduce the approach.",
    "has_experiments": "Add an experiments/evaluation section with datasets, metrics, and baselines.",
    "has_results": "Add a results section that interprets the main quantitative and qualitative findings.",
    "has_discussion": "Add a discussion section explaining limitations, failure cases, and implications.",
    "has_conclusion": "Add a concise conclusion that restates contribution, evidence, limitations, and future work.",
    "has_appendix": "Add supplementary details only where they help reproducibility or reviewer verification.",
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
