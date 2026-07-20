from __future__ import annotations

import math

from research_review.features import FEATURE_NAMES


FEATURE_LABELS = {
    "word_count": "Paper length / content volume",
    "abstract_word_count": "Abstract detail",
    "section_count": "Section structure",
    "citation_count": "Related-work citation coverage",
    "figure_count": "Use of figures",
    "table_count": "Use of tables",
    "equation_count": "Technical formulation",
    "numeric_result_count": "Quantitative evidence",
    "has_abstract": "Abstract section",
    "has_introduction": "Introduction section",
    "has_related_work": "Related work / background",
    "has_method": "Method section",
    "has_experiments": "Experiments / evaluation section",
    "has_results": "Results / analysis section",
    "has_limitations": "Limitations discussion",
    "has_ethics": "Ethics / broader impact discussion",
    "has_conclusion": "Conclusion / discussion section",
    "baseline_terms": "Baseline comparisons",
    "ablation_terms": "Ablation / sensitivity analysis",
    "reproducibility_terms": "Reproducibility evidence",
    "novelty_terms": "Novelty / contribution framing",
    "limitation_terms": "Limitation and failure-case wording",
    "evaluation_terms": "Evaluation terminology",
    "avg_sentence_words": "Sentence complexity",
}


FEATURE_RECOMMENDATIONS = {
    "word_count": "Expand the manuscript enough to clearly cover motivation, method, evidence, and limitations.",
    "abstract_word_count": "Rewrite the abstract with the problem, method, key result, and contribution in a compact form.",
    "section_count": "Improve the paper structure with clear sections and reviewer-friendly headings.",
    "citation_count": "Strengthen the related work section with precise citations and comparisons.",
    "figure_count": "Add figures that explain the method, pipeline, or important experimental trends.",
    "table_count": "Add tables for datasets, baselines, ablations, and key quantitative comparisons.",
    "equation_count": "Clarify the technical formulation with definitions, assumptions, or equations where needed.",
    "numeric_result_count": "Add more quantitative results, metrics, and uncertainty/error analysis.",
    "has_abstract": "Add a clear abstract before submission.",
    "has_introduction": "Add a focused introduction that states the problem, research gap, and contributions.",
    "has_related_work": "Add or strengthen related work and position the contribution against prior methods.",
    "has_method": "Clarify the method enough for a reviewer to understand and reproduce the approach.",
    "has_experiments": "Add or strengthen experiments with datasets, metrics, baselines, and evaluation protocol.",
    "has_results": "Add a results and analysis section that interprets the evidence, not only reports numbers.",
    "has_limitations": "Add limitations and failure cases to reduce reviewer risk.",
    "has_ethics": "Add ethics, broader impact, or responsible-use discussion if relevant to the domain.",
    "has_conclusion": "Add a concise conclusion or discussion section.",
    "baseline_terms": "Compare against explicit and strong baselines, including recent or standard methods.",
    "ablation_terms": "Add ablation or sensitivity studies to justify the main design choices.",
    "reproducibility_terms": "Include implementation details, hyperparameters, code/data availability, or reproducibility notes.",
    "novelty_terms": "Make the novelty and contribution claims more explicit and easier for reviewers to locate.",
    "limitation_terms": "Discuss limitations, failure cases, and future work more openly.",
    "evaluation_terms": "Make the evaluation protocol and metrics more explicit.",
    "avg_sentence_words": "Improve readability by shortening long sentences and simplifying dense explanations.",
}


def explain_prediction(model: dict, prediction: dict, top_n: int = 8) -> dict:
    """Create a local XAI explanation using feature-distance contributions.

    This is a lightweight, dependency-free XAI layer. It explains why the trained
    probabilistic model preferred the predicted class by comparing each feature's
    fit to the predicted class against its fit to the other classes.
    """
    features = prediction.get("features", {})
    probabilities = prediction.get("probabilities", {})
    predicted_label = max(probabilities, key=probabilities.get)
    predicted_class = model["classes"][predicted_label]
    other_labels = [label for label in model["labels"] if label != predicted_label]

    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        value = float(features.get(name, 0.0))
        transformed = math.log1p(value)
        pred_mean = float(predicted_class["mean"][index])
        pred_var = max(float(predicted_class["var"][index]), 1e-6)
        pred_distance = ((transformed - pred_mean) ** 2) / pred_var

        other_distances = []
        for label in other_labels:
            cls = model["classes"][label]
            mean = float(cls["mean"][index])
            var = max(float(cls["var"][index]), 1e-6)
            other_distances.append(((transformed - mean) ** 2) / var)
        other_distance = sum(other_distances) / max(len(other_distances), 1)

        contribution = other_distance - pred_distance
        rows.append(
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name.replace("_", " ").title()),
                "value": round(value, 3),
                "contribution": round(contribution, 4),
                "direction": "supports_decision" if contribution >= 0 else "raises_risk",
                "recommendation": FEATURE_RECOMMENDATIONS.get(name, "Improve this aspect before submission."),
            }
        )

    rows.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
    key_factors = rows[:top_n]
    risk_factors = [item for item in rows if item["direction"] == "raises_risk"][:top_n]
    supporting_factors = [item for item in rows if item["direction"] == "supports_decision"][:top_n]

    return {
        "method": "Local XAI feature-distance explanation",
        "default_model": "XAI_LOCAL",
        "predicted_label": predicted_label,
        "verdict": prediction.get("verdict"),
        "quality_score": prediction.get("quality_score"),
        "key_factors": key_factors,
        "supporting_factors": supporting_factors,
        "risk_factors": risk_factors,
        "recommendations": build_xai_recommendations(prediction, risk_factors),
        "note": (
            "This XAI layer explains the local screening model. OpenAI is optional "
            "and should be used only when more detailed natural-language feedback is needed."
        ),
    }


def build_xai_recommendations(prediction: dict, risk_factors: list[dict], limit: int = 6) -> list[str]:
    recommendations = []
    for gap in prediction.get("feature_gaps", []):
        if gap not in recommendations:
            recommendations.append(gap)
    for item in risk_factors:
        suggestion = item.get("recommendation")
        if suggestion and suggestion not in recommendations:
            recommendations.append(str(suggestion))
    if not recommendations:
        recommendations.append("The local XAI layer found no major structural risk; polish clarity and reviewer framing.")
    return recommendations[:limit]
