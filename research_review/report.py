from __future__ import annotations

import json


def render_markdown(
    path: str,
    local_prediction: dict,
    ai_review: dict | None = None,
    xai_review: dict | None = None,
) -> str:
    lines = [
        f"# Blind Review Report",
        "",
        f"**Paper:** `{path}`",
        f"**Local verdict:** {local_prediction['verdict']}",
        f"**Local quality score:** {local_prediction['quality_score']}/100",
        "",
        "## Local ML Probabilities",
        "",
    ]
    for label, probability in local_prediction["probabilities"].items():
        lines.append(f"- `{label}`: {probability:.3f}")

    lines.extend(["", "## Structural Gaps", ""])
    gaps = local_prediction.get("feature_gaps") or []
    if gaps:
        lines.extend(f"- {gap}" for gap in gaps)
    else:
        lines.append("- No major structural gaps detected by the local model.")

    if xai_review:
        lines.extend(["", "## XAI Explanation", ""])
        lines.append(f"**Method:** {xai_review['method']}")
        lines.append("")
        lines.append("### Key Decision Factors")
        lines.append("")
        for item in xai_review["key_factors"]:
            direction = "supports" if item["direction"] == "supports_decision" else "raises risk for"
            lines.append(
                f"- **{item['label']}**: value `{item['value']}`, "
                f"contribution `{item['contribution']}` ({direction} the prediction)."
            )
        lines.extend(["", "### XAI-Based Recommendations", ""])
        lines.extend(f"- {recommendation}" for recommendation in xai_review["recommendations"])

    if ai_review:
        lines.extend(
            [
                "",
                "## OpenAI Blind Review Recommendation",
                "",
                f"**Final verdict:** {ai_review['final_verdict']}",
                f"**Confidence:** {ai_review['confidence']:.2f}",
                "",
                ai_review["overall_summary"],
                "",
                "### Main Reasons",
                "",
            ]
        )
        lines.extend(f"- {reason}" for reason in ai_review["main_reasons"])
        lines.extend(["", "### Section-Level Suggestions", ""])
        for item in ai_review["section_level_suggestions"]:
            lines.append(
                f"- **{item['section']}** ({item['priority']}): "
                f"{item['issue']} Recommendation: {item['recommendation']}"
            )
        lines.extend(["", "### Acceptance Plan", ""])
        lines.extend(f"- {step}" for step in ai_review["acceptance_plan"])
        lines.extend(["", "### Likely Reviewer Questions", ""])
        lines.extend(f"- {question}" for question in ai_review["reviewer_questions"])
    else:
        lines.extend(
            [
                "",
                "## OpenAI Blind Review Recommendation",
                "",
                "Not requested. Re-run with `--use-openai` after setting `OPENAI_API_KEY` in `.env`.",
            ]
        )

    lines.extend(["", "## Raw Local Features", "", "```json"])
    lines.append(json.dumps(local_prediction["features"], indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"
