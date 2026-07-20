from __future__ import annotations

import json
from pathlib import Path

from research_review.config import load_env, openai_model
from research_review.features import extract_abstract


REVIEW_SCHEMA = {
    "type": "json_schema",
    "name": "blind_review_recommendation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "final_verdict": {
                "type": "string",
                "enum": ["GOOD_PAPER", "NEEDS_MODIFICATION", "REJECT_RISK"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "overall_summary": {"type": "string"},
            "main_reasons": {"type": "array", "items": {"type": "string"}},
            "section_level_suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "section": {"type": "string"},
                        "issue": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                    },
                    "required": ["section", "issue", "recommendation", "priority"],
                },
            },
            "acceptance_plan": {"type": "array", "items": {"type": "string"}},
            "reviewer_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "final_verdict",
            "confidence",
            "overall_summary",
            "main_reasons",
            "section_level_suggestions",
            "acceptance_plan",
            "reviewer_questions",
        ],
    },
}


def trim_for_review(text: str, max_chars: int = 55000) -> str:
    if len(text) <= max_chars:
        return text
    abstract = extract_abstract(text)
    head = text[: int(max_chars * 0.55)]
    tail = text[-int(max_chars * 0.35) :]
    return (
        f"{abstract}\n\n"
        "[Document shortened for API review: beginning and ending retained.]\n\n"
        f"{head}\n\n[...]\n\n{tail}"
    )


def get_openai_recommendation(
    text: str,
    local_prediction: dict,
    env_path: Path = Path(".env"),
    max_chars: int = 55000,
) -> dict:
    load_env(env_path)
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK missing. Run: pip install -r requirements.txt") from exc

    client = OpenAI()
    model = openai_model()
    prompt = {
        "task": "Act as an anonymous NeurIPS-style blind reviewer before submission.",
        "local_ml_prediction": local_prediction,
        "criteria": [
            "technical soundness",
            "novel contribution",
            "clarity and organization",
            "strength of experiments/evaluation",
            "baselines and ablations",
            "reproducibility",
            "limitations and ethics",
        ],
        "paper_text": trim_for_review(text, max_chars=max_chars),
    }

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a strict but constructive blind-review assistant. "
                    "Give decision support, not a guarantee. Be specific about what to change."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        text={"format": REVIEW_SCHEMA},
    )
    return json.loads(response.output_text)
