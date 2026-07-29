from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from peerread_review.agent import local_prediction_for_openai, review_new_paper
from peerread_review.config import DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR
from peerread_review.model import load_model
from research_review.confidentiality import ConfidentialityMode, parse_mode, prepare_review_text
from research_review.io import read_document
from research_review.openai_reviewer import get_openai_recommendation


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a new paper with the PeerRead-trained AI agent.")
    parser.add_argument("paper", help="Path to .md, .txt, .tex, or .pdf paper")
    parser.add_argument("--use-openai", action="store_true", help="Add optional OpenAI detailed suggestions")
    parser.add_argument(
        "--confidentiality-mode",
        default=ConfidentialityMode.LOCAL_ONLY.value,
        choices=[mode.value for mode in ConfidentialityMode],
    )
    parser.add_argument("--json-output", default="new_paper_ai_review.json")
    args = parser.parse_args()

    paper_path = Path(args.paper)
    text = read_document(paper_path)
    mode = parse_mode(args.confidentiality_mode)
    review_text, audit = prepare_review_text(text, str(paper_path), mode)
    model = load_model(DEFAULT_MODEL_PATH)
    result = review_new_paper(review_text, paper_path.name, model)
    prediction = result["prediction"]
    output = {
        "paper": str(paper_path),
        "title": result["title"],
        "prediction": {key: value for key, value in prediction.items() if key not in {"text", "scaled_features"}},
        "agent_review": result["agent_review"],
        "xai": result["xai"],
        "confidentiality_audit": audit,
    }
    if args.use_openai:
        if not audit.get("api_allowed"):
            raise SystemExit(
                "OpenAI is blocked in local_only mode. Use abstract_only, section_summary_only, or full_paper_with_consent."
            )
        output["openai_review"] = get_openai_recommendation(review_text, local_prediction_for_openai(prediction))
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DEFAULT_REPORT_DIR / args.json_output
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
