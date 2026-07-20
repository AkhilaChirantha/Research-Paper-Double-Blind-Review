from __future__ import annotations

import argparse
from pathlib import Path

from research_review.config import DEFAULT_CRITERIA_PATH, DEFAULT_MODEL_PATH, DEFAULT_PAPERS_DIR
from research_review.confidentiality import ConfidentialityMode, mode_help, parse_mode, prepare_review_text
from research_review.io import read_document, write_json
from research_review.model import load_model, predict_with_model, train
from research_review.openai_reviewer import get_openai_recommendation
from research_review.report import render_markdown


def ensure_model(path: Path) -> dict:
    if not path.exists():
        return train(DEFAULT_CRITERIA_PATH, DEFAULT_PAPERS_DIR, path)
    return load_model(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind-review a research paper before submission.")
    parser.add_argument("paper", help="Path to a .md, .txt, .tex, or .pdf paper")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API for detailed recommendations")
    parser.add_argument("--confidentiality-mode", default=ConfidentialityMode.LOCAL_ONLY.value, help=mode_help())
    parser.add_argument("--section-summary", action="store_true", help="Shortcut for --confidentiality-mode section_summary_only")
    parser.add_argument("--json-output", help="Optional JSON output path")
    parser.add_argument("--md-output", help="Optional Markdown report output path")
    args = parser.parse_args()

    paper_path = Path(args.paper)
    model_path = Path(args.model_path)
    model = ensure_model(model_path)
    text = read_document(paper_path)
    mode = parse_mode(args.confidentiality_mode)
    if args.section_summary:
        mode = ConfidentialityMode.SECTION_SUMMARY_ONLY
    review_text, confidentiality_audit = prepare_review_text(text, str(paper_path), mode)
    if args.use_openai and not confidentiality_audit.get("api_allowed"):
        raise SystemExit(
            "OpenAI review is blocked in local_only mode. "
            "Use --confidentiality-mode abstract_only, section_summary_only, or full_paper_with_consent."
        )
    local_prediction = predict_with_model(model, review_text)
    ai_review = get_openai_recommendation(review_text, local_prediction) if args.use_openai else None

    result = {
        "paper": str(paper_path),
        "confidentiality": confidentiality_audit,
        "local_prediction": local_prediction,
        "openai_review": ai_review,
    }

    if args.json_output:
        write_json(Path(args.json_output), result)

    report = render_markdown(str(paper_path), local_prediction, ai_review)
    if args.md_output:
        out = Path(args.md_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
