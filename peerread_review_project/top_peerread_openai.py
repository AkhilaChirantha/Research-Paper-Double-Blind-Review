from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from peerread_review.config import DEFAULT_REPORT_DIR
from peerread_review.data import paper_text, read_peerread_rows
from peerread_review.model import load_model, predict
from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_review.openai_reviewer import get_openai_recommendation


def local_prediction_for_openai(prediction: dict) -> dict:
    verdict = {
        "Accept": "GOOD_PAPER",
        "Modify": "NEEDS_MODIFICATION",
        "Reject": "REJECT_RISK",
    }[prediction["decision"]]
    return {
        "verdict": verdict,
        "quality_score": round(100 * prediction["accept_probability"], 1),
        "probabilities": {
            "good_paper": prediction["accept_probability"],
            "needs_modification": 1.0 - abs(prediction["accept_probability"] - 0.5) * 2,
            "reject_risk": prediction["reject_probability"],
        },
        "features": prediction["features"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional OpenAI feedback for selected PeerRead papers.")
    parser.add_argument("--per-group", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=12000)
    args = parser.parse_args()
    rows = read_peerread_rows(DEFAULT_DATASET_PATH)
    model = load_model(DEFAULT_MODEL_PATH)
    scored = []
    for row in rows:
        prediction = predict(model, row)
        scored.append((row, prediction))
    groups = {
        "Top Accept": sorted(scored, key=lambda item: item[1]["accept_probability"], reverse=True)[: args.per_group],
        "Top Modify": sorted(scored, key=lambda item: abs(item[1]["accept_probability"] - 0.5))[: args.per_group],
        "Top Reject": sorted(scored, key=lambda item: item[1]["accept_probability"])[: args.per_group],
    }
    output = {"per_group": args.per_group, "papers": []}
    for group, items in groups.items():
        for row, prediction in items:
            review_text = paper_text(row)[: args.max_chars]
            ai_review = get_openai_recommendation(review_text, local_prediction_for_openai(prediction), max_chars=args.max_chars)
            output["papers"].append(
                {
                    "group": group,
                    "paper_id": row.get("paper_id"),
                    "title": row.get("title"),
                    "actual_label": row.get("accepted"),
                    "local_decision": prediction["decision"],
                    "accept_probability": prediction["accept_probability"],
                    "ai_review": ai_review,
                }
            )
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = DEFAULT_REPORT_DIR / "peerread_openai_reviews.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
