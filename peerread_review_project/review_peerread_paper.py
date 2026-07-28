from __future__ import annotations

import argparse
import json

from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR
from peerread_review.data import parse_bool, read_peerread_rows
from peerread_review.model import load_model, predict
from peerread_review.xai import explain


def main() -> None:
    parser = argparse.ArgumentParser(description="Review one PeerRead paper by paper_id.")
    parser.add_argument("paper_id")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    rows = read_peerread_rows(DEFAULT_DATASET_PATH)
    row = next((item for item in rows if str(item.get("paper_id")) == str(args.paper_id)), None)
    if not row:
        raise SystemExit(f"Paper not found: {args.paper_id}")
    model = load_model(DEFAULT_MODEL_PATH)
    prediction = predict(model, row)
    xai = explain(model, prediction)
    result = {
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "actual_label": "Accept" if parse_bool(row.get("accepted")) else "Reject",
        "prediction": prediction,
        "xai": xai,
    }
    result["prediction"].pop("text", None)
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json_output:
        DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = DEFAULT_REPORT_DIR / args.json_output
        path.write_text(output, encoding="utf-8")
        print(f"Saved {path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
