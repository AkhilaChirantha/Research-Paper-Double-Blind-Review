from __future__ import annotations

import json
from collections import Counter

from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, DEFAULT_REPORT_DIR
from peerread_review.data import parse_bool, read_peerread_rows
from peerread_review.model import evaluate, train_model


def main() -> None:
    rows = read_peerread_rows(DEFAULT_DATASET_PATH)
    train_rows = [row for row in rows if row.get("split") == "train"]
    dev_rows = [row for row in rows if row.get("split") == "dev"]
    test_rows = [row for row in rows if row.get("split") == "test"]
    model = train_model(train_rows, DEFAULT_MODEL_PATH)
    report = {
        "dataset": str(DEFAULT_DATASET_PATH),
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "test_rows": len(test_rows),
        "labels": dict(Counter("accept" if parse_bool(row.get("accepted")) else "reject" for row in rows)),
        "model": model,
        "dev_evaluation": evaluate(model, dev_rows),
        "test_evaluation": evaluate(model, test_rows),
    }
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_REPORT_DIR / "training_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# PeerRead Training Summary",
        "",
        f"- Total rows: {len(rows):,}",
        f"- Train rows: {len(train_rows):,}",
        f"- Dev rows: {len(dev_rows):,}",
        f"- Test rows: {len(test_rows):,}",
        f"- Labels: {report['labels']}",
        f"- Dev evaluation: {report['dev_evaluation']}",
        f"- Test evaluation: {report['test_evaluation']}",
    ]
    (DEFAULT_REPORT_DIR / "training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {DEFAULT_MODEL_PATH}")
    print(f"Saved {DEFAULT_REPORT_DIR / 'training_summary.md'}")


if __name__ == "__main__":
    main()
