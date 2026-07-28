from __future__ import annotations

import json

from peerread_review.config import DEFAULT_REPORT_DIR


def main() -> None:
    training = json.loads((DEFAULT_REPORT_DIR / "training_summary.json").read_text(encoding="utf-8"))
    decisions = json.loads((DEFAULT_REPORT_DIR / "peerread_decisions.json").read_text(encoding="utf-8"))
    lines = [
        "# PeerRead Final Evaluation",
        "",
        "## Dataset",
        "",
        f"- Total rows: {training['total_rows']:,}",
        f"- Labels: {training['labels']}",
        "",
        "## Model Evaluation",
        "",
        f"- Dev: {training['dev_evaluation']}",
        f"- Test: {training['test_evaluation']}",
        "",
        "## Decision Counts",
        "",
        f"- Predicted: {decisions['counts']}",
        f"- Actual: {decisions['actual_counts']}",
        "",
        "## Completed Parts",
        "",
        "- supervised PeerRead accept/reject model",
        "- XAI explanations",
        "- paper-by-paper table",
        "- dataset summary",
        "- section summaries",
        "- SFT dataset",
        "- poster figures",
        "- dashboard",
    ]
    (DEFAULT_REPORT_DIR / "final_evaluation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {DEFAULT_REPORT_DIR / 'final_evaluation_summary.md'}")


if __name__ == "__main__":
    main()
