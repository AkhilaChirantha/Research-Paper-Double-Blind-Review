from __future__ import annotations

import argparse
from pathlib import Path

from research_review.config import DEFAULT_CRITERIA_PATH, DEFAULT_MODEL_PATH, DEFAULT_PAPERS_DIR
from research_review.model import train, training_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local research paper review model.")
    parser.add_argument("--criteria", default=str(DEFAULT_CRITERIA_PATH))
    parser.add_argument("--papers-dir", default=str(DEFAULT_PAPERS_DIR))
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()

    model = train(Path(args.criteria), Path(args.papers_dir), Path(args.output))
    print(f"Saved {args.output}; {training_summary(model)}")


if __name__ == "__main__":
    main()
