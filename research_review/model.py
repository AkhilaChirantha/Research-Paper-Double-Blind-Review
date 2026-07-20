from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from research_review.features import FEATURE_NAMES, extract_features, feature_vector, summarize_feature_gaps
from research_review.io import read_document, read_json, write_json


LABELS = ["good_paper", "needs_modification", "reject_risk"]


def parse_score(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.split(":", 1)[0].split()[0])
    except (ValueError, IndexError):
        return None


def paper_training_label(item: dict) -> str:
    decision = str(item.get("decision") or "").lower()
    ratings = []
    sub_scores = []
    for review in item.get("reviews", []):
        rating = parse_score(review.get("rating"))
        if rating is not None:
            ratings.append(rating)
        for key in ("soundness", "presentation", "contribution"):
            score = parse_score(review.get(key))
            if score is not None:
                sub_scores.append(score)

    avg_rating = sum(ratings) / len(ratings) if ratings else 5.0
    min_rating = min(ratings) if ratings else 5.0
    avg_sub = sum(sub_scores) / len(sub_scores) if sub_scores else 2.5

    if "oral" in decision or "spotlight" in decision or avg_rating >= 6.4:
        return "good_paper"
    if min_rating <= 3.0 or avg_rating < 4.6 or avg_sub < 2.2:
        return "reject_risk"
    return "needs_modification"


def build_training_rows(criteria_path: Path, papers_dir: Path) -> list[dict]:
    criteria = read_json(criteria_path)
    rows = []
    for item in criteria:
        paper_id = item.get("paper_id")
        if not paper_id:
            continue
        paper_path = papers_dir / f"{paper_id}.md"
        if not paper_path.exists():
            continue
        text = read_document(paper_path)
        rows.append(
            {
                "paper_id": paper_id,
                "title": item.get("title", ""),
                "decision": item.get("decision", ""),
                "label": paper_training_label(item),
                "features": extract_features(text),
            }
        )
    return rows


def train(criteria_path: Path, papers_dir: Path, output_path: Path) -> dict:
    rows = build_training_rows(criteria_path, papers_dir)
    if not rows:
        raise RuntimeError("No training papers found. Check data paths.")

    by_label: dict[str, list[list[float]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(feature_vector(row["features"]))

    total = sum(len(values) for values in by_label.values())
    model = {
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "labels": LABELS,
        "training_count": total,
        "label_counts": {label: len(by_label.get(label, [])) for label in LABELS},
        "classes": {},
    }

    global_matrix = [[math.log1p(value) for value in feature_vector(row["features"])] for row in rows]
    global_var = column_variance(global_matrix, floor=1e-4)

    for label in LABELS:
        values = by_label.get(label) or by_label.get("needs_modification") or []
        transformed = [[math.log1p(value) for value in vector] for vector in values]
        means = column_mean(transformed)
        variances = column_variance(transformed, floor=1e-4)
        model["classes"][label] = {
            "prior": max(len(values), 1) / total,
            "mean": means,
            "var": [variances[i] + global_var[i] * 0.15 + 1e-4 for i in range(len(FEATURE_NAMES))],
        }

    write_json(output_path, model)
    return model


def load_model(path: Path) -> dict:
    return read_json(path)


def predict_with_model(model: dict, text: str) -> dict:
    features = extract_features(text)
    x = [math.log1p(value) for value in feature_vector(features)]
    log_scores = {}
    for label in model["labels"]:
        cls = model["classes"][label]
        mean = [float(value) for value in cls["mean"]]
        var = [float(value) for value in cls["var"]]
        log_likelihood = 0.0
        for index, value in enumerate(x):
            variance = max(var[index], 1e-6)
            log_likelihood += -0.5 * (
                math.log(2 * math.pi * variance) + ((value - mean[index]) ** 2 / variance)
            )
        log_scores[label] = float(math.log(cls["prior"]) + log_likelihood)

    max_log = max(log_scores.values())
    exp_scores = {label: math.exp(value - max_log) for label, value in log_scores.items()}
    total = sum(exp_scores.values())
    probabilities = {label: value / total for label, value in exp_scores.items()}
    label = max(probabilities, key=probabilities.get)

    quality_score = round(
        100
        * (
            probabilities.get("good_paper", 0.0)
            + 0.55 * probabilities.get("needs_modification", 0.0)
        ),
        1,
    )
    verdict = {
        "good_paper": "GOOD_PAPER",
        "needs_modification": "NEEDS_MODIFICATION",
        "reject_risk": "REJECT_RISK",
    }[label]

    return {
        "verdict": verdict,
        "quality_score": quality_score,
        "probabilities": probabilities,
        "features": features,
        "feature_gaps": summarize_feature_gaps(features),
    }


def training_summary(model: dict) -> str:
    counts = Counter(model.get("label_counts", {}))
    return (
        f"trained on {model.get('training_count', 0)} papers "
        f"(good={counts.get('good_paper', 0)}, modify={counts.get('needs_modification', 0)}, "
        f"reject-risk={counts.get('reject_risk', 0)})"
    )


def column_mean(matrix: list[list[float]]) -> list[float]:
    if not matrix:
        return [0.0 for _ in FEATURE_NAMES]
    width = len(matrix[0])
    return [sum(row[i] for row in matrix) / len(matrix) for i in range(width)]


def column_variance(matrix: list[list[float]], floor: float = 0.0) -> list[float]:
    if not matrix:
        return [floor for _ in FEATURE_NAMES]
    means = column_mean(matrix)
    width = len(matrix[0])
    variances = []
    for i in range(width):
        variances.append(sum((row[i] - means[i]) ** 2 for row in matrix) / len(matrix) + floor)
    return variances
