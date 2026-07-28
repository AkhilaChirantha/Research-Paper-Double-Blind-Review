from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from peerread_review.data import paper_text, parse_bool
from peerread_review.features import FEATURE_NAMES, extract_features, feature_vector


LABELS = ["reject", "accept"]


def train_model(rows: list[dict], output_path: Path) -> dict:
    split_counts = Counter()
    label_counts = Counter()
    vectors = []
    labels = []
    for row in rows:
        label = "accept" if parse_bool(row.get("accepted")) else "reject"
        split_counts[row.get("split") or "unknown"] += 1
        label_counts[label] += 1
        text = paper_text(row)
        vectors.append(feature_vector(extract_features(row, text)))
        labels.append(1.0 if label == "accept" else 0.0)

    means = column_mean(vectors)
    stds = column_std(vectors, means)
    scaled = [scale_vector(vector, means, stds) for vector in vectors]
    weights, bias, loss = fit_logistic_regression(scaled, labels)

    model = {
        "version": 1,
        "model_type": "logistic_regression",
        "labels": LABELS,
        "feature_names": FEATURE_NAMES,
        "training_count": sum(label_counts.values()),
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "scaler": {"mean": means, "std": stds},
        "weights": weights,
        "bias": bias,
        "training_loss": loss,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predict(model: dict, row: dict) -> dict:
    text = paper_text(row)
    features = extract_features(row, text)
    vector = feature_vector(features)
    scaled = scale_vector(vector, model["scaler"]["mean"], model["scaler"]["std"])
    accept_probability = sigmoid(dot(model["weights"], scaled) + float(model["bias"]))
    if accept_probability >= 0.65:
        decision = "Accept"
    elif accept_probability <= 0.35:
        decision = "Reject"
    else:
        decision = "Modify"
    return {
        "decision": decision,
        "accept_probability": round(accept_probability, 4),
        "reject_probability": round(1.0 - accept_probability, 4),
        "features": features,
        "scaled_features": scaled,
        "text": text,
    }


def evaluate(model: dict, rows: list[dict]) -> dict:
    totals = Counter()
    correct = 0
    for row in rows:
        actual = "accept" if parse_bool(row.get("accepted")) else "reject"
        prediction = predict(model, row)
        predicted = "accept" if prediction["accept_probability"] >= 0.5 else "reject"
        totals[(actual, predicted)] += 1
        if actual == predicted:
            correct += 1
    total = max(len(rows), 1)
    tp = totals[("accept", "accept")]
    fp = totals[("reject", "accept")]
    fn = totals[("accept", "reject")]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "rows": len(rows),
        "accuracy": round(correct / total, 4),
        "accept_precision": round(precision, 4),
        "accept_recall": round(recall, 4),
        "accept_f1": round(f1, 4),
        "confusion": {f"{actual}_as_{predicted}": count for (actual, predicted), count in totals.items()},
    }


def column_mean(matrix: list[list[float]]) -> list[float]:
    if not matrix:
        return [0.0 for _ in FEATURE_NAMES]
    return [sum(row[i] for row in matrix) / len(matrix) for i in range(len(FEATURE_NAMES))]


def column_std(matrix: list[list[float]], means: list[float]) -> list[float]:
    if not matrix:
        return [1.0 for _ in FEATURE_NAMES]
    stds = []
    for i in range(len(FEATURE_NAMES)):
        variance = sum((row[i] - means[i]) ** 2 for row in matrix) / len(matrix)
        stds.append(max(math.sqrt(variance), 1e-6))
    return stds


def scale_vector(vector: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(vector[i] - float(means[i])) / max(float(stds[i]), 1e-6) for i in range(len(FEATURE_NAMES))]


def fit_logistic_regression(
    matrix: list[list[float]],
    labels: list[float],
    epochs: int = 900,
    learning_rate: float = 0.08,
    l2: float = 0.002,
) -> tuple[list[float], float, float]:
    width = len(FEATURE_NAMES)
    weights = [0.0 for _ in range(width)]
    positive_rate = sum(labels) / max(len(labels), 1)
    bias = math.log(max(positive_rate, 1e-6) / max(1.0 - positive_rate, 1e-6))
    loss = 0.0
    for _ in range(epochs):
        grad_w = [0.0 for _ in range(width)]
        grad_b = 0.0
        loss = 0.0
        for vector, y in zip(matrix, labels):
            pred = sigmoid(dot(weights, vector) + bias)
            error = pred - y
            grad_b += error
            for i, value in enumerate(vector):
                grad_w[i] += error * value
            loss += -(y * math.log(max(pred, 1e-9)) + (1 - y) * math.log(max(1 - pred, 1e-9)))
        n = max(len(labels), 1)
        for i in range(width):
            grad = grad_w[i] / n + l2 * weights[i]
            weights[i] -= learning_rate * grad
        bias -= learning_rate * grad_b / n
        loss = loss / n + 0.5 * l2 * sum(weight * weight for weight in weights)
    return [round(weight, 8) for weight in weights], round(bias, 8), round(loss, 6)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)
