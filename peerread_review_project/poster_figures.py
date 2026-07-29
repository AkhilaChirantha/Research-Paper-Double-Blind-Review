from __future__ import annotations

import json
from collections import Counter

from peerread_review.config import DEFAULT_REPORT_DIR


COLORS = {"Accept": "#157347", "Modify": "#c77700", "Reject": "#b42318"}


def svg_bar(title: str, counts: dict, labels: list[str], width: int = 1280, height: int = 720) -> str:
    max_value = max([counts.get(label, 0) for label in labels] + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="16" y="16" width="1248" height="688" rx="22" fill="#ffffff" stroke="#d0d5dd" stroke-width="2"/>',
        f'<text x="54" y="78" style="font:850 44px Arial;fill:#111827">{title}</text>',
    ]
    for i, label in enumerate(labels):
        value = counts.get(label, 0)
        x = 160 + i * 300
        h = 430 * value / max_value
        y = 590 - h
        color = COLORS.get(label, "#2477b3")
        parts.append(f'<rect x="{x}" y="{y}" width="150" height="{h}" rx="14" fill="{color}"/>')
        parts.append(f'<text x="{x + 75}" y="{y - 20}" text-anchor="middle" style="font:800 28px Arial;fill:#111827">{value:,}</text>')
        parts.append(f'<text x="{x + 75}" y="650" text-anchor="middle" style="font:700 26px Arial;fill:#344054">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_metrics(training: dict, width: int = 1280, height: int = 720) -> str:
    metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "accept_precision"),
        ("Recall", "accept_recall"),
        ("F1 Score", "accept_f1"),
    ]
    dev = training["dev_evaluation"]
    test = training["test_evaluation"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="16" y="16" width="1248" height="688" rx="22" fill="#ffffff" stroke="#d0d5dd" stroke-width="2"/>',
        '<text x="54" y="78" style="font:850 42px Arial;fill:#111827">Classification Metrics</text>',
        '<text x="54" y="116" style="font:22px Arial;fill:#667085">Dev vs test performance for PeerRead accept/reject screening</text>',
        '<line x1="100" y1="590" x2="1160" y2="590" stroke="#98a2b3" stroke-width="2"/>',
        '<line x1="100" y1="160" x2="100" y2="590" stroke="#98a2b3" stroke-width="2"/>',
    ]
    for tick in range(0, 6):
        value = tick / 5
        y = 590 - value * 410
        parts.append(f'<line x1="96" y1="{y}" x2="1160" y2="{y}" stroke="#eaecf0" stroke-width="1"/>')
        parts.append(f'<text x="82" y="{y + 7}" text-anchor="end" style="font:16px Arial;fill:#667085">{value:.1f}</text>')
    for index, (label, key) in enumerate(metrics):
        x = 180 + index * 240
        dev_h = 410 * float(dev.get(key, 0))
        test_h = 410 * float(test.get(key, 0))
        parts.append(f'<rect x="{x}" y="{590 - dev_h}" width="70" height="{dev_h}" rx="8" fill="#2563eb"/>')
        parts.append(f'<rect x="{x + 82}" y="{590 - test_h}" width="70" height="{test_h}" rx="8" fill="#f97316"/>')
        parts.append(f'<text x="{x + 35}" y="{575 - dev_h}" text-anchor="middle" style="font:700 17px Arial;fill:#111827">{float(dev.get(key, 0)):.2f}</text>')
        parts.append(f'<text x="{x + 117}" y="{575 - test_h}" text-anchor="middle" style="font:700 17px Arial;fill:#111827">{float(test.get(key, 0)):.2f}</text>')
        parts.append(f'<text x="{x + 76}" y="642" text-anchor="middle" style="font:700 20px Arial;fill:#344054">{label}</text>')
    parts.append('<rect x="940" y="70" width="22" height="22" fill="#2563eb"/><text x="972" y="88" style="font:18px Arial;fill:#344054">Dev</text>')
    parts.append('<rect x="1025" y="70" width="22" height="22" fill="#f97316"/><text x="1057" y="88" style="font:18px Arial;fill:#344054">Test</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_confusion(title: str, confusion: dict, width: int = 900, height: int = 720) -> str:
    cells = [
        ("Actual Reject", "Pred Reject", confusion.get("reject_as_reject", 0), 170, 210),
        ("Actual Reject", "Pred Accept", confusion.get("reject_as_accept", 0), 480, 210),
        ("Actual Accept", "Pred Reject", confusion.get("accept_as_reject", 0), 170, 430),
        ("Actual Accept", "Pred Accept", confusion.get("accept_as_accept", 0), 480, 430),
    ]
    max_value = max([cell[2] for cell in cells] + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="16" y="16" width="868" height="688" rx="22" fill="#ffffff" stroke="#d0d5dd" stroke-width="2"/>',
        f'<text x="52" y="78" style="font:850 40px Arial;fill:#111827">{title}</text>',
        '<text x="360" y="145" text-anchor="middle" style="font:700 22px Arial;fill:#344054">Predicted Label</text>',
        '<text x="48" y="365" transform="rotate(-90 48 365)" text-anchor="middle" style="font:700 22px Arial;fill:#344054">Actual Label</text>',
        '<text x="280" y="185" text-anchor="middle" style="font:700 20px Arial;fill:#667085">Reject</text>',
        '<text x="590" y="185" text-anchor="middle" style="font:700 20px Arial;fill:#667085">Accept</text>',
        '<text x="138" y="300" text-anchor="end" style="font:700 20px Arial;fill:#667085">Reject</text>',
        '<text x="138" y="520" text-anchor="end" style="font:700 20px Arial;fill:#667085">Accept</text>',
    ]
    for _, _, value, x, y in cells:
        intensity = value / max_value
        shade = int(245 - intensity * 135)
        color = f"rgb({shade},{min(shade + 25, 255)},255)"
        parts.append(f'<rect x="{x}" y="{y}" width="260" height="180" rx="16" fill="{color}" stroke="#bfdbfe" stroke-width="2"/>')
        parts.append(f'<text x="{x + 130}" y="{y + 92}" text-anchor="middle" style="font:850 44px Arial;fill:#111827">{value}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_probability_histogram(rows: list[dict], width: int = 1280, height: int = 720) -> str:
    bins = [i / 10 for i in range(11)]
    accept_bins = [0] * 10
    reject_bins = [0] * 10
    for row in rows:
        prob = float(row["accept_probability"])
        index = min(int(prob * 10), 9)
        if row["actual_label"] == "Accept":
            accept_bins[index] += 1
        else:
            reject_bins[index] += 1
    max_value = max(accept_bins + reject_bins + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="16" y="16" width="1248" height="688" rx="22" fill="#ffffff" stroke="#d0d5dd" stroke-width="2"/>',
        '<text x="54" y="78" style="font:850 42px Arial;fill:#111827">Accept Probability Distribution</text>',
        '<text x="54" y="116" style="font:22px Arial;fill:#667085">Predicted probability grouped by true PeerRead label</text>',
        '<line x1="90" y1="590" x2="1190" y2="590" stroke="#98a2b3" stroke-width="2"/>',
    ]
    for i in range(10):
        x = 115 + i * 105
        ah = 390 * accept_bins[i] / max_value
        rh = 390 * reject_bins[i] / max_value
        parts.append(f'<rect x="{x}" y="{590 - rh}" width="38" height="{rh}" fill="#b42318"/>')
        parts.append(f'<rect x="{x + 42}" y="{590 - ah}" width="38" height="{ah}" fill="#157347"/>')
        parts.append(f'<text x="{x + 40}" y="630" text-anchor="middle" style="font:15px Arial;fill:#667085">{bins[i]:.1f}-{bins[i+1]:.1f}</text>')
    parts.append('<rect x="910" y="70" width="22" height="22" fill="#b42318"/><text x="942" y="88" style="font:18px Arial;fill:#344054">Actual Reject</text>')
    parts.append('<rect x="1070" y="70" width="22" height="22" fill="#157347"/><text x="1102" y="88" style="font:18px Arial;fill:#344054">Actual Accept</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_feature_importance(training: dict, width: int = 1280, height: int = 720) -> str:
    model = training["model"]
    pairs = sorted(
        zip(model["feature_names"], model["weights"]),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:12]
    max_value = max([abs(float(weight)) for _, weight in pairs] + [1e-9])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="16" y="16" width="1248" height="688" rx="22" fill="#ffffff" stroke="#d0d5dd" stroke-width="2"/>',
        '<text x="54" y="78" style="font:850 42px Arial;fill:#111827">Top Logistic Regression Feature Weights</text>',
        '<text x="54" y="116" style="font:22px Arial;fill:#667085">Green increases accept probability; red increases reject probability</text>',
    ]
    for i, (name, weight) in enumerate(pairs):
        y = 155 + i * 42
        w = 620 * abs(float(weight)) / max_value
        color = "#157347" if float(weight) >= 0 else "#b42318"
        parts.append(f'<text x="70" y="{y + 24}" style="font:18px Arial;fill:#344054">{name}</text>')
        parts.append(f'<rect x="420" y="{y}" width="{w}" height="28" rx="7" fill="{color}"/>')
        parts.append(f'<text x="{430 + w}" y="{y + 22}" style="font:700 17px Arial;fill:#111827">{float(weight):.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_calibration(rows: list[dict], width: int = 1280, height: int = 720) -> str:
    points = []
    for i in range(10):
        low = i / 10
        high = (i + 1) / 10
        bucket = [row for row in rows if low <= float(row["accept_probability"]) < high or (i == 9 and float(row["accept_probability"]) == 1.0)]
        if bucket:
            avg_pred = sum(float(row["accept_probability"]) for row in bucket) / len(bucket)
            observed = sum(1 for row in bucket if row["actual_label"] == "Accept") / len(bucket)
            points.append((avg_pred, observed, len(bucket)))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="16" y="16" width="1248" height="688" rx="22" fill="#ffffff" stroke="#d0d5dd" stroke-width="2"/>',
        '<text x="54" y="78" style="font:850 42px Arial;fill:#111827">Probability Calibration</text>',
        '<text x="54" y="116" style="font:22px Arial;fill:#667085">Regression-style view: predicted probability vs observed accept rate</text>',
        '<line x1="130" y1="590" x2="1130" y2="590" stroke="#98a2b3" stroke-width="2"/>',
        '<line x1="130" y1="590" x2="130" y2="160" stroke="#98a2b3" stroke-width="2"/>',
        '<line x1="130" y1="590" x2="1130" y2="160" stroke="#94a3b8" stroke-width="2" stroke-dasharray="8 8"/>',
    ]
    for tick in range(0, 6):
        value = tick / 5
        x = 130 + value * 1000
        y = 590 - value * 430
        parts.append(f'<text x="{x}" y="630" text-anchor="middle" style="font:16px Arial;fill:#667085">{value:.1f}</text>')
        parts.append(f'<text x="112" y="{y + 6}" text-anchor="end" style="font:16px Arial;fill:#667085">{value:.1f}</text>')
    previous = None
    for pred, observed, count in points:
        x = 130 + pred * 1000
        y = 590 - observed * 430
        if previous:
            parts.append(f'<line x1="{previous[0]}" y1="{previous[1]}" x2="{x}" y2="{y}" stroke="#2563eb" stroke-width="4"/>')
        size = min(9 + count / 35, 26)
        parts.append(f'<circle cx="{x}" cy="{y}" r="{size}" fill="#2563eb" opacity="0.82"/>')
        previous = (x, y)
    parts.append('<text x="630" y="670" text-anchor="middle" style="font:20px Arial;fill:#344054">Predicted accept probability</text>')
    parts.append('<text x="52" y="375" transform="rotate(-90 52 375)" text-anchor="middle" style="font:20px Arial;fill:#344054">Observed accept rate</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_architecture(width: int = 1600, height: int = 900) -> str:
    def box(x, y, w, h, title, lines, fill):
        parts = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{fill}" stroke="#cbd5e1" stroke-width="3"/>']
        parts.append(f'<text x="{x + 24}" y="{y + 42}" style="font:800 25px Arial;fill:#111827">{title}</text>')
        for i, line in enumerate(lines):
            parts.append(f'<text x="{x + 24}" y="{y + 78 + i * 28}" style="font:18px Arial;fill:#344054">{line}</text>')
        return parts

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M2,2 L10,6 L2,10 Z" fill="#475467"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<rect x="22" y="22" width="1556" height="856" rx="26" fill="#ffffff" stroke="#d0d5dd" stroke-width="3"/>',
        '<text x="58" y="78" style="font:850 46px Arial;fill:#111827">PeerRead Supervised Review Framework</text>',
        '<text x="60" y="116" style="font:24px Arial;fill:#667085">True Accept/Reject labels + local supervised model + XAI suggestions</text>',
    ]
    parts.extend(box(70, 190, 250, 140, "PeerRead Data", ["Accepted labels", "Rejected labels", "Paper sections"], "#e8f3ff"))
    parts.append('<path d="M320 260 L390 260" stroke="#475467" stroke-width="4" marker-end="url(#arrow)"/>')
    parts.extend(box(390, 190, 250, 140, "Feature Extraction", ["Section counts", "Review scores", "Text evidence"], "#fff7e6"))
    parts.append('<path d="M640 260 L710 260" stroke="#475467" stroke-width="4" marker-end="url(#arrow)"/>')
    parts.extend(box(710, 190, 250, 140, "Supervised Model", ["Accept / Reject", "Modify borderline", "Probabilities"], "#eaf8ef"))
    parts.append('<path d="M960 260 L1030 260" stroke="#475467" stroke-width="4" marker-end="url(#arrow)"/>')
    parts.extend(box(1030, 190, 250, 140, "XAI Explanation", ["Risk factors", "Feature effects", "Suggestions"], "#eaf0ff"))
    parts.append('<path d="M1155 330 L1155 410" stroke="#475467" stroke-width="4" marker-end="url(#arrow)"/>')
    parts.extend(box(1030, 410, 250, 145, "Optional OpenAI", ["Extra feedback", "Only if enabled"], "#f2edff"))
    parts.append('<path d="M1030 482 L960 482" stroke="#475467" stroke-width="4" marker-end="url(#arrow)"/>')
    parts.extend(box(710, 410, 250, 145, "Reports", ["Decision table", "Metrics", "Charts"], "#fff0f0"))
    parts.append('<path d="M710 482 L640 482" stroke="#475467" stroke-width="4" marker-end="url(#arrow)"/>')
    parts.extend(box(390, 410, 250, 145, "SFT Dataset", ["Train JSONL", "Validation JSONL"], "#eaf8ef"))
    parts.append('<path d="M390 482 L320 482" stroke="#475467" stroke-width="4" marker-end="url(#arrow)"/>')
    parts.extend(box(70, 410, 250, 145, "Dashboard", ["Filters", "XAI table", "Downloads"], "#fff7e6"))
    parts.append('<rect x="70" y="650" width="570" height="120" rx="18" fill="#f8fafc" stroke="#cbd5e1" stroke-width="3"/>')
    parts.append('<text x="100" y="695" style="font:800 25px Arial;fill:#111827">Dataset Advantage</text>')
    parts.append('<text x="100" y="728" style="font:18px Arial;fill:#344054">PeerRead contains true accepted and rejected papers.</text>')
    parts.append('<text x="100" y="756" style="font:18px Arial;fill:#344054">This supports supervised accept/reject training.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    data = json.loads((DEFAULT_REPORT_DIR / "peerread_decisions.json").read_text(encoding="utf-8"))
    training = json.loads((DEFAULT_REPORT_DIR / "training_summary.json").read_text(encoding="utf-8"))
    out = DEFAULT_REPORT_DIR / "poster_figures"
    out.mkdir(parents=True, exist_ok=True)
    predicted = Counter(row["predicted_decision"] for row in data["papers"])
    actual = Counter(row["actual_label"] for row in data["papers"])
    (out / "01_peerread_predicted_decisions.svg").write_text(svg_bar("PeerRead Predicted Decisions", predicted, ["Accept", "Modify", "Reject"]), encoding="utf-8")
    (out / "02_peerread_actual_labels.svg").write_text(svg_bar("PeerRead Actual Labels", actual, ["Accept", "Reject"]), encoding="utf-8")
    (out / "03_classification_metrics.svg").write_text(svg_metrics(training), encoding="utf-8")
    (out / "04_test_confusion_heatmap.svg").write_text(svg_confusion("Test Confusion Matrix Heatmap", training["test_evaluation"]["confusion"]), encoding="utf-8")
    (out / "05_probability_distribution.svg").write_text(svg_probability_histogram(data["papers"]), encoding="utf-8")
    (out / "06_feature_importance.svg").write_text(svg_feature_importance(training), encoding="utf-8")
    (out / "07_probability_calibration.svg").write_text(svg_calibration(data["papers"]), encoding="utf-8")
    (out / "SYSTEM_ARCHITECTURE.svg").write_text(svg_architecture(), encoding="utf-8")
    notes = "\n".join(
        [
            "# PeerRead Poster Figures",
            "",
            "- Predicted decisions",
            "- Actual labels",
            "- Classification metrics",
            "- Test confusion matrix heatmap",
            "- Accept probability distribution",
            "- Feature importance",
            "- Probability calibration",
            "- System architecture",
        ]
    )
    (out / "poster_figure_notes.md").write_text(notes, encoding="utf-8")
    print(f"Saved PeerRead poster figures to {out}")


if __name__ == "__main__":
    main()
