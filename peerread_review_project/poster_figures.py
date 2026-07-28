from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

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
    out = DEFAULT_REPORT_DIR / "poster_figures"
    out.mkdir(parents=True, exist_ok=True)
    predicted = Counter(row["predicted_decision"] for row in data["papers"])
    actual = Counter(row["actual_label"] for row in data["papers"])
    (out / "01_peerread_predicted_decisions.svg").write_text(svg_bar("PeerRead Predicted Decisions", predicted, ["Accept", "Modify", "Reject"]), encoding="utf-8")
    (out / "02_peerread_actual_labels.svg").write_text(svg_bar("PeerRead Actual Labels", actual, ["Accept", "Reject"]), encoding="utf-8")
    (out / "SYSTEM_ARCHITECTURE.svg").write_text(svg_architecture(), encoding="utf-8")
    notes = "# PeerRead Poster Figures\n\n- Predicted decisions\n- Actual labels\n- System architecture\n"
    (out / "poster_figure_notes.md").write_text(notes, encoding="utf-8")
    print(f"Saved PeerRead poster figures to {out}")


if __name__ == "__main__":
    main()
