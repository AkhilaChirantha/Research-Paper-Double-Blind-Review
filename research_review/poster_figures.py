from __future__ import annotations

import argparse
import html
from collections import Counter, defaultdict
from pathlib import Path

from research_review.io import read_json


COLORS = {"Accept": "#157347", "Modify": "#c77700", "Reject": "#b42318"}
LABELS = ["Accept", "Modify", "Reject"]


def nice_max(value: int) -> int:
    if value <= 10:
        return 10
    magnitude = 10 ** (len(str(value)) - 1)
    return int(((value + magnitude - 1) // magnitude) * magnitude)


def style() -> str:
    return (
        "text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#111827}"
        ".title{font-size:44px;font-weight:850}"
        ".subtitle{font-size:22px;fill:#667085}"
        ".axis{stroke:#344054;stroke-width:3}"
        ".grid{stroke:#d0d5dd;stroke-width:1.5}"
        ".label{font-size:28px;font-weight:650}"
        ".tick{font-size:22px;fill:#475467}"
        ".value{font-size:27px;font-weight:850;text-anchor:middle}"
        ".legend{font-size:25px;font-weight:650}"
    )


def svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        "<style>",
        style(),
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<rect x="16" y="16" width="{width - 32}" height="{height - 32}" rx="22" fill="#ffffff" stroke="#d0d5dd" stroke-width="2"/>',
        f'<text x="46" y="68" class="title">{html.escape(title)}</text>',
    ]


def svg_bar_chart(
    title: str,
    items: list[tuple[str, int]],
    colors: dict[str, str],
    width: int = 1280,
    height: int = 720,
    subtitle: str | None = None,
) -> str:
    margin_left = 132
    margin_right = 70
    margin_bottom = 112
    margin_top = 128 if subtitle else 112
    usable_w = width - margin_left - margin_right
    usable_h = height - margin_top - margin_bottom
    max_value = nice_max(max([value for _, value in items] + [1]))
    col_w = usable_w / max(len(items), 1)
    parts = svg_start(width, height, title)
    if subtitle:
        parts.append(f'<text x="48" y="104" class="subtitle">{html.escape(subtitle)}</text>')

    for step in range(5):
        value = max_value * step / 4
        y = height - margin_bottom - usable_h * step / 4
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 18}" y="{y + 8:.1f}" class="tick" text-anchor="end">{value:,.0f}</text>')
    parts.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="axis"/>')

    for i, (label, value) in enumerate(items):
        bar_h = usable_h * value / max_value
        x = margin_left + i * col_w + col_w * 0.22
        y = height - margin_bottom - bar_h
        w = col_w * 0.56
        color = colors.get(label, "#2477b3")
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{bar_h:.1f}" rx="12" fill="{color}"/>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{max(y - 18, margin_top - 14):.1f}" class="value">{value:,}</text>')
        parts.append(f'<text x="{x + w / 2:.1f}" y="{height - 48}" class="label" text-anchor="middle">{html.escape(label)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_grouped_bar(title: str, local_counts: Counter, openai_counts: Counter, width: int = 1280, height: int = 720) -> str:
    margin_left = 132
    margin_right = 70
    margin_bottom = 112
    margin_top = 128
    usable_w = width - margin_left - margin_right
    usable_h = height - margin_top - margin_bottom
    max_value = nice_max(max([local_counts.get(label, 0) for label in LABELS] + [openai_counts.get(label, 0) for label in LABELS] + [1]))
    group_w = usable_w / len(LABELS)
    bar_w = group_w * 0.25
    parts = svg_start(width, height, title)
    parts.extend(
        [
            '<rect x="878" y="42" width="30" height="30" rx="7" fill="#2477b3"/><text x="922" y="67" class="legend">Local</text>',
            '<rect x="1038" y="42" width="30" height="30" rx="7" fill="#7d5fb2"/><text x="1082" y="67" class="legend">OpenAI</text>',
        ]
    )
    for step in range(5):
        value = max_value * step / 4
        y = height - margin_bottom - usable_h * step / 4
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{margin_left - 18}" y="{y + 8:.1f}" class="tick" text-anchor="end">{value:,.0f}</text>')
    parts.append(f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" class="axis"/>')
    parts.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" class="axis"/>')

    for i, label in enumerate(LABELS):
        center = margin_left + i * group_w + group_w / 2
        for offset, value, color in [
            (-bar_w * 0.58, local_counts.get(label, 0), "#2477b3"),
            (bar_w * 0.58, openai_counts.get(label, 0), "#7d5fb2"),
        ]:
            bar_h = usable_h * value / max_value
            x = center + offset - bar_w / 2
            y = height - margin_bottom - bar_h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="10" fill="{color}"/>')
            parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{max(y - 14, margin_top - 10):.1f}" class="value">{value:,}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height - 48}" class="label" text-anchor="middle">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_donut(title: str, counts: Counter, width: int = 1280, height: int = 720) -> str:
    total = sum(counts.get(label, 0) for label in LABELS) or 1
    cx, cy, radius = 405, 380, 170
    circumference = 2 * 3.141592653589793 * radius
    offset = 0.0
    parts = svg_start(width, height, title)
    parts.append(
        '<style>.center{font-size:30px;text-anchor:middle;font-weight:700;fill:#475467}'
        '.big{font-size:48px;text-anchor:middle;font-weight:900}'
        f'.donut{{fill:none;stroke-width:58;transform:rotate(-90deg);transform-origin:{cx}px {cy}px}}</style>'
    )
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="none" stroke="#e8edf1" stroke-width="58"/>')
    for label in LABELS:
        value = counts.get(label, 0)
        dash = circumference * value / total
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" class="donut" stroke="{COLORS[label]}" '
            f'stroke-dasharray="{dash:.2f} {circumference - dash:.2f}" stroke-dashoffset="{-offset:.2f}"/>'
        )
        offset += dash
    parts.append(f'<text x="{cx}" y="{cy - 14}" class="center">Papers</text>')
    parts.append(f'<text x="{cx}" y="{cy + 44}" class="big">{total:,}</text>')
    for i, label in enumerate(LABELS):
        value = counts.get(label, 0)
        pct = 100 * value / total
        y = 284 + i * 78
        parts.append(f'<rect x="760" y="{y - 28}" width="34" height="34" rx="8" fill="{COLORS[label]}"/>')
        parts.append(f'<text x="816" y="{y}" class="label">{label}: {value:,} ({pct:.1f}%)</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def svg_quality_histogram(rows: list[dict], width: int = 1280, height: int = 720) -> str:
    scores = [float(row["quality_score"]) for row in rows if row.get("quality_score") != ""]
    bins = [0 for _ in range(10)]
    for score in scores:
        index = min(9, max(0, int(score // 10)))
        bins[index] += 1
    items = [(f"{i * 10}-{i * 10 + 9}" if i < 9 else "90-100", value) for i, value in enumerate(bins)]
    return svg_bar_chart(
        "Local Quality Score Distribution",
        items,
        defaultdict(lambda: "#2477b3"),
        width,
        height,
        subtitle="Higher scores indicate stronger local accept readiness.",
    )


def svg_matrix(comparison: dict, width: int = 1120, height: int = 820) -> str:
    rows = comparison.get("papers", [])
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        matrix[row["local_decision"]][row["openai_decision"]] += 1
    max_value = max([matrix[local][openai] for local in LABELS for openai in LABELS] + [1])
    cell = 160
    start_x = 350
    start_y = 200
    parts = svg_start(width, height, "Local vs OpenAI Agreement Matrix")
    parts.append('<text x="590" y="138" class="label" text-anchor="middle">OpenAI decision</text>')
    parts.append('<text x="70" y="470" class="label" transform="rotate(-90 70 470)" text-anchor="middle">Local decision</text>')
    for j, label in enumerate(LABELS):
        parts.append(f'<text x="{start_x + j * cell + cell / 2}" y="176" class="label" text-anchor="middle">{label}</text>')
    for i, local in enumerate(LABELS):
        parts.append(f'<text x="{start_x - 24}" y="{start_y + i * cell + cell / 2 + 10}" class="label" text-anchor="end">{local}</text>')
        for j, openai in enumerate(LABELS):
            value = matrix[local][openai]
            intensity = 0.18 + 0.72 * value / max_value
            color = f"rgba(36,119,179,{intensity:.2f})"
            x = start_x + j * cell
            y = start_y + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 12}" height="{cell - 12}" rx="16" fill="{color}" stroke="#ffffff" stroke-width="4"/>')
            parts.append(f'<text x="{x + cell / 2 - 6}" y="{y + cell / 2 + 12}" class="value">{value}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def write_poster_notes(local_rows: list[dict], output_dir: Path, comparison: dict | None) -> None:
    counts = Counter(row["predicted_decision"] for row in local_rows)
    lines = [
        "# Poster Figure Notes",
        "",
        "Generated SVG figures have white backgrounds and high-contrast labels:",
        "",
        "- `01_local_decision_distribution.svg`: local model decision counts.",
        "- `02_quality_score_distribution.svg`: quality score spread across papers.",
        "- `03_decision_share.svg`: percentage split of Accept / Modify / Reject.",
    ]
    if comparison:
        total = comparison["counts"]["total"]
        agreement = comparison["counts"]["agreement"]
        lines.extend(
            [
                "- `04_local_vs_openai_decisions.svg`: local vs OpenAI decision distribution.",
                "- `05_agreement_matrix.svg`: where local and OpenAI decisions agree/disagree.",
                "",
                f"OpenAI comparison sample: {total} papers.",
                f"Agreement rate: {100 * agreement / total if total else 0:.1f}%.",
            ]
        )
    else:
        lines.append("- OpenAI comparison figures will be generated after `reports/openai_comparison.json` exists.")
    lines.extend(
        [
            "",
            "Local model counts:",
            f"- Accept: {counts.get('Accept', 0):,}",
            f"- Modify: {counts.get('Modify', 0):,}",
            f"- Reject: {counts.get('Reject', 0):,}",
        ]
    )
    write(output_dir / "poster_figure_notes.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate poster-ready SVG result figures.")
    parser.add_argument("--local-json", default="reports/paper_decisions.json")
    parser.add_argument("--openai-json", default="reports/openai_comparison.json")
    parser.add_argument("--output-dir", default="reports/poster_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local = read_json(Path(args.local_json))
    local_rows = local["papers"]
    local_counts = Counter(row["predicted_decision"] for row in local_rows)

    write(
        output_dir / "01_local_decision_distribution.svg",
        svg_bar_chart(
            "Local Model Paper Decisions",
            [(label, local_counts.get(label, 0)) for label in LABELS],
            COLORS,
            subtitle="Predicted decisions for all available papers.",
        ),
    )
    write(output_dir / "02_quality_score_distribution.svg", svg_quality_histogram(local_rows))
    write(output_dir / "03_decision_share.svg", svg_donut("Share of Predicted Decisions", local_counts))

    comparison = read_json(Path(args.openai_json)) if Path(args.openai_json).exists() else None
    if comparison:
        local_compare = Counter(row["local_decision"] for row in comparison["papers"])
        openai_compare = Counter(row["openai_decision"] for row in comparison["papers"])
        write(output_dir / "04_local_vs_openai_decisions.svg", svg_grouped_bar("Local vs OpenAI Decisions", local_compare, openai_compare))
        write(output_dir / "05_agreement_matrix.svg", svg_matrix(comparison))

    write_poster_notes(local_rows, output_dir, comparison)
    print(f"Saved poster figures to {output_dir}")


if __name__ == "__main__":
    main()
