from __future__ import annotations

import html
import json
from collections import Counter, defaultdict

from peerread_review.config import DEFAULT_DATASET_PATH, DEFAULT_REPORT_DIR
from peerread_review.data import parse_bool, read_peerread_rows


def main() -> None:
    rows = read_peerread_rows(DEFAULT_DATASET_PATH)
    labels = Counter("Accept" if parse_bool(row.get("accepted")) else "Reject" for row in rows)
    splits = Counter(row.get("split") or "unknown" for row in rows)
    conferences = Counter(row.get("conference") or "unknown" for row in rows)
    by_conference = defaultdict(Counter)
    for row in rows:
        by_conference[row.get("conference") or "unknown"]["Accept" if parse_bool(row.get("accepted")) else "Reject"] += 1

    payload = {
        "dataset": str(DEFAULT_DATASET_PATH),
        "total_rows": len(rows),
        "labels": dict(labels),
        "splits": dict(splits),
        "conferences": dict(conferences),
        "by_conference": {key: dict(value) for key, value in by_conference.items()},
    }
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (DEFAULT_REPORT_DIR / "dataset_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = [
        "# PeerRead Dataset Summary",
        "",
        f"- Total papers: {len(rows):,}",
        f"- Accept labels: {labels.get('Accept', 0):,}",
        f"- Reject labels: {labels.get('Reject', 0):,}",
        f"- Splits: {dict(splits)}",
        "",
        "## Conferences",
        "",
    ]
    for conference, count in conferences.most_common():
        md.append(f"- {conference}: {count:,} ({dict(by_conference[conference])})")
    (DEFAULT_REPORT_DIR / "dataset_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    rows_html = "".join(
        "<tr>"
        f"<td>{html.escape(conf)}</td>"
        f"<td>{count:,}</td>"
        f"<td>{by_conference[conf].get('Accept', 0):,}</td>"
        f"<td>{by_conference[conf].get('Reject', 0):,}</td>"
        "</tr>"
        for conf, count in conferences.most_common()
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>PeerRead Dataset Summary</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:28px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}</style>
</head><body>
<h1>PeerRead Dataset Summary</h1>
<p>Total papers: <strong>{len(rows):,}</strong></p>
<p>Accept: <strong>{labels.get('Accept', 0):,}</strong> | Reject: <strong>{labels.get('Reject', 0):,}</strong></p>
<h2>Conferences</h2>
<table><thead><tr><th>Conference</th><th>Total</th><th>Accept</th><th>Reject</th></tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""
    (DEFAULT_REPORT_DIR / "dataset_summary.html").write_text(doc, encoding="utf-8")
    print(f"Saved dataset summary to {DEFAULT_REPORT_DIR}")


if __name__ == "__main__":
    main()
