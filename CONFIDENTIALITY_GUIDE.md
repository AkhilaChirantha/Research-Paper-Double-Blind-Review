# Confidentiality Modes

The review system now supports four confidentiality modes.

## Modes

```text
local_only
abstract_only
section_summary_only
full_paper_with_consent
```

## 1. local_only

No external API call is allowed.

Use this for private or unpublished papers when you only want the local model output.

```bash
.venv312/bin/python review_paper.py "paper.md" --confidentiality-mode local_only
```

## 2. abstract_only

Only the masked abstract/opening text is sent to the AI API.

```bash
.venv312/bin/python review_paper.py "paper.md" --use-openai --confidentiality-mode abstract_only
```

## 3. section_summary_only

The system extracts/summarizes available sections and sends only the summary text to the AI API.

```bash
.venv312/bin/python review_paper.py "paper.md" --use-openai --confidentiality-mode section_summary_only
```

Shortcut:

```bash
.venv312/bin/python review_paper.py "paper.md" --section-summary
```

## 4. full_paper_with_consent

The masked full paper text is used. Use this only when the paper owner explicitly agrees.

```bash
.venv312/bin/python review_paper.py "paper.md" --use-openai --confidentiality-mode full_paper_with_consent
```

## What Gets Masked

- emails
- URLs
- OpenReview links
- ORCID-like identifiers
- obvious affiliation lines
- acknowledgement sections
- PDF/forum links

## Advanced AI Review Commands

Default safer mode:

```bash
.venv312/bin/python top_papers_openai.py --per-group 5 --confidentiality-mode section_summary_only
```

Abstract-only:

```bash
.venv312/bin/python top_papers_openai.py --per-group 5 --confidentiality-mode abstract_only
```

Full paper with consent:

```bash
.venv312/bin/python top_papers_openai.py --per-group 5 --confidentiality-mode full_paper_with_consent
```

