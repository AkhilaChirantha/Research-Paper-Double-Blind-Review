# Research Review

Blind-review assistant for research papers before submission.

It has three layers:

1. A local Python ML model trained from `data/Analyse Critaria/metadata.json` and `data/Research Papers/*.md`.
2. A default local XAI layer that explains the model decision and generates XAI-based recommendations.
3. An optional OpenAI API reviewer that gives extra structured section-level recommendations.

The local model is intentionally conservative. Your current dataset contains accepted NeurIPS papers only, so the training script derives three practical classes from reviewer ratings and sub-scores:

- `GOOD_PAPER`
- `NEEDS_MODIFICATION`
- `REJECT_RISK`

This is decision support, not a real conference acceptance guarantee.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Your `.env` already has:

```bash
OPENAI_API_KEY=...
```

Optional:

```bash
OPENAI_MODEL=gpt-4.1-mini
```

This project is configured to use `gpt-4.1-mini` for OpenAI-assisted reviews.

## Train

```bash
python -m research_review.train
```

This creates:

```text
models/research_review_model.json
```

## Review A Paper Locally

```bash
python review_paper.py "data/Research Papers/7WTA298wts.md"
```

This uses the proposal-aligned default mode:

```text
XAI Local Review
```

## Review With OpenAI Recommendations

```bash
python review_paper.py "data/Research Papers/7WTA298wts.md" --review-mode xai-openai --confidentiality-mode section_summary_only --md-output reports/7WTA298wts.md --json-output reports/7WTA298wts.json
```

## Run the Dashboard

Use this for supervisor demos and paper-by-paper inspection:

```bash
.venv312/bin/streamlit run dashboard_app.py
```

The dashboard includes:

- dataset-level decision charts
- paper-by-paper table with suggestions
- OpenAI comparison report viewer
- new paper upload and review flow
- confidentiality modes before any OpenAI API call

Supported input types:

- `.md`
- `.txt`
- `.tex`
- `.pdf` after installing `pypdf`

## Output

The report includes:

- local verdict and quality score
- probabilities for good / needs modification / reject risk
- structural gaps detected by the local model
- OpenAI blind-review recommendation when `--use-openai` is enabled
- section-level changes to improve the paper
- rejection-risk reasons and an acceptance plan

## Paper-By-Paper Table

Generate one row per available paper with a local `Accept`, `Modify`, or `Reject` prediction:

```bash
python3 paper_decisions.py
```

Outputs:

```text
reports/paper_decisions.html
reports/paper_decisions.csv
reports/paper_decisions.json
```

## OpenAI Comparison Sample

This sends selected paper content or summaries to the OpenAI API, so run it only when you are comfortable with that. The default sample is balanced: 10 local Accept, 10 local Modify, and 10 local Reject papers.

```bash
.venv312/bin/python openai_compare.py --per-class 10 --confidentiality-mode section_summary_only
```

Safer abstract-only comparison:

```bash
.venv312/bin/python openai_compare.py --per-class 10 --abstract-only
```

See `CONFIDENTIALITY_GUIDE.md` for all modes.

Outputs:

```text
reports/openai_comparison.html
reports/openai_comparison.csv
reports/openai_comparison.json
```

## Poster Figures

Generate poster-ready SVG graphs:

```bash
python3 poster_figures.py
```

After `reports/openai_comparison.json` exists, this also generates local-vs-OpenAI comparison figures.

Outputs:

```text
reports/poster_figures/
```
