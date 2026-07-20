# XAI Guide

## What XAI Means Here

XAI means explainable artificial intelligence. In this project, XAI is not a separate paid API. It is a local explanation layer built on top of the trained local screening model.

The local model predicts:

```text
Accept
Modify
Reject-risk
```

The XAI layer explains why the local model produced that result by showing which paper features supported or weakened the prediction.

## Why XAI Is the Default

The proposal says suggestions/explanations should come from XAI. Therefore the system now uses this default flow:

```text
Paper
-> Local ML screening model
-> XAI feature explanation
-> XAI-based recommendations
```

OpenAI is kept only as an optional extra layer:

```text
Paper
-> XAI local review
-> optional OpenAI detailed natural-language review
```

This keeps the proposal-aligned method as the main/default method while still allowing richer OpenAI suggestions when needed.

## How the Current XAI Layer Works

The project uses a lightweight local XAI method called:

```text
Local XAI feature-distance explanation
```

High-level steps:

1. Extract structural features from the paper.
2. Run the local model and get the predicted class.
3. Compare each feature with the learned feature pattern for the predicted class.
4. Compare the same feature with the other possible classes.
5. Calculate whether that feature supports the prediction or raises risk.
6. Convert the highest-risk features into reviewer-style recommendations.

Example:

```text
Feature: Experiments / evaluation section
Direction: raises risk
Recommendation: Add or strengthen experiments with datasets, metrics, baselines, and evaluation protocol.
```

## Features Explained by XAI

The XAI layer can explain features such as:

- abstract detail
- paper length
- section structure
- citation coverage
- method section
- experiments/evaluation section
- results section
- baselines
- ablation studies
- reproducibility evidence
- novelty/contribution framing
- limitations discussion
- ethics/broader impact discussion
- quantitative evidence

## How to Use XAI in the Dashboard

Run:

```bash
.venv312/bin/streamlit run dashboard_app.py
```

Then open:

```text
Review New Paper
```

Default selected model:

```text
XAI Local Review
```

This uses no OpenAI API.

If more detailed suggestions are needed, select:

```text
XAI + OpenAI Detailed Review
```

Then choose a confidentiality mode such as:

```text
abstract_only
section_summary_only
full_paper_with_consent
```

## How to Use XAI from CLI

Default XAI review:

```bash
.venv312/bin/python review_paper.py "data/Research Papers/0A9f2jZDGW.md"
```

Explicit XAI mode:

```bash
.venv312/bin/python review_paper.py "data/Research Papers/0A9f2jZDGW.md" --review-mode xai
```

XAI + OpenAI:

```bash
.venv312/bin/python review_paper.py "data/Research Papers/0A9f2jZDGW.md" --review-mode xai-openai --confidentiality-mode section_summary_only
```

Save outputs:

```bash
.venv312/bin/python review_paper.py "data/Research Papers/0A9f2jZDGW.md" --json-output reports/xai_review.json --md-output reports/xai_review.md
```

## Do We Need an XAI API Key?

No. The current XAI layer is local and does not need an API key.

OpenAI needs:

```text
OPENAI_API_KEY
```

XAI does not need:

```text
API key
paid account
external server
internet
```

## How to Explain This to Supervisor

Use this wording:

```text
The proposal-aligned default explanation method is XAI. The system first runs a local ML screening model and then uses a local XAI module to identify the most influential paper features behind the decision. Those XAI factors are converted into reviewer-style recommendations. OpenAI is retained only as an optional secondary layer for more detailed natural-language suggestions, not as the default decision or explanation method.
```

## Future XAI Improvements

The current XAI method is lightweight and dependency-free. Future versions can add:

- SHAP feature importance
- LIME explanations
- feature contribution charts
- per-section explanation scores
- comparison between XAI suggestions and OpenAI suggestions
- explanation fidelity metrics
