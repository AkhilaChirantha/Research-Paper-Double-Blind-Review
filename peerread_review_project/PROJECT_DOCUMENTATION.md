# PeerRead Separate Project Documentation

## Objective

This is a separate PeerRead-based version of the research paper screening project. It does not change the existing OpenReview-based project.

The purpose is to train a supervised model using a dataset that contains real labels:

```text
accepted=True
accepted=False
```

## Dataset

Input dataset:

```text
data/PeerRead Set/peerread_features.csv
```

Verified counts:

```text
Total papers: 4,492
Accepted: 1,956
Rejected: 2,536
Train: 4,028
Dev: 225
Test: 239
```

## Model

The model is a pure Python Gaussian probabilistic classifier.

It uses:

- paper section word counts
- review statistics
- title features
- author-count features
- boolean section-presence features
- citation-like counts
- quantitative evidence counts
- baseline / ablation / reproducibility / novelty / limitation terms

The model is trained as a supervised accept/reject classifier.

User-facing output:

- `Accept`: accept probability >= 0.65
- `Modify`: 0.35 < accept probability < 0.65
- `Reject`: accept probability <= 0.35

## XAI

The default explanation layer is local XAI.

XAI explains:

- which features support accept
- which features support reject
- what paper-specific changes are recommended

No XAI API key is needed.

## Outputs

Generated inside:

```text
peerread_review_project/models/
peerread_review_project/reports/
```

Important files:

```text
models/peerread_acceptance_model.json
reports/training_summary.md
reports/peerread_decisions.html
reports/peerread_decisions.csv
reports/peerread_decisions.json
```

## Commands

Train:

```bash
.venv312/bin/python peerread_review_project/train_peerread_model.py
```

Generate reports:

```bash
.venv312/bin/python peerread_review_project/generate_peerread_reports.py
```

Run dashboard:

```bash
.venv312/bin/streamlit run peerread_review_project/dashboard_app.py
```
