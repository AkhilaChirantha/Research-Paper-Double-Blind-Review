# PeerRead-Based AI-Assisted Double-Blind Research Paper Review System

## 1. Project Overview

This document explains the methodology used in the new PeerRead-based version of the research paper review system. The purpose of this version is to build a supervised research paper screening workflow using a dataset that contains both accepted and rejected papers.

The previous OpenReview-based prototype used accepted papers only, so reject and modify decisions had to be treated as reviewer-risk estimates. In this new version, the PeerRead dataset contains actual `accepted=True` and `accepted=False` labels. Because of this, the system can train a supervised accept/reject classifier and then provide a practical three-level decision for users:

- `Accept`: the paper appears strong enough for submission.
- `Modify`: the paper is borderline and should be revised.
- `Reject`: the paper has high rejection risk in its current form.

The complete system includes local machine learning, XAI explanations, OpenAI-based detailed review suggestions, paper-by-paper tables, thesis-ready charts, a dashboard, and a new-paper AI agent review workflow.

## 2. Dataset

The dataset used for this version is stored separately from the earlier OpenReview project:

```text
data/PeerRead Set/peerread_features.csv
```

The dataset is not pushed to GitHub because it is large. The project instead stores the trained model and generated reports inside:

```text
peerread_review_project/
```

Verified dataset statistics:

| Item | Count |
|---|---:|
| Total papers | 4,492 |
| Accepted papers | 1,956 |
| Rejected papers | 2,536 |
| Training split | 4,028 |
| Development split | 225 |
| Test split | 239 |

Conference/source distribution:

| Source | Papers | Label Distribution |
|---|---:|---|
| `arxiv.cs.lg_2007-2017` | 2,018 | Accept: 1,113, Reject: 905 |
| `arxiv.cs.cl_2007-2017` | 1,102 | Accept: 437, Reject: 665 |
| `arxiv.cs.ai_2007-2017` | 1,023 | Accept: 259, Reject: 764 |
| `iclr_2017` | 348 | Accept: 147, Reject: 201 |
| `acl_2017` | 1 | Reject: 1 |

## 3. Why This Dataset Is Better for This Model

For a real accept/reject classifier, both positive and negative examples are required. This is similar to spam detection, where both good emails and spam emails are needed. The earlier dataset contained accepted papers only, so the model could not learn a true reject boundary. The PeerRead dataset solves that limitation because it includes true accepted and rejected examples.

This makes the PeerRead version more suitable for a thesis model because:

- it supports supervised classification,
- it allows measurable accuracy, precision, recall, F1 score, and confusion matrix,
- it gives a clearer experimental methodology,
- it allows the system to compare local model predictions against real labels,
- it produces stronger evidence for thesis evaluation.

## 4. System Architecture

The system follows this workflow:

1. Load PeerRead dataset.
2. Extract paper-level and review-level features.
3. Train a supervised accept/reject model.
4. Convert model probabilities into user-facing decisions: Accept, Modify, Reject.
5. Generate local XAI explanations.
6. Use OpenAI API for detailed LLM-based feedback on selected papers.
7. Generate charts, tables, and dashboard views.
8. Allow users to upload or paste a new paper for AI agent review.

The architecture diagram is generated at:

```text
peerread_review_project/reports/poster_figures/SYSTEM_ARCHITECTURE.svg
```

## 5. Technologies and Frameworks Used

| Tool / Framework | Purpose |
|---|---|
| Python 3.12 virtual environment | Main development and execution environment |
| Pandas | Dataset loading, tabular reports, dashboard tables |
| Streamlit | Web dashboard and interactive user interface |
| OpenAI API | LLM-based detailed feedback and paper-specific suggestions |
| `gpt-4.1-mini` | OpenAI model used for AI review generation |
| pypdf | PDF paper text extraction for new-paper upload review |
| Pure Python logistic regression | Local supervised accept/reject model |
| Local XAI feature contribution method | Explain why the local model predicts accept/reject risk |
| SVG generation | White-background thesis/poster charts |
| GitHub | Version control and deployment source |
| Streamlit Cloud | Shareable web app deployment |

## 6. Local Supervised ML Model

The local model is a dependency-light logistic regression classifier implemented in Python. It is trained using PeerRead's true `accepted` label.

Target labels:

```text
accepted=True  -> accept
accepted=False -> reject
```

The model outputs an accept probability. This probability is then converted into three practical paper-screening decisions:

| Rule | User-Facing Decision |
|---|---|
| accept probability >= 0.65 | Accept |
| 0.35 < accept probability < 0.65 | Modify |
| accept probability <= 0.35 | Reject |

The `Modify` class is not a direct dataset label. It is created as a practical decision layer for borderline papers. This is useful because real pre-submission review systems should not only say accept or reject; they should also identify papers that can become acceptable after revision.

## 7. Features Used by the Model

The model uses a combination of paper structure, reviewer metadata, and text evidence features.

### 7.1 Paper Structure Features

Examples:

- title word count,
- abstract word count,
- introduction word count,
- methodology word count,
- experiments word count,
- results word count,
- conclusion word count,
- total word count,
- section count.

Importance:

These features help the model identify whether the paper has enough academic structure. A paper with missing methodology, experiments, or conclusion sections can have higher rejection risk.

### 7.2 Boolean Section Presence Features

Examples:

- has introduction,
- has related work,
- has methodology,
- has experiments,
- has results,
- has discussion,
- has conclusion,
- has appendix.

Importance:

These features make the model sensitive to missing sections. For example, if a paper lacks experiments or result analysis, the system can recommend adding evaluation evidence.

### 7.3 Review Statistics Features

Examples:

- review count,
- average recommendation,
- average confidence,
- recommendation standard deviation,
- confidence standard deviation,
- minimum recommendation,
- maximum recommendation.

Importance:

These features capture reviewer opinion patterns in the dataset. High disagreement or low recommendation scores can indicate risk. For new uploaded papers that do not yet have reviews, reviewer-only features are treated neutrally to avoid unfairly penalizing new submissions.

### 7.4 Text Evidence Features

Examples:

- citation-like count,
- numeric result count,
- baseline terms,
- ablation terms,
- reproducibility terms,
- limitation terms,
- novelty terms,
- readability sentence length.

Importance:

These features capture academic quality signals. For example:

- baseline terms indicate comparative evaluation,
- ablation terms indicate deeper experiment analysis,
- reproducibility terms indicate implementation transparency,
- limitation terms indicate awareness of weaknesses,
- novelty terms indicate contribution framing.

## 8. XAI Methodology

The local XAI layer explains the logistic regression model using feature contribution values. Each feature receives a contribution score based on:

```text
feature contribution = model weight * scaled feature value
```

If the contribution is positive, it supports acceptance. If the contribution is negative, it supports rejection risk.

The XAI layer provides:

- key factors,
- risk factors,
- feature values,
- contribution direction,
- paper-specific recommendations.

Example XAI recommendations:

- add a clear introduction explaining problem, gap, contribution, and motivation,
- strengthen experiments with datasets, metrics, and baselines,
- improve related-work coverage with precise citations,
- add limitations, failure cases, and future work,
- improve reproducibility with code/data and hyperparameter details.

Importance:

XAI is important because the system should not only output a decision. It should explain why the decision was made and how the author can improve the paper.

## 9. OpenAI LLM Review Layer

The OpenAI layer is used as the AI reviewer/agent component. It generates natural language feedback similar to a strict but constructive double-blind academic reviewer.

Model used:

```text
gpt-4.1-mini
```

The OpenAI prompt includes:

- local model decision,
- accept probability,
- reject probability,
- XAI focus factors,
- local XAI suggestions,
- anonymized paper text or section summaries,
- review criteria.

The OpenAI output is structured as JSON with:

- `ai_decision`,
- `confidence`,
- `short_summary`,
- `good_points`,
- `weak_points`,
- `must_modify`,
- `acceptance_plan`,
- `supervisor_note`.

The `must_modify` section includes:

| Field | Meaning |
|---|---|
| `section` | The paper section that needs work |
| `problem` | What is weak or missing |
| `suggestion` | How to fix it |
| `priority` | `high`, `medium`, or `low` |

Importance:

This LLM layer satisfies the research requirement for an AI bot/reviewer that gives detailed, human-readable improvement suggestions rather than only numerical predictions.

Generated OpenAI reports:

```text
peerread_review_project/reports/peerread_openai_reviews.json
peerread_review_project/reports/peerread_openai_reviews.csv
peerread_review_project/reports/peerread_openai_reviews.html
```

## 10. Paper Selection for OpenAI Review

Because OpenAI API usage costs money, the system does not send all 4,492 papers to OpenAI. Instead, it selects a representative sample:

- best 5 Accept candidates,
- best 5 Modify candidates,
- best 5 Reject-risk papers.

Total OpenAI-reviewed papers:

```text
15 papers
```

This balances cost and research value. It provides enough examples for thesis demonstration and supervisor review while avoiding unnecessary API cost.

## 11. New Paper AI Agent Review

The dashboard includes a new-paper review page:

```text
AI Agent New Paper Review
```

The user can:

- upload a PDF, Markdown, text, or LaTeX paper,
- paste paper text directly,
- run local XAI review,
- optionally run OpenAI detailed review,
- select confidentiality mode.

The output includes:

- decision,
- quality score,
- accept/modify/reject probabilities,
- good points,
- weak points,
- must-modify suggestions,
- acceptance plan,
- XAI evidence table,
- OpenAI detailed review if enabled,
- downloadable JSON report.

This is the practical AI agent part of the project. It allows a researcher to submit a new paper before real conference submission and receive feedback on whether the paper is ready, needs modification, or is at reject risk.

## 12. Confidentiality Handling

The system supports multiple confidentiality modes:

| Mode | Meaning |
|---|---|
| `local_only` | No external API call; only local model and XAI are used |
| `abstract_only` | Only masked abstract/opening text is sent to OpenAI |
| `section_summary_only` | Masked section summaries are sent to OpenAI |
| `full_paper_with_consent` | Full masked paper is sent to OpenAI only with user consent |

Sensitive text masking removes or replaces:

- emails,
- URLs,
- ORCID identifiers,
- affiliation lines,
- acknowledgement details.

Importance:

This supports double-blind review principles and reduces confidentiality risk when using external LLM APIs.

## 13. Dashboard Views

The Streamlit dashboard contains:

| View | Purpose |
|---|---|
| Overview | Dataset/model summary and basic decision charts |
| Paper Table | Paper-by-paper local decisions and XAI suggestions |
| OpenAI Comparison | Detailed OpenAI feedback for selected PeerRead papers |
| Dataset/Evaluation | Dataset summary and model evaluation details |
| Advanced Metrics | Accuracy, precision, recall, F1, confusion matrix, calibration |
| Poster Figures | Thesis/poster-ready SVG charts |
| AI Agent New Paper Review | Upload or paste a new paper for AI review |
| Single Paper Review | Review an existing PeerRead paper by paper ID |

## 14. Generated Charts and Their Importance

All chart files are stored in:

```text
peerread_review_project/reports/poster_figures/
```

### 14.1 Predicted Decision Distribution

File:

```text
01_peerread_predicted_decisions.svg
```

Shows how many papers the model classified as Accept, Modify, and Reject.

Importance:

This chart explains the behavior of the deployed system. It shows whether the model is strict, lenient, or producing many borderline Modify decisions.

### 14.2 Actual Label Distribution

File:

```text
02_peerread_actual_labels.svg
```

Shows the real PeerRead Accept/Reject label balance.

Importance:

This chart proves that the new dataset contains both accepted and rejected papers. It also shows whether the dataset is balanced or imbalanced.

### 14.3 Classification Metrics

File:

```text
03_classification_metrics.svg
```

Shows development and test values for:

- accuracy,
- precision,
- recall,
- F1 score.

Current evaluation:

| Split | Accuracy | Accept Precision | Accept Recall | Accept F1 |
|---|---:|---:|---:|---:|
| Dev | 0.6356 | 0.5814 | 0.5208 | 0.5495 |
| Test | 0.6109 | 0.5797 | 0.3846 | 0.4624 |

Importance:

These metrics are important for thesis evaluation because they show measurable model performance.

### 14.4 Confusion Matrix Heatmap

File:

```text
04_test_confusion_heatmap.svg
```

Test confusion matrix:

| Actual / Predicted | Predicted Reject | Predicted Accept |
|---|---:|---:|
| Actual Reject | 106 | 29 |
| Actual Accept | 64 | 40 |

Importance:

The confusion matrix shows what type of mistakes the model makes. For this system, false rejects are important because accepted papers predicted as reject indicate that the model is conservative. This can be acceptable for pre-submission screening if the goal is to warn authors, but it should be discussed as a limitation.

### 14.5 Accept Probability Distribution

File:

```text
05_probability_distribution.svg
```

Shows how predicted accept probabilities are distributed for actual accepted and rejected papers.

Importance:

This chart helps explain whether the model separates accepted and rejected papers clearly or whether many papers fall into the borderline region.

### 14.6 Feature Importance

File:

```text
06_feature_importance.svg
```

Shows the top logistic regression feature weights.

Importance:

This chart supports explainability. It helps identify which features influence the model most strongly and connects directly to the XAI component.

### 14.7 Probability Calibration

File:

```text
07_probability_calibration.svg
```

Shows predicted accept probability against observed accept rate.

Importance:

This is a regression-style evaluation chart. It checks whether predicted probabilities behave like meaningful confidence scores. A well-calibrated model should have predicted probability close to real observed acceptance rate.

### 14.8 System Architecture

File:

```text
SYSTEM_ARCHITECTURE.svg
```

Shows the overall workflow from PeerRead dataset to model training, XAI, OpenAI review, reports, and dashboard.

Importance:

This figure is useful for thesis methodology and poster presentation because it explains the whole system in one visual.

## 15. Evaluation Results

Training summary:

| Item | Value |
|---|---:|
| Total rows | 4,492 |
| Train rows | 4,028 |
| Dev rows | 225 |
| Test rows | 239 |
| Accepted labels | 1,956 |
| Rejected labels | 2,536 |

Development performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.6356 |
| Accept precision | 0.5814 |
| Accept recall | 0.5208 |
| Accept F1 | 0.5495 |

Test performance:

| Metric | Value |
|---|---:|
| Accuracy | 0.6109 |
| Accept precision | 0.5797 |
| Accept recall | 0.3846 |
| Accept F1 | 0.4624 |

Interpretation:

The model provides a working baseline for supervised paper screening. The accuracy is moderate, showing that paper acceptance prediction is difficult using only lightweight engineered features. However, the system is valuable because it combines prediction with XAI and LLM-based feedback. The goal is not to replace real reviewers but to provide pre-submission decision support.

## 16. Output Files

Model:

```text
peerread_review_project/models/peerread_acceptance_model.json
```

Main decision reports:

```text
peerread_review_project/reports/peerread_decisions.json
peerread_review_project/reports/peerread_decisions.csv
peerread_review_project/reports/peerread_decisions.html
```

OpenAI detailed reports:

```text
peerread_review_project/reports/peerread_openai_reviews.json
peerread_review_project/reports/peerread_openai_reviews.csv
peerread_review_project/reports/peerread_openai_reviews.html
```

Evaluation reports:

```text
peerread_review_project/reports/training_summary.md
peerread_review_project/reports/final_evaluation_summary.md
peerread_review_project/reports/dataset_summary.md
```

SFT dataset:

```text
peerread_review_project/data/sft/train.jsonl
peerread_review_project/data/sft/validation.jsonl
peerread_review_project/data/sft/sft_peerread_reviews.jsonl
```

Poster figures:

```text
peerread_review_project/reports/poster_figures/
```

## 17. Commands Used

Train PeerRead model:

```bash
.venv312/bin/python peerread_review_project/train_peerread_model.py
```

Generate paper decision reports:

```bash
.venv312/bin/python peerread_review_project/generate_peerread_reports.py
```

Generate dataset summary:

```bash
.venv312/bin/python peerread_review_project/dataset_report.py
```

Generate section summaries:

```bash
.venv312/bin/python peerread_review_project/section_summaries.py
```

Create SFT dataset:

```bash
.venv312/bin/python peerread_review_project/create_sft_dataset.py
```

Generate poster/thesis figures:

```bash
.venv312/bin/python peerread_review_project/poster_figures.py
```

Generate final evaluation report:

```bash
.venv312/bin/python peerread_review_project/final_evaluation.py
```

Generate OpenAI detailed comparison reviews:

```bash
.venv312/bin/python peerread_review_project/top_peerread_openai.py --per-group 5 --confidentiality-mode section_summary_only
```

Run dashboard:

```bash
.venv312/bin/streamlit run peerread_review_project/dashboard_app.py
```

Review a new paper:

```bash
.venv312/bin/python peerread_review_project/review_new_paper.py /path/to/paper.pdf
```

Review a new paper with OpenAI:

```bash
.venv312/bin/python peerread_review_project/review_new_paper.py /path/to/paper.pdf --use-openai --confidentiality-mode section_summary_only
```

## 18. Research Contribution

This project contributes an AI-assisted double-blind paper screening framework that combines:

- supervised learning from true accepted/rejected paper labels,
- explainable local model predictions,
- LLM-based detailed feedback generation,
- confidentiality-aware paper handling,
- dashboard-based paper analysis,
- thesis/poster-ready reports and figures,
- reusable SFT dataset creation for future fine-tuning.

The system can be used by researchers before submission to identify whether a paper is ready, needs modification, or is at rejection risk.

## 19. Limitations

The current model has several limitations:

- The local model is a lightweight baseline and does not fully understand deep technical content.
- Accept/reject prediction is difficult because conference decisions depend on reviewer judgment, novelty, timing, and venue fit.
- The Modify class is derived from probability thresholds rather than a direct dataset label.
- OpenAI suggestions depend on the quality and amount of text sent to the API.
- Section summary mode protects confidentiality but may lose some fine technical details.
- The model should be treated as decision support, not as an official reviewer or guaranteed acceptance predictor.

## 20. Future Improvements

Future work can include:

- fine-tuning a transformer model using the generated SFT dataset,
- adding better text embeddings,
- improving probability calibration,
- evaluating with more conferences and newer datasets,
- comparing OpenAI, XAI, and future XAI API models,
- adding human expert evaluation of AI suggestions,
- improving PDF section extraction,
- adding reviewer-style scoring rubrics.

## 21. Deployment

The PeerRead dashboard can be deployed separately from the earlier OpenReview app using Streamlit Cloud.

Use this main file path:

```text
peerread_review_project/dashboard_app.py
```

This keeps the old app and new PeerRead app separate. The old app uses:

```text
dashboard_app.py
```

The new PeerRead app uses:

```text
peerread_review_project/dashboard_app.py
```

Therefore both can exist in the same GitHub repository without breaking each other.
