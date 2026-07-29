# PeerRead Review Project

Separate PeerRead-based version of the research paper screening system.

This folder does not modify the existing OpenReview-based project. It uses:

```text
data/PeerRead Set/peerread_features.csv
```

The dataset has true labels:

```text
accepted=True
accepted=False
```

Therefore this version trains a supervised accept/reject model. The dashboard still exposes three user-facing decisions:

- `Accept`: high accept probability
- `Modify`: uncertain/borderline probability
- `Reject`: high reject probability

## Run

From the repository root:

```bash
.venv312/bin/python peerread_review_project/train_peerread_model.py
.venv312/bin/python peerread_review_project/dataset_report.py
.venv312/bin/python peerread_review_project/generate_peerread_reports.py
.venv312/bin/python peerread_review_project/section_summaries.py
.venv312/bin/python peerread_review_project/create_sft_dataset.py
.venv312/bin/python peerread_review_project/poster_figures.py
.venv312/bin/python peerread_review_project/final_evaluation.py
.venv312/bin/streamlit run peerread_review_project/dashboard_app.py
```

Outputs are saved inside:

```text
peerread_review_project/models/
peerread_review_project/reports/
```

Dashboard views include:

- paper-by-paper XAI suggestions
- optional XAI vs OpenAI comparison
- dataset and final evaluation summary
- accuracy, precision, recall, F1, confusion heatmap, probability distribution, feature importance, and calibration figures
- poster/thesis-ready SVG figures

## XAI

Default explanations are local XAI explanations. No XAI API key is required.

OpenAI is not required for this PeerRead supervised model.

Optional OpenAI sample:

```bash
.venv312/bin/python peerread_review_project/top_peerread_openai.py --per-group 3
```
