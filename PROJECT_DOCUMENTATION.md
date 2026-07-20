# Research Review Project Documentation

## 1. Project Goal

Me project eke main goal eka une research paper ekak submission karanna kalin blind-review style evaluation ekak karaganna system ekak hadana eka.

System eken paper ekakata:

- Accept wenna chance ekak thiyenawada
- Modify karanna one paper ekakda
- Reject-risk paper ekakda
- Modification one nam mokakda karanna one
- Reviewer-style suggestions mokakda

kiyana dewal output karanna one.

## 2. Initial Data

Project folder eke data folders dekak thibuna:

```text
data/Analyse Critaria/
data/Research Papers/
```

### 2.1 Analyse Criteria

Main criteria files:

```text
data/Analyse Critaria/metadata.json
data/Analyse Critaria/metadata.csv
```

`metadata.json` file eke paper metadata, title, abstract, decision, reviewer ratings, reviewer strengths, weaknesses, questions wage dewal thibuna.

Dataset summary:

- Total metadata papers: `7,253`
- Available markdown paper files: `5,868`
- Reviewer reviews: around `30,244`

### 2.2 Research Papers

Research papers markdown format eken thibuna:

```text
data/Research Papers/*.md
```

Paper file ekak usually me wage structure ekak thibuna:

- Title
- Source / year
- decision
- abstract
- reviewer comments
- strengths
- weaknesses
- questions

## 3. Important Dataset Limitation

Dataset eke direct rejected papers thibune na. Decisions mostly:

- `Accept (poster)`
- `Accept (spotlight)`
- `Accept (oral)`

E nisa true accept/reject supervised classifier ekak train karanna direct reject labels naha.

Meka handle karanna derived classes haduwa:

- `GOOD_PAPER`
- `NEEDS_MODIFICATION`
- `REJECT_RISK`

Me derived classes reviewer ratings, soundness, presentation, contribution wage scores walin calculate karanawa.

## 4. Local ML Model

Local ML model eka Python walin haduwa.

Main files:

```text
research_review/model.py
research_review/features.py
research_review/train.py
models/research_review_model.json
```

### 4.1 Feature Extraction

Each paper ekakata features extract karanawa. Example features:

- word count
- abstract word count
- section count
- citation count
- figure count
- table count
- equation count
- numeric result count
- abstract/introduction/method/experiments/results/conclusion sections thiyenawada
- baseline terms
- ablation terms
- reproducibility terms
- novelty terms
- limitation terms
- evaluation terms
- average sentence length

### 4.2 Model Type

Python environment eke NumPy issue ekak thibuna nisa model eka pure Python implementation ekak widiyata haduwa.

Model eka lightweight probabilistic classifier ekak:

- feature vectors log-transform karanawa
- each class eke mean/variance calculate karanawa
- probability estimate karanawa

Output classes:

```text
GOOD_PAPER
NEEDS_MODIFICATION
REJECT_RISK
```

User-friendly labels:

```text
Accept
Modify
Reject
```

### 4.3 Training

Training command:

```bash
python3 -m research_review.train
```

Generated model:

```text
models/research_review_model.json
```

Training result:

```text
trained on 5868 papers
good=1468
modify=3778
reject-risk=622
```

## 5. Single Paper Review

Single paper ekak review karanna CLI script ekak haduwa.

Main file:

```text
review_paper.py
```

Command:

```bash
python3 review_paper.py "data/Research Papers/7WTA298wts.md"
```

Outputs:

- local verdict
- quality score
- class probabilities
- structural gaps
- raw features

Example generated reports:

```text
reports/sample_local_review.md
reports/sample_local_review.json
```

## 6. Paper-by-Paper Decision Report

User request eka anuwa, dataset eke paper ekin eka balala table ekak generate karanna script ekak haduwa.

Main files:

```text
paper_decisions.py
research_review/paper_decision_report.py
```

Command:

```bash
python3 paper_decisions.py
```

Generated files:

```text
reports/paper_decisions.html
reports/paper_decisions.csv
reports/paper_decisions.json
```

### 6.1 Table Columns

Paper-by-paper table eke columns:

- `paper_id`
- `title`
- `actual_decision`
- `predicted_decision`
- `quality_score`
- `accept_probability`
- `modify_probability`
- `reject_probability`
- `suggestions`

### 6.2 Final Local Counts

Available `5,868` paper files classify kala.

Results:

```text
Accept: 554
Modify: 5,164
Reject: 150
Total: 5,868
```

### 6.3 Missing Files

Metadata eke `7,253` papers thibunath markdown paper files available thibune `5,868`.

Default report eka missing paper files skip karanawa.

Missing ewath include karanna:

```bash
python3 paper_decisions.py --include-missing
```

## 7. XAI-Based Suggestions

Default suggestions generate wenne XAI layer eken. XAI layer eka local model prediction eka explain karala, paper-specific risk factors identify karanawa.

Examples:

- Add a clear Abstract section.
- Add a focused Introduction with problem, gap, and contribution.
- Clarify the method/approach.
- Add or strengthen Experiments/Evaluation section.
- Show concrete results and analysis.
- Compare against stronger baselines.
- Include ablations or sensitivity analysis.
- Add limitations/failure cases.

Important: `reports/paper_decisions.json` eke default suggestions OpenAI API eken generate karapu ewa newei. Ewa local XAI-based suggestions. OpenAI is optional for extra detailed feedback only.

## 8. Dataset Summary Report

Dataset-level summary report ekak haduwa.

Main files:

```text
dataset_report.py
research_review/dataset_report.py
```

Command:

```bash
python3 dataset_report.py
```

Generated files:

```text
reports/dataset_report.html
reports/dataset_summary.md
```

Meke include une:

- decision distribution
- reviewer rating distribution
- derived quality classes
- common reviewer concerns
- weakness/question terms
- suggestions

## 9. Poster Graphs

Poster presentation ekata use karanna result graphs generate karanna script ekak haduwa.

Main files:

```text
poster_figures.py
research_review/poster_figures.py
```

Command:

```bash
python3 poster_figures.py
```

Generated folder:

```text
reports/poster_figures/
```

Generated figures:

```text
01_local_decision_distribution.svg
02_quality_score_distribution.svg
03_decision_share.svg
poster_figure_notes.md
```

### 9.1 Chart Improvements

Initial charts transparent background thibuna nisa poster walata clear nathi una.

Eka fix karala:

- white background add kala
- 1280x720 canvas use kala
- bigger labels
- stronger colors
- axis/gridlines
- chart border
- clearer text

## 10. OpenAI API Integration

OpenAI API support ekak add kala.

Main files:

```text
research_review/openai_reviewer.py
research_review/openai_decision_compare.py
openai_compare.py
```

### 10.1 API Key

`.env` file eke:

```text
OPENAI_API_KEY=...
```

Optional model:

```text
OPENAI_MODEL=gpt-4.1-mini
```

Default model:

```text
gpt-4.1-mini
```

### 10.2 Python Environment

Default system Python eka Python 3.13 alpha wage thibuna nisa NumPy/Pydantic binary issues awa.

E nisa stable Python 3.12 environment ekak haduwa:

```text
.venv312/
```

Install command:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv312
.venv312/bin/python -m pip install -r requirements.txt
```

### 10.3 OpenAI API Test

Minimal API ping test ekak run kala. Paper data yawwe na.

Result:

```text
model= gpt-4.1-mini
OpenAI API OK
```

So OpenAI API key, SDK, model access ok.

## 11. OpenAI Comparison

Local model vs OpenAI results compare karanna script ekak haduwa.

Command:

```bash
.venv312/bin/python openai_compare.py --per-class 10 --max-chars 12000
```

Safer abstract-only mode:

```bash
.venv312/bin/python openai_compare.py --per-class 10 --abstract-only
```

Generated files after running:

```text
reports/openai_comparison.html
reports/openai_comparison.csv
reports/openai_comparison.json
```

After OpenAI comparison JSON exists, poster figures regenerate kalama extra graphs generate wenawa:

```bash
python3 poster_figures.py
```

Extra figures:

```text
04_local_vs_openai_decisions.svg
05_agreement_matrix.svg
```

### 11.1 Privacy / Cost Note

OpenAI comparison run karanna paper text OpenAI API ekata yawanna one.

E nisa:

- cost incur wenna puluwan
- paper content external API ekata yanawa
- safer option eka `--abstract-only`

Codex side eken full paper API call eka blocked una privacy/cost reason nisa. But user local terminal eken explicit decision ekak ganna puluwan.

## 12. Requirements

Requirements file:

```text
requirements.txt
```

Current dependencies:

```text
openai>=2.0.0
pypdf>=4.0.0
```

NumPy/Pandas/Matplotlib avoid kala, environment issues reduce karanna.

Charts pure SVG walin generate wenawa.

## 13. Git Ignore

`.gitignore` add kala.

Ignored:

```text
__pycache__/
*.py[cod]
.venv/
models/
reports/
.env
```

Note: `.venv312/` add karanna onenam `.gitignore` ekata add karanna puluwan.

## 14. Main Commands

### Train model

```bash
python3 -m research_review.train
```

### Review one paper locally

```bash
python3 review_paper.py "data/Research Papers/7WTA298wts.md"
```

### Review one paper with OpenAI

```bash
.venv312/bin/python review_paper.py "data/Research Papers/7WTA298wts.md" --use-openai
```

### Generate paper-by-paper table

```bash
python3 paper_decisions.py
```

### Generate dataset summary

```bash
python3 dataset_report.py
```

### Generate poster figures

```bash
python3 poster_figures.py
```

### OpenAI comparison sample

```bash
.venv312/bin/python openai_compare.py --per-class 10 --abstract-only
```

## 15. Current Project Outputs

Important generated outputs:

```text
models/research_review_model.json
reports/paper_decisions.html
reports/paper_decisions.csv
reports/paper_decisions.json
reports/dataset_report.html
reports/dataset_summary.md
reports/poster_figures/01_local_decision_distribution.svg
reports/poster_figures/02_quality_score_distribution.svg
reports/poster_figures/03_decision_share.svg
```

## 16. Best Way Forward

Next best steps:

1. OpenAI comparison sample run karanna, preferably `--abstract-only` walin start karanna.
2. Local vs OpenAI agreement compare karanna.
3. If OpenAI suggestions significantly better nam only `Modify` and `Reject` papers walata OpenAI detailed review run karanna.
4. Real rejected papers dataset ekak add karanna.
5. Model retrain karanna accepted + rejected data ekka.
6. Poster presentation eke local results + OpenAI comparison charts use karanna.

## 17. Summary

Me project eke dan thiyenne:

- Local ML-style paper classifier
- Paper-by-paper Accept/Modify/Reject table
- Local modification suggestions
- Dataset-level analytics
- Poster-ready white-background SVG charts
- OpenAI API integration
- Local vs OpenAI comparison script
- OpenAI API verified working

Current local model results:

```text
Total papers checked: 5,868
Accept: 554
Modify: 5,164
Reject: 150
```

OpenAI API:

```text
Working: Yes
Model: gpt-4.1-mini
```
