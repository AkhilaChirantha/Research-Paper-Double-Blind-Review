# Supervisor Defense Document

## Project Title

AI-Assisted Double-Blind Research Paper Review and Pre-Submission Screening Framework

## 1. Main Objective

Me project eke main objective eka research paper ekak conference/journal submission karanna kalin preliminary blind-review style evaluation ekak karanna AI-assisted framework ekak build karana eka.

System eka paper ekak input widiyata ganna. Ita passe system eka:

- paper eka accept-ready da kiyala estimate karanawa
- modify karanna one paper ekakda kiyala identify karanawa
- reject-risk thiyena paper ekakda kiyala flag karanawa
- hoda than, adu than, modify wenna one than paper-wise explain karanawa
- reviewer-style suggestions and acceptance improvement plan ekak generate karanawa
- confidentiality preserve karanna local-only and limited-content OpenAI modes provide karanawa

Meka final conference decision eka replace karana system ekak newei. Meka pre-submission decision-support tool ekak.

## 2. Why This Project Is Useful

Research paper submit karanna kalin authors lata objective feedback labena eka godak important. Normal peer-review process eka:

- feedback labenna kal yano
- rejected unama thamai major weaknesses clearly penenne
- reviewer comments sometimes late and venue-specific
- authors lata submission kalin paper readiness balanna systematic tool ekak naha

Me system eka early warning system ekak wage use karanna puluwan. Paper eka submit karanna kalin:

- methodology weak da?
- experiment section madi da?
- baselines/ablations madi da?
- clarity problem thiyenawada?
- contribution clearly explain wela thiyenawada?
- reviewer questions monawa wage enna puluwanda?

kiyana dewal kalin identify karanna puluwan.

## 3. Dataset Used

Project eke dataset folders:

```text
data/Analyse Critaria/
data/Research Papers/
```

### 3.1 Analyse Criteria Folder

Main files:

```text
data/Analyse Critaria/metadata.csv
data/Analyse Critaria/metadata.json
```

Me files wala paper metadata saha reviewer evidence thiyenawa:

- paper id
- title
- final decision
- reviewer rating
- reviewer confidence
- soundness score
- presentation score
- contribution score
- reviewer strengths
- reviewer weaknesses
- reviewer questions

### 3.2 Research Papers Folder

Research paper markdown files:

```text
data/Research Papers/*.md
```

Me files wala paper content and OpenReview-style text thiyenawa.

## 4. Important Dataset Discovery

Project eka build karaddi important issue ekak found una:

Dataset eke final decision labels okkoma accepted categories:

```text
Accept (poster)
Accept (spotlight)
Accept (oral)
```

Paper-level metadata count:

```text
Accept (poster): 6,421
Accept (spotlight): 704
Accept (oral): 128
Total: 7,253
```

Available markdown papers:

```text
5,868
```

### 4.1 Why This Matters

Normally supervised ML classifier ekak train karanna true labels dekama/multiple classes one.

Example:

- spam detector ekak hadanna spam mails + good mails dekama one
- medical classifier ekak hadanna positive + negative cases dekama one
- paper acceptance classifier ekak hadanna accepted + rejected papers dekama one

Me dataset eke true rejected papers nathi nisa direct supervised accept/reject classifier ekak claim karanna scientific widiyata weak.

### 4.2 How We Handled This

Reject labels nathi nisa api project eka reframed kala:

Instead of:

```text
Final accept/reject prediction model
```

we use:

```text
Pre-submission screening and risk-estimation framework
```

Reviewer ratings, soundness, contribution, presentation wage scores use karala derived/pseudo classes haduwa:

```text
GOOD_PAPER
NEEDS_MODIFICATION
REJECT_RISK
```

User-facing labels:

```text
Accept
Modify
Reject
```

Important: `Reject` kiyana dashboard/report label eka true final conference reject label ekak newei. Eka `Reject-risk` category ekak. Thesis/presentation eke meka clearly mention karanna one.

## 5. Overall System Architecture

System pipeline eka:

```text
Raw OpenReview-style dataset
-> dataset cleaning
-> reviewer evidence filtering
-> double-blind anonymization
-> local feature extraction
-> local screening model
-> XAI feature explanation and recommendations
-> section summarization
-> optional OpenAI feedback generation
-> SFT dataset creation
-> LoRA fine-tuning setup
-> reports, charts, dashboard
```

Main components:

- Local ML model
- XAI explanation layer
- OpenAI feedback layer
- Confidentiality layer
- SFT dataset generation
- Fine-tuning setup
- Report and dashboard layer

## 6. Dataset Cleaning

Dataset cleaning step eka implement kale co-supervisor kiwwa first step ekata align wenna.

Main script:

```text
prepare_data.py
research_review/data_preparation.py
```

Run command:

```bash
.venv312/bin/python prepare_data.py
```

Generated outputs:

```text
data/processed/clean_papers.json
data/processed/clean_reviews.csv
data/processed/double_blind_papers.jsonl
reports/dataset_cleaning_report.md
```

Cleaning process:

- missing markdown papers remove kala
- real reviewer reviews nathi records exclude karanna logic add kala
- author-only rows avoid kala
- reviewer evidence collect kala
- review score statistics calculate kala
- paper text anonymize kala
- reviewer identities mask kala
- emails, URLs, ORCID-like IDs, affiliation lines mask kala
- acknowledgements remove/mask kala

Cleaning result:

```text
Metadata papers: 7,253
Missing markdown files: 1,385
Cleaned papers kept: 5,868
Good paper derived: 1,468
Needs modification derived: 3,778
Reject-risk derived: 622
```

## 7. Double-Blind Preparation

Double-blind framework eka build karanna paper and review data anonymize kala.

Examples:

```text
email@example.com -> [EMAIL]
https://openreview.net/... -> [URL]
0000-0000-0000-0000 -> [ORCID]
University / department lines -> [AFFILIATION REMOVED]
Reviewer identity -> Anonymous Reviewer 1
```

Why:

- paper confidentiality protect karanna
- reviewer identities expose wenna denna epa
- future SFT dataset eka ethically safer karanna
- supervisor kiwwa confidentiality requirement ekata align wenna

Relevant file:

```text
research_review/confidentiality.py
```

## 8. Local ML Model

Local model eka lightweight Python model ekak.

Main files:

```text
research_review/model.py
research_review/features.py
research_review/train.py
models/research_review_model.json
```

### 8.1 Why Local Model Is Needed

OpenAI API paid and limited. Every paper ekama OpenAI walata yawanna cost and confidentiality issues thiyenawa.

E nisa first screening eka local model eken karanawa:

- fast
- free after setup
- paper content local machine eke thiyenawa
- OpenAI calls reduce wenawa
- all papers scan karanna puluwan

### 8.2 Feature Extraction

Paper text walin structural features extract karanawa:

- word count
- abstract word count
- section count
- citation count
- figure count
- table count
- equation count
- numeric result count
- average sentence length
- abstract section thiyenawada
- introduction thiyenawada
- method section thiyenawada
- experiments section thiyenawada
- results section thiyenawada
- conclusion thiyenawada
- limitations discuss karalada
- baselines mention karalada
- ablations mention karalada
- reproducibility/code/data mention karalada
- novelty/contribution terms thiyenawada
- evaluation terms thiyenawada

### 8.3 Model Type

Pure Python probabilistic classifier ekak implement kala.

High-level idea:

- paper features vector ekakata convert karanawa
- feature values log-transform karanawa
- each class eke mean/variance calculate karanawa
- Bayesian-style likelihood calculation eken class probabilities estimate karanawa

Output:

```text
GOOD_PAPER
NEEDS_MODIFICATION
REJECT_RISK
```

### 8.4 Why Simple Model

Reasons:

- dataset labels true accept/reject labels newei
- current stage eka prototype/research framework stage
- local model should be explainable and fast
- dependency problems avoid karanna
- XAI layer handles default proposal-aligned explanations; OpenAI is optional for deeper natural-language feedback

## 9. Local Model Training

Training command:

```bash
.venv312/bin/python -m research_review.train
```

Generated model:

```text
models/research_review_model.json
```

Training summary:

```text
trained on 5,868 papers
good=1,468
modify=3,778
reject-risk=622
```

## 10. Paper-by-Paper Decision Report

User requirement eka une dataset eke paper ekin eka balala:

- paper name/id
- actual dataset decision
- predicted decision
- quality score
- probabilities
- modification suggestions

table ekakata denna.

Implemented files:

```text
paper_decisions.py
research_review/paper_decision_report.py
```

Run command:

```bash
.venv312/bin/python paper_decisions.py
```

Generated outputs:

```text
reports/paper_decisions.html
reports/paper_decisions.csv
reports/paper_decisions.json
```

Current report counts:

```text
Accept: 554
Modify: 5,164
Reject: 150
Total checked: 5,868
```

Interpretation:

- 554 papers model eken accept-ready candidates widiyata identify wela
- 5,164 papers modification needed category ekata watila
- 150 papers high rejection-risk widiyata flag wela

Again, `Reject` means `Reject-risk`, not true final rejected paper.

## 11. XAI Explanation Layer

Proposal eke mention karala thiyenne suggestions/explanations XAI eken gannawa kiyala. E nisa current final architecture eke default explanation method eka XAI.

Main file:

```text
research_review/xai.py
```

XAI layer eka local model prediction eka explain karanawa:

- model eka Accept / Modify / Reject-risk kiyala predict kale ai
- mona features decision eka support karanawada
- mona features risk raise karanawada
- e feature evidence walin mona modifications suggest karanna onada

XAI method:

```text
Local XAI feature-distance explanation
```

Meka external paid API ekak newei. API key one na. Local machine eke run wenawa.

XAI suggestions examples:

- experiments/evaluation section weak nam experiments strengthen karanna
- citation coverage adu nam related work strengthen karanna
- baseline terms adu nam stronger baselines add karanna
- ablation terms adu nam ablation/sensitivity analysis add karanna
- limitations missing nam limitations section add karanna

## 12. Optional OpenAI API Layer

OpenAI API use kale optional paper-specific detailed feedback generate karanna.

Current model:

```text
gpt-4.1-mini
```

Configured in:

```text
.env
OPENAI_MODEL=gpt-4.1-mini
```

Main file:

```text
research_review/openai_reviewer.py
```

### 12.1 Why OpenAI Is Kept

XAI layer eka default explanation layer eka. But sometimes userta more detailed natural-language reviewer feedback one wenna puluwan.

OpenAI layer eka add kale:

- paper-specific good points identify karanna
- weak points identify karanna
- section-wise modifications denna
- reviewer questions predict karanna
- acceptance plan generate karanna
- supervisor-friendly explanation create karanna

### 12.2 OpenAI Output

OpenAI review schema includes:

- final verdict
- confidence
- overall summary
- main reasons
- section-level suggestions
- acceptance plan
- reviewer questions

Advanced AI report includes:

```text
reports/advanced_ai_reviews.json
reports/advanced_ai_reviews.csv
reports/advanced_ai_reviews.html
```

### 12.3 Why Only Selected Papers Use OpenAI

OpenAI API paid and limited.

E nisa every paper ekama OpenAI walata yawwe naha. Instead:

- best 5 accept candidates
- best 5 modify candidates
- best 5 reject-risk candidates

select karala detailed OpenAI suggestions ganna setup kala.

This balances:

- cost
- API limits
- confidentiality
- representative comparison

## 12. Confidentiality Modes

Research papers sensitive wenna puluwan. E nisa OpenAI calls direct full paper send karana eka default nemei.

Implemented modes:

```text
local_only
abstract_only
section_summary_only
full_paper_with_consent
```

### 12.1 local_only

- no external API
- paper local machine eke witharai process wenne
- default safest mode

### 12.2 abstract_only

- masked abstract only OpenAI walata yawenawa
- low token and safer mode

### 12.3 section_summary_only

- full paper direct yawanne naha
- system eka sections summarize karala masked summary only yawenawa
- better balance between privacy and useful feedback

### 12.4 full_paper_with_consent

- masked full paper OpenAI walata yawenawa
- only use if author consent thiyenawa

Why this matters:

- double-blind process preserve karanna
- unpublished research confidentiality protect karanna
- ethical use of AI API demonstrate karanna

## 13. Section Summarization

Co-supervisor kiwwa system change ekak thibuna: screening karanna kalin required sections summarize karanna.

Implemented files:

```text
summarize_sections.py
research_review/section_summary.py
```

Generated output:

```text
data/processed/section_summaries.jsonl
```

Count:

```text
Section summaries: 5,868
```

Purpose:

- token usage reduce karanna
- OpenAI API cost reduce karanna
- paper full content expose nokara review suggestions ganna
- important sections extract karanna

## 14. SFT Dataset Creation

Co-supervisor suggestion eka anuwa SFT dataset ekak create kala.

Main files:

```text
create_sft_dataset.py
research_review/sft_dataset.py
```

Run command:

```bash
.venv312/bin/python create_sft_dataset.py
```

Generated files:

```text
data/sft/sft_double_blind_reviews.jsonl
data/sft/train.jsonl
data/sft/validation.jsonl
reports/sft_dataset_report.md
```

Counts:

```text
Total SFT examples: 5,868
Train examples: 5,282
Validation examples: 586
```

SFT format:

```text
system message
user paper screening request
assistant double-blind feedback JSON
metadata
```

Purpose:

- future custom reviewer model fine-tune karanna
- all reviews aggregate karala single double-blind feedback output ekak create karanna
- OpenAI/XAI layer replace karanna future local model path ekak provide karanna

## 15. LoRA / PEFT Fine-Tuning Setup

Co-supervisor kiwwa parameter-efficient fine-tuning step ekata setup kala.

Main files:

```text
scripts/train_lora.py
requirements-finetune.txt
FINETUNING_VSCODE_GUIDE.md
```

Recommended base model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Why Qwen 1.5B:

- smaller model
- Apple Silicon/MPS environment eke possible
- LoRA training walata manageable
- full large model fine-tune karanna GPU cost wadi

Dry-run validation completed:

- tokenizer load una
- train/validation dataset map una
- max sequence length check kala
- LoRA trainable parameter setup validate kala

Full training mandatory nemei prototype stage ekata, mokada:

- dataset and training code ready
- hardware/time limitations thiyenawa
- OpenAI layer currently feedback generation karanawa
- full fine-tuning future work / next experiment widiyata present karanna puluwan

## 16. Reports and Figures

User requested charts with white background and clear presentation quality.

Main script:

```text
poster_figures.py
research_review/poster_figures.py
```

Generated figures:

```text
reports/poster_figures/01_local_decision_distribution.svg
reports/poster_figures/02_quality_score_distribution.svg
reports/poster_figures/03_decision_share.svg
reports/poster_figures/SYSTEM_ARCHITECTURE.svg
```

Generated report files:

```text
reports/dataset_report.html
reports/dataset_summary.md
reports/paper_decisions.html
reports/paper_decisions.csv
reports/paper_decisions.json
reports/advanced_ai_reviews.html
reports/advanced_ai_reviews.csv
reports/advanced_ai_reviews.json
reports/final_evaluation_report.html
reports/final_evaluation_summary.md
reports/final_presentation_points.md
```

These are useful for:

- supervisor demo
- poster presentation
- thesis result section
- viva explanation

## 17. Dashboard Application

Next practical step widiyata dashboard app ekak add kala.

Main file:

```text
dashboard_app.py
```

Run command:

```bash
.venv312/bin/streamlit run dashboard_app.py
```

Dashboard URL:

```text
http://localhost:8501
```

Dashboard includes:

- overview metrics
- decision distribution chart
- quality score chart
- poster figures
- paper-by-paper table
- OpenAI comparison viewer
- upload new paper and review
- confidentiality mode selector
- local-only default behavior

Why dashboard:

- supervisor ta web demo ekak pennanna
- reports manually open nokara single UI ekakin results pennanna
- research output professional widiyata present karanna
- new paper upload karala system behavior demonstrate karanna

## 18. Project Files Summary

Important source files:

```text
review_paper.py
paper_decisions.py
top_papers_openai.py
prepare_data.py
create_sft_dataset.py
summarize_sections.py
poster_figures.py
final_evaluation.py
dashboard_app.py
```

Important package files:

```text
research_review/model.py
research_review/features.py
research_review/confidentiality.py
research_review/openai_reviewer.py
research_review/data_preparation.py
research_review/sft_dataset.py
research_review/section_summary.py
research_review/paper_decision_report.py
research_review/poster_figures.py
research_review/evaluation_report.py
```

Important generated artifacts:

```text
models/research_review_model.json
data/processed/
data/sft/
reports/
```

Important documentation:

```text
README.md
PROJECT_DOCUMENTATION.md
SUPERVISOR_EXPLANATION_DOCUMENT.md
PROPOSAL_ALIGNED_IMPLEMENTATION_GUIDE.md
THESIS_METHODOLOGY_SECTION.md
PRESENTATION_OUTLINE.md
CONFIDENTIALITY_GUIDE.md
FINETUNING_VSCODE_GUIDE.md
SUPERVISOR_DEFENSE_DOCUMENT.md
```

## 19. What Was Done According to Co-Supervisor Steps

### Step 01: Clean dataset from OpenReview and exclude papers with no review

Done.

Implemented:

```text
prepare_data.py
research_review/data_preparation.py
```

Output:

```text
data/processed/clean_papers.json
data/processed/clean_reviews.csv
```

### Step 02: Take required data to create double-blind outputs and natural-language explanations

Done.

Implemented:

```text
data/processed/double_blind_papers.jsonl
research_review/confidentiality.py
research_review/openai_reviewer.py
```

### Step 03: Create SFT dataset for double-blind framework

Done.

Implemented:

```text
data/sft/train.jsonl
data/sft/validation.jsonl
```

### Step 04: Parameter-efficient fine-tuning with minimal dataset

Setup done, dry-run validation done.

Implemented:

```text
scripts/train_lora.py
requirements-finetune.txt
FINETUNING_VSCODE_GUIDE.md
```

Full training can be done later on stronger GPU/Colab/MPS depending on time.

### Step 05: Update system to summarize required sections before screening

Done.

Implemented:

```text
summarize_sections.py
research_review/section_summary.py
```

### Step 06: Update system to keep confidentiality of research papers

Done.

Implemented:

```text
research_review/confidentiality.py
CONFIDENTIALITY_GUIDE.md
dashboard_app.py confidentiality selector
```

## 20. What Was Not Done Yet and Why

### 20.1 True Accept/Reject Supervised Classifier

Not fully done because current dataset does not include true rejected final labels.

Reason:

- all final decisions are accept variants
- direct accept/reject training would be misleading
- current output should be explained as risk estimation

Future improvement:

- collect rejected papers / weak-reject labels
- train supervised classifier
- evaluate with confusion matrix, precision, recall, F1-score

### 20.2 Full LoRA Fine-Tuning

Setup done, but full training not mandatory completed.

Reason:

- Apple MPS/CPU training takes long
- prototype already works with OpenAI layer
- SFT dataset and training script are ready

Future improvement:

- run on Google Colab GPU
- compare base model vs fine-tuned model
- evaluate generated feedback quality

### 20.3 Advanced XAI Tools

Current system already has a local XAI explanation layer. Advanced third-party XAI tools are future improvements:

- SHAP
- LIME
- feature contribution visualizations

Reason:

- current local XAI layer is lightweight and dependency-free
- current local model is simple and interpretable
- advanced XAI libraries can be added after real accepted/rejected labels are collected

Future improvement:

- add SHAP/LIME for feature importance
- show why model predicted Accept/Modify/Reject-risk
- include feature contribution charts in dashboard

## 21. How to Explain This to Supervisor

Short explanation:

```text
This system is an AI-assisted pre-submission double-blind review framework. 
It first cleans and anonymizes OpenReview-style data, derives practical 
screening labels from reviewer scores because the dataset does not contain 
true rejected papers, trains a local screening model, and uses a local XAI 
layer as the default explanation and suggestion mechanism. OpenAI is kept 
only as an optional extra layer for selected detailed natural-language 
feedback under confidentiality modes. It also prepares an SFT dataset and 
LoRA fine-tuning pipeline for future custom model development.
```

## 22. Expected Supervisor Questions and Answers

### Q1: Is this a real accept/reject classifier?

Answer:

Not fully. Current dataset has only accepted final decisions, so the system is framed as a pre-submission screening and rejection-risk estimation framework. It predicts Accept, Modify, and Reject-risk using reviewer-score-derived pseudo-labels.

### Q2: Why do we need OpenAI if we already have a local model?

Answer:

OpenAI is not the default model. The default is local model + XAI. OpenAI is optional for richer natural-language reviewer feedback when the user asks for more suggestions.

### Q3: Does every paper go to OpenAI?

Answer:

No. To control cost and protect confidentiality, OpenAI is used only when explicitly enabled. For dataset-level analysis, only selected top papers from each category are reviewed with OpenAI.

### Q4: How is confidentiality handled?

Answer:

The system has local-only, abstract-only, section-summary-only, and full-paper-with-consent modes. Default is local-only. Sensitive text such as emails, URLs, affiliations, ORCID-like IDs, and acknowledgements are masked.

### Q5: Why are there reject outputs if dataset has no rejected papers?

Answer:

The output label should be understood as reject-risk. It is derived from low reviewer ratings, weak sub-scores, and structural weaknesses. It is not a true final rejected-paper label.

### Q6: Can this model be improved?

Answer:

Yes. The next major improvement is to add a balanced dataset with accepted, rejected, and borderline papers. Then a stronger supervised model can be trained and evaluated with standard metrics.

### Q7: What is the AI part of the system?

Answer:

There are multiple AI parts:

- local ML screening model
- local XAI explanation and recommendation layer
- optional OpenAI API feedback generation
- SFT dataset generation for future fine-tuning
- LoRA fine-tuning setup

### Q8: Why not use only manual suggestions?

Answer:

Simple manual suggestions are generic. The XAI layer produces recommendations from the features that actually influenced the local model prediction. OpenAI is optional when the user wants longer natural-language feedback.

### Q9: Is full fine-tuning completed?

Answer:

The fine-tuning pipeline and dataset are prepared, and dry-run validation was completed. Full training can be performed later on GPU/Colab because local hardware training is slow.

### Q10: What can be shown in demo?

Answer:

The Streamlit dashboard can show:

- decision distribution
- paper table
- OpenAI comparison
- poster figures
- upload and review a new paper
- confidentiality controls

## 23. Demo Flow

Recommended supervisor demo order:

1. Open dashboard:

```bash
.venv312/bin/streamlit run dashboard_app.py
```

2. Show Overview page:

- training paper count
- Accept/Modify/Reject-risk counts
- decision distribution chart
- architecture figure

3. Show Paper Table page:

- paper-by-paper predictions
- quality score
- suggestions
- filter by Modify or Reject

4. Show OpenAI Comparison page:

- selected paper AI feedback
- good points
- weak points
- must modify section
- acceptance plan

5. Show Review New Paper page:

- upload a sample paper
- run local-only review
- explain OpenAI toggle and confidentiality modes

6. Show thesis/presentation docs:

```text
THESIS_METHODOLOGY_SECTION.md
PRESENTATION_OUTLINE.md
```

## 24. Main Strengths of Current Work

- end-to-end working prototype
- real dataset cleaning pipeline
- paper-by-paper review output
- local ML model
- XAI reviewer explanation layer
- optional OpenAI reviewer integration
- confidentiality controls
- double-blind anonymized data
- SFT dataset generation
- LoRA fine-tuning setup
- clear reports and poster figures
- web dashboard for demo
- honest handling of dataset limitations

## 25. Main Limitations

- dataset does not include true rejected final labels
- reject category is derived reject-risk, not true reject
- local model is lightweight
- optional OpenAI output depends on prompt/model behavior
- full LoRA fine-tuning not completed yet
- advanced XAI libraries such as SHAP/LIME are future work
- current markdown papers may not contain complete full-paper sections for every record

## 26. Future Work

Recommended future improvements:

- add accepted and rejected labeled papers
- train supervised classifier with real labels
- add validation metrics: accuracy, precision, recall, F1-score
- add confusion matrix
- run full LoRA fine-tuning on GPU
- compare local model, OpenAI model, and fine-tuned model
- add SHAP/LIME feature explanations
- improve PDF section extraction
- add user authentication if deployed online
- add storage for uploaded review reports
- add advanced XAI/local fine-tuned explanation model if required

## 27. Final Position of the Project

Current project status:

```text
Working advanced prototype completed.
```

It includes:

- dataset preparation
- double-blind cleaning
- local ML model
- XAI suggestion system
- optional OpenAI suggestion system
- confidentiality framework
- report generation
- figures
- SFT dataset
- LoRA setup
- Streamlit dashboard
- thesis/presentation support documents

Scientifically correct framing:

```text
AI-assisted double-blind pre-submission paper screening and feedback generation framework.
```

Avoid overclaiming:

```text
This is not yet a fully supervised final accept/reject prediction model because the dataset lacks true rejected-paper labels.
```

Best explanation:

```text
The current system estimates paper readiness and rejection risk, generates reviewer-style feedback using a default local XAI layer, keeps OpenAI as an optional extra suggestion layer, and prepares the foundation for future fine-tuning and advanced XAI explanations.
```
