# Proposal-Aligned Implementation Guide

## Project Title

**A Machine Learning and Explainable AI Framework for Prediction of Research Paper Acceptance in Academic Publications**

## 1. What the Proposal Expects

Proposal eke main goal eka:

```text
Research paper ekak submission karanna kalin accept da reject da kiyala predict karanna,
e decision eka explain karanna,
authors lata paper eka improve karanna actionable feedback denna.
```

Proposal eke research objectives:

- RO1: acceptance/rejection determine karana main features identify karanna
- RO2: LLM fine-tuning dataset ekak develop karanna
- RO3: reasoning-enabled LLM fine-tune karanna
- RO4: Agentic AI system ekak develop karanna
- RO5: XAI use karala predictions explain karanna

Current implementation eka me objectives walata align karanna puluwan, but current dataset limitation eka clearly mention karanna one.

## 2. Current Dataset Reality

Proposal eke PeerRead dataset mention karala thiyenawa, but current project eke use karanne OpenReview/NeurIPS-style dataset.

Current dataset:

```text
data/Analyse Critaria/metadata.json
data/Analyse Critaria/metadata.csv
data/Research Papers/*.md
```

Verified label situation:

```text
Accept (poster)
Accept (spotlight)
Accept (oral)
```

Actual `Reject` labels naha. Actual `Modify` final labels naha.

Therefore:

```text
Current system cannot be claimed as a fully supervised accept/reject classifier.
```

Correct claim:

```text
Current system is a risk-estimation and AI-assisted double-blind feedback framework,
using accepted papers and reviewer-score-derived pseudo labels.
```

## 3. How to Align Current Dataset with Proposal

Proposal eka accept/reject prediction kiyala thiyenawa. Dataset eke reject labels nathi nisa two-stage plan ekak use karanna.

### Stage A: Current Dataset Prototype

Use current dataset to build:

- review-quality estimation
- accept readiness prediction
- modify-needed prediction
- reject-risk estimation
- AI-generated double-blind feedback

Labels:

```text
Accept       -> strong accepted papers
Modify       -> borderline accepted papers / reviewer concerns
Reject-risk  -> weak accepted papers with low scores or high concerns
```

### Stage B: Final Research Dataset

Later add true rejected papers:

- OpenReview rejected submissions
- PeerRead accepted/rejected papers
- ICLR/NeurIPS reviews with final decisions

Then retrain as real:

```text
Accept vs Reject classifier
```

## 4. Updated Research Direction

Because current dataset has no rejected labels, project should be framed as:

```text
An AI-assisted double-blind research paper screening and feedback framework
with local risk estimation and OpenAI-based natural language explanations.
```

This still matches proposal because:

- ML prediction is included
- LLM reasoning is included
- feedback generation is included
- explainability is included
- author pre-submission support is included

XAI can be replaced temporarily by OpenAI natural-language explanations.

Later:

```text
OpenAI explanation layer -> XAI / SHAP / LIME / local fine-tuned model explanation
```

## 5. Step-by-Step Implementation Plan

## Step 01: Clean the Dataset

Goal:

```text
Remove papers that have no useful reviewer reviews.
```

Need to remove:

- author-only review rows
- papers with no reviewer rating
- papers with empty strengths/weaknesses/questions
- duplicate or incomplete rows
- metadata rows whose paper file is missing

Expected outputs:

```text
data/processed/clean_papers.json
data/processed/clean_reviews.csv
reports/dataset_cleaning_report.md
```

Current generated outputs:

```text
data/processed/clean_papers.json
data/processed/clean_reviews.csv
data/processed/double_blind_papers.jsonl
reports/dataset_cleaning_report.md
```

Current cleaning result:

```text
Metadata papers: 7,253
Missing markdown files: 1,385
Cleaned papers kept: 5,868
Good paper: 1,468
Needs modification: 3,778
Reject-risk: 622
```

This step supports:

- RO1
- co-supervisor Step 01

## Step 02: Extract Required Double-Blind Data

Goal:

```text
Only keep the data needed for double-blind feedback generation.
```

Keep:

- paper id
- title
- abstract
- paper text / section summaries
- decision
- rating
- soundness
- presentation
- contribution
- strengths
- weaknesses
- questions

Remove or mask:

- reviewer ids
- author names
- affiliations
- emails
- acknowledgements
- links that expose identity

Expected output:

```text
data/processed/double_blind_papers.jsonl
```

This has now been generated from the current dataset.

This supports:

- RO1
- RO2
- confidentiality requirement

## Step 03: Generate Double-Blind Feedbacks with AI

Goal:

```text
Use AI to combine all reviews of one paper into a single double-blind feedback.
```

For each paper:

Input:

- paper title
- abstract
- cleaned reviewer comments
- scores
- decision/risk class

Output:

- summary
- strengths
- weaknesses
- required modifications
- likely decision
- natural language explanation

For now:

```text
Use OpenAI API: gpt-4.1-mini
```

Later:

```text
Shift to XAI / DeepSeek / fine-tuned local model
```

Expected output:

```text
data/processed/generated_double_blind_feedback.jsonl
```

This supports:

- RO2
- RO3
- co-supervisor Step 02

## Step 04: Create SFT Dataset

Goal:

```text
Create supervised fine-tuning dataset for a double-blind reviewer model.
```

Format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a double-blind academic research paper reviewer."
    },
    {
      "role": "user",
      "content": "Paper summary, metadata, and criteria..."
    },
    {
      "role": "assistant",
      "content": "Double-blind feedback, strengths, weaknesses, decision, explanation..."
    }
  ]
}
```

Expected output:

```text
data/sft/sft_double_blind_reviews.jsonl
```

Current generated outputs:

```text
data/sft/sft_double_blind_reviews.jsonl
data/sft/train.jsonl
data/sft/validation.jsonl
reports/sft_dataset_report.md
```

Current SFT dataset result:

```text
Total examples: 5,868
Training examples: 5,282
Validation examples: 586
```

This supports:

- RO2
- RO3
- co-supervisor Step 03

## Step 05: PEFT / LoRA Fine-Tuning in Google Colab

Goal:

```text
Fine-tune a small open model using minimum dataset.
```

Recommended approach:

- use Google Colab
- use LoRA or QLoRA
- start with small dataset, e.g. 50-200 examples
- later expand dataset

Candidate models:

- TinyLlama
- Mistral 7B Instruct
- Llama 3.1 8B Instruct
- Qwen 2.5 7B Instruct

Expected files:

```text
notebooks/peft_finetune_colab.ipynb
models/fine_tuned_double_blind_reviewer/
```

Current VS Code implementation:

```text
scripts/train_lora.py
requirements-finetune.txt
FINETUNING_VSCODE_GUIDE.md
outputs/lora/double_blind_reviewer/
```

Current validation:

```text
Dry-run completed successfully with Qwen/Qwen2.5-1.5B-Instruct tokenizer.
SFT train/validation JSONL format is valid.
```

This supports:

- RO3
- co-supervisor Step 04

## Step 06: Section Summarization Before Screening

Goal:

```text
Summarize required paper sections before prediction/review.
```

Pipeline:

```text
Paper
-> extract sections
-> summarize Abstract
-> summarize Introduction
-> summarize Method
-> summarize Experiments
-> summarize Results
-> summarize Limitations
-> run screening model
-> generate decision + feedback
```

Why:

- reduces token cost
- improves focus
- improves confidentiality
- handles long papers better

Expected output:

```text
data/processed/section_summaries.jsonl
```

Current implementation:

```text
summarize_sections.py
research_review/section_summary.py
data/processed/section_summaries.jsonl
reports/section_summary_review_sample.json
reports/section_summary_review_sample.md
```

Current result:

```text
Papers summarized: 5,868
```

Screening can now use summaries instead of full text:

```bash
.venv312/bin/python review_paper.py "data/Research Papers/0A9f2jZDGW.md" --section-summary
```

Note: many current markdown papers include abstract and review text rather than full manuscript sections, so section extraction often finds the abstract only. The pipeline is ready for full-paper inputs when available.

This supports:

- RO4
- RO5
- co-supervisor Step 05

## Step 07: Confidentiality-Preserving System

Goal:

```text
Protect confidential research paper content.
```

System modes:

```text
local_only
abstract_only
section_summary_only
full_paper_with_consent
```

Rules:

- default should not send full paper externally
- mask author names, reviewer ids, emails, affiliations
- use explicit consent for full paper API review
- do not store API responses with sensitive identity fields
- log only paper id and processing status

This supports:

- ethics
- thesis discussion
- co-supervisor Step 06

Current implementation:

```text
research_review/confidentiality.py
CONFIDENTIALITY_GUIDE.md
review_paper.py --confidentiality-mode ...
top_papers_openai.py --confidentiality-mode ...
openai_compare.py --confidentiality-mode ...
```

The system now blocks OpenAI use in `local_only` mode and records a confidentiality audit in JSON outputs.

## Step 08: Agentic AI Workflow

Proposal mentions LangChain / LangGraph style agentic workflow.

Recommended agent pipeline:

```text
Agent 1: Paper Section Extractor
Agent 2: Anonymizer
Agent 3: Section Summarizer
Agent 4: Criteria Evaluator
Agent 5: Decision Predictor
Agent 6: Feedback Generator
Agent 7: Explanation Generator
Agent 8: Report Builder
```

Current project can implement this first as scripts.

Later convert to LangGraph.

This supports:

- RO4
- proposal method Step 3
- proposal tool development objective

## Step 09: Explainability Strategy

Proposal says XAI.

Because currently we use OpenAI API, temporary explanation method:

```text
AI-generated natural language explanation
```

But thesis should say:

```text
In the prototype, OpenAI-generated explanations are used as an explanation layer.
In the final system, XAI methods such as SHAP/LIME or feature-importance analysis can be integrated with the local model.
```

Final XAI plan:

- feature importance from local model
- section-level contribution
- reviewer-score-derived explanation
- natural language explanation by LLM

This supports:

- RO5

## 6. What Has Already Been Done

Current completed work:

- dataset loaded
- metadata analyzed
- no true reject labels verified
- local feature extractor built
- local probabilistic model trained
- paper-by-paper decision table generated
- local Accept/Modify/Reject-risk report generated
- OpenAI API integrated
- OpenAI API tested with `gpt-4.1-mini`
- poster charts generated
- advanced AI review script added
- supervisor explanation document created

Current outputs:

```text
models/research_review_model.json
reports/paper_decisions.json
reports/paper_decisions.html
reports/paper_decisions.csv
reports/poster_figures/
SUPERVISOR_EXPLANATION_DOCUMENT.md
PROJECT_DOCUMENTATION.md
```

## 7. What Still Needs To Be Done

Priority order:

1. Create dataset cleaning script.
2. Create double-blind anonymization script.
3. Generate cleaned double-blind dataset.
4. Generate AI double-blind feedback for selected papers.
5. Create SFT JSONL dataset.
6. Create Colab PEFT fine-tuning notebook.
7. Add section summarization pipeline.
8. Add confidentiality modes.
9. Add final thesis methodology diagrams.
10. Add evaluation results.

## 8. Recommended Final Thesis Framing

Because dataset has no direct reject labels, thesis should avoid overclaiming.

Recommended phrasing:

```text
This research develops an AI-assisted double-blind paper screening framework
that estimates acceptance readiness and rejection risk using reviewer-derived
signals and generates interpretable feedback using an LLM-based explanation layer.
```

If rejected papers are added later:

```text
The framework is extended into a supervised accept/reject prediction system.
```

## 9. Recommended Presentation Story

Presentation flow:

1. Peer review problem
2. Proposal goal
3. Dataset and limitation
4. Cleaning and double-blind preparation
5. Local model baseline
6. OpenAI feedback generation
7. SFT dataset creation plan
8. Fine-tuning plan
9. Confidentiality-preserving workflow
10. Results and future work

## 10. Immediate Next Task

The next implementation task should be:

```text
Build dataset cleaning + double-blind dataset generation scripts.
```

After that:

```text
Generate SFT-ready examples using OpenAI API.
```
