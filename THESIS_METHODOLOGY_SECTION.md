# Methodology

## 1. Overview

This research develops an AI-assisted double-blind research paper screening framework for pre-submission evaluation. The system is designed to estimate acceptance readiness, identify rejection risk, and generate actionable feedback for authors before submitting a manuscript to an academic venue.

The framework follows a multi-stage pipeline:

```text
OpenReview-style dataset
-> dataset cleaning
-> double-blind anonymization
-> local screening model
-> XAI feature explanation
-> section summarization
-> confidentiality-aware AI review
-> SFT dataset creation
-> PEFT/LoRA fine-tuning setup
-> final evaluation and reporting
```

The original proposal focused on a machine learning and explainable AI framework for predicting research paper acceptance readiness. In this implementation, the default explanation component is a local XAI layer that explains the screening model using feature-level evidence. The OpenAI API is retained only as an optional secondary layer for richer natural-language feedback when the user explicitly requests it.

## 2. Dataset Collection

The dataset used in this implementation was collected from an OpenReview-style source and contains research paper metadata, reviewer comments, ratings, confidence values, and paper markdown files.

The main dataset folders are:

```text
data/Analyse Critaria/
data/Research Papers/
```

The metadata includes:

- paper ID
- title
- venue/source
- year
- final decision
- reviewer rating
- reviewer confidence
- soundness score
- presentation score
- contribution score
- reviewer summary
- strengths
- weaknesses
- questions

## 3. Dataset Limitation

An important limitation discovered during dataset analysis is that the available final decision labels contain accepted papers only:

```text
Accept (poster)
Accept (spotlight)
Accept (oral)
```

The dataset does not contain direct final-decision labels for rejected papers. Therefore, this prototype does not claim to be a fully supervised accept/reject classifier. Instead, reviewer-score-derived pseudo-labels are used to estimate three practical screening classes:

```text
Accept
Modify
Reject-risk
```

This framing is suitable for a pre-submission screening assistant because the purpose is not to replace conference reviewers, but to identify paper readiness and provide improvement guidance.

## 4. Dataset Cleaning

The first processing stage removes unusable or non-informative records. Author-only rows and papers without real reviewer evidence are excluded. The system keeps only reviewer comments that include meaningful text and numeric rating information.

Cleaning rules:

- remove author-only review rows
- remove missing markdown files
- keep only real reviewer reviews
- mask reviewer identities
- mask emails, URLs, OpenReview links, ORCID-like IDs, and affiliation lines
- remove review blocks from paper text when preparing manuscript-only input

Generated outputs:

```text
data/processed/clean_papers.json
data/processed/clean_reviews.csv
data/processed/double_blind_papers.jsonl
reports/dataset_cleaning_report.md
```

Cleaning result:

```text
Metadata papers: 7,253
Missing markdown files: 1,385
Cleaned papers kept: 5,868
Good paper: 1,468
Needs modification: 3,778
Reject-risk: 622
```

## 5. Double-Blind Data Preparation

The cleaned dataset is transformed into a double-blind format. Reviewer IDs are replaced with anonymous reviewer labels, and identity-revealing fields are masked.

Example:

```text
Reviewer_qb7d -> Anonymous Reviewer 1
email@example.com -> [EMAIL]
https://... -> [URL]
```

The double-blind dataset stores:

- anonymized paper text
- anonymized reviewer comments
- reviewer score statistics
- derived paper label
- review count

This data is used for both local screening and SFT dataset generation.

## 6. Local Screening Model

A lightweight local probabilistic model was developed to estimate paper readiness. The model uses manuscript and review-derived features rather than relying only on the final decision label.

Extracted features include:

- word count
- abstract length
- section count
- citation count
- figure count
- table count
- numeric result count
- presence of abstract, introduction, method, experiments, results, limitations, and conclusion
- baseline-related terms
- ablation-related terms
- reproducibility terms
- novelty terms
- evaluation terms
- average sentence length

The local model predicts:

```text
Accept
Modify
Reject-risk
```

The paper-by-paper output is stored in:

```text
reports/paper_decisions.json
reports/paper_decisions.csv
reports/paper_decisions.html
```

Local screening result:

```text
Accept: 554
Modify: 5,164
Reject-risk: 150
Total checked: 5,868
```

## 7. Section Summarization Before Screening

To reduce token usage and improve confidentiality, the system includes a section summarization stage before screening or AI review.

The system attempts to extract and summarize:

- Abstract
- Introduction
- Related Work
- Method
- Experiments
- Results
- Limitations
- Conclusion

Generated output:

```text
data/processed/section_summaries.jsonl
```

This step supports long-document handling and allows the system to review only the most relevant content instead of the full manuscript.

## 8. Confidentiality-Preserving Review Modes

Because research manuscripts can be confidential, the system includes explicit confidentiality modes.

Available modes:

```text
local_only
abstract_only
section_summary_only
full_paper_with_consent
```

Mode descriptions:

- `local_only`: no external API is used
- `abstract_only`: only the abstract/opening text is sent to the AI API
- `section_summary_only`: only section summaries are sent to the AI API
- `full_paper_with_consent`: masked full paper text is sent only when the author explicitly agrees

The system blocks OpenAI usage in `local_only` mode and records a confidentiality audit in JSON outputs.

## 9. XAI-Assisted Explanation Layer

The default explanation method is a local XAI layer. After the local screening model predicts Accept, Modify, or Reject-risk, the XAI module identifies which paper features most strongly support the decision and which features raise reviewer risk.

The XAI layer explains features such as:

- abstract detail
- section structure
- citation coverage
- method section presence
- experiment/evaluation section presence
- result and quantitative evidence
- baseline comparisons
- ablation studies
- reproducibility evidence
- novelty framing
- limitations discussion

These feature-level explanations are converted into reviewer-style recommendations. For example, if the XAI layer identifies missing experiments and weak quantitative evidence as risk factors, the system recommends strengthening the evaluation section with datasets, metrics, baselines, and result analysis.

This approach is aligned with the proposal because recommendations are generated from explainable local model evidence rather than from a black-box external API by default.

## 9.1 Optional OpenAI Feedback Layer

The OpenAI API is kept as an optional secondary layer for users who need more detailed natural-language feedback. In the current implementation, the configured model is:

```text
gpt-4.1-mini
```

The AI review layer provides:

- decision explanation
- paper-specific strengths
- paper-specific weaknesses
- required modifications
- section-level suggestions
- acceptance improvement plan
- supervisor-friendly notes

This layer is not the default proposal-aligned explanation method. It is used only when the user selects the OpenAI option and chooses an allowed confidentiality mode. Future work can additionally incorporate formal XAI libraries such as SHAP or LIME.

## 10. SFT Dataset Creation

A supervised fine-tuning dataset was created from the double-blind reviews. For each paper, all reviewer reviews are combined into a single double-blind feedback target.

The SFT dataset uses chat fine-tuning format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a double-blind academic research paper reviewer."},
    {"role": "user", "content": "Paper summary, metadata, and criteria..."},
    {"role": "assistant", "content": "Double-blind feedback..."}
  ]
}
```

Generated outputs:

```text
data/sft/sft_double_blind_reviews.jsonl
data/sft/train.jsonl
data/sft/validation.jsonl
```

SFT dataset result:

```text
Total examples: 5,868
Training examples: 5,282
Validation examples: 586
```

## 11. PEFT / LoRA Fine-Tuning Setup

The project includes a VS Code-compatible LoRA fine-tuning script. Full training is GPU-intensive, so the system currently provides a validated fine-tuning pipeline rather than requiring full training on a local CPU.

Fine-tuning files:

```text
scripts/train_lora.py
requirements-finetune.txt
FINETUNING_VSCODE_GUIDE.md
```

Recommended base model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The dry-run validation confirms:

- the SFT dataset format is valid
- tokenizer loading works
- train/validation data can be mapped into tokenized training examples
- LoRA training setup is ready

## 12. Evaluation and Outputs

The system generates several evaluation and presentation outputs:

```text
reports/final_evaluation_report.html
reports/final_evaluation_summary.md
reports/final_presentation_points.md
reports/poster_figures/
```

The final evaluation summarizes:

- cleaned dataset size
- local screening results
- SFT dataset size
- confidentiality modes
- system capabilities
- limitations

## 13. Limitations

The current implementation has the following limitations:

- the dataset does not contain true rejected final-decision labels
- reject predictions are reviewer-derived risk estimates
- many markdown papers include abstracts and reviews rather than full manuscript sections
- full LoRA fine-tuning requires GPU resources
- AI-generated feedback should be used as decision support, not as a replacement for human peer review

## 14. Future Work

Future improvements include:

- adding real rejected papers from OpenReview or PeerRead
- training a true supervised accept/reject classifier
- completing PEFT/LoRA training on GPU
- integrating formal XAI methods such as SHAP or LIME
- adding a user interface for paper upload and feedback display
- evaluating the system with human reviewers or authors

## 15. Summary

The implemented system is a research-grade prototype for AI-assisted double-blind paper screening. It combines local reviewer-derived risk estimation, XAI-based feature explanations, confidentiality-preserving document processing, optional OpenAI-based natural-language feedback, SFT dataset construction, and a LoRA fine-tuning pipeline.

This methodology aligns with the proposal goal of developing a machine learning and explainable AI framework for pre-submission research paper acceptance prediction, while transparently addressing the limitation that the current dataset lacks true rejected-paper labels.
