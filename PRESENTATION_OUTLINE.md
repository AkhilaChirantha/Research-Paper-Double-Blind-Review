# Presentation Outline

## Slide 1: Title

**AI-Assisted Double-Blind Research Paper Screening Framework**

Subtitle:

```text
Acceptance-readiness prediction, rejection-risk estimation, and paper-specific feedback generation
```

## Slide 2: Research Problem

- Peer review is overloaded and inconsistent.
- Authors often receive major feedback only after submission.
- Novice researchers need pre-submission guidance.
- Existing models often focus on post-submission review data and lack practical author-facing feedback.

## Slide 3: Research Goal

Develop an AI-assisted tool that:

- estimates paper acceptance readiness
- identifies rejection risk
- highlights strengths and weaknesses
- gives paper-specific modification suggestions
- preserves confidentiality

## Slide 4: Dataset

- OpenReview-style research paper dataset
- Metadata, decisions, reviewer ratings, strengths, weaknesses, and questions
- 7,253 metadata papers
- 5,868 cleaned usable papers

Important limitation:

```text
Current final decisions include accepted papers only.
Reject is treated as reviewer-derived risk, not true final reject label.
```

## Slide 5: Proposed System Architecture

Use:

[SYSTEM_ARCHITECTURE.svg](reports/poster_figures/SYSTEM_ARCHITECTURE.svg)

Main flow:

```text
Dataset -> Cleaning -> Double-blind preparation -> Local model -> AI feedback -> Reports
```

## Slide 6: Dataset Cleaning

Cleaning actions:

- removed author-only rows
- removed missing files
- kept real reviewer reviews
- anonymized reviewers
- masked URLs/emails/identity fields

Result:

```text
Cleaned papers: 5,868
Good paper: 1,468
Needs modification: 3,778
Reject-risk: 622
```

## Slide 7: Local Screening Model

Inputs:

- structural features
- review-derived scores
- content signals

Outputs:

```text
Accept
Modify
Reject-risk
```

Local result:

```text
Accept: 554
Modify: 5,164
Reject-risk: 150
```

## Slide 8: Poster Chart Results

Use:

- `01_local_decision_distribution.svg`
- `02_quality_score_distribution.svg`
- `03_decision_share.svg`

## Slide 9: SFT Dataset

Created chat-format SFT dataset:

```text
Total examples: 5,868
Training: 5,282
Validation: 586
```

Purpose:

- fine-tune a double-blind reviewer model
- combine multiple reviews into one feedback response

## Slide 10: OpenAI Explanation Layer

Current prototype uses:

```text
gpt-4.1-mini
```

Used to generate:

- strengths
- weaknesses
- section-level modifications
- acceptance plan
- natural-language explanations

Later this can shift to:

- DeepSeek
- fine-tuned local model
- SHAP/LIME XAI

## Slide 11: Confidentiality Modes

Modes:

```text
local_only
abstract_only
section_summary_only
full_paper_with_consent
```

Why important:

- protects unpublished papers
- reduces API exposure
- supports ethical AI use

## Slide 12: Fine-Tuning Setup

Implemented:

- LoRA training script
- VS Code compatible
- Qwen2.5-1.5B-Instruct tokenizer dry-run passed
- train/validation dataset ready

Full training requires GPU resources.

## Slide 13: Evaluation Summary

Use:

```text
reports/final_evaluation_report.html
reports/final_evaluation_summary.md
reports/final_presentation_points.md
```

Key claim:

```text
The system is a working prototype for AI-assisted double-blind paper screening.
```

## Slide 14: Limitations

- no true rejected final-decision labels
- reject class is reviewer-derived risk
- many papers contain abstract/review text rather than full sections
- AI feedback is decision support, not final peer review

## Slide 15: Future Work

- add true rejected papers
- train true supervised accept/reject classifier
- complete LoRA training on GPU
- add formal XAI layer
- build UI for paper upload
- evaluate with human reviewers/authors

## Slide 16: Conclusion

The project provides:

- cleaned double-blind dataset
- local screening model
- SFT training dataset
- OpenAI explanation layer
- confidentiality-preserving modes
- supervisor-ready reports and charts

