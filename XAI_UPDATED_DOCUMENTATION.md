# XAI-Updated Project Documentation

## 1. Updated System Position

The project is now aligned with the proposal by making XAI the default explanation and suggestion method.

Current final architecture:

```text
Paper
-> Local ML screening model
-> XAI explanation layer
-> XAI-based suggestions
-> Optional OpenAI detailed feedback
```

OpenAI is not the default model. It is only an optional secondary layer for users who need more detailed natural-language suggestions.

## 2. Why XAI Is the Default

The proposal mentions XAI-based explanations. Therefore, the system should not depend on OpenAI as the main explanation method.

The updated system uses:

- local model prediction
- XAI feature contribution explanation
- XAI risk factors
- XAI-based reviewer-style recommendations

This means the system can explain:

- why a paper is classified as Accept, Modify, or Reject-risk
- which paper features influenced the decision
- what modifications should be made based on those features

## 3. XAI Implementation

Main file:

```text
research_review/xai.py
```

XAI method:

```text
Local XAI feature-distance explanation
```

This method compares each paper feature against the feature pattern learned by the local model. It then identifies whether each feature supports the prediction or raises risk.

Example output:

```text
XAI Focus:
- Ablation / sensitivity analysis: 0
- Sentence complexity: 19.06
- Paper length / content volume: 2116

Suggestions:
- Add ablation or sensitivity studies.
- Improve readability by shortening dense sentences.
- Expand the manuscript to better cover motivation, method, evidence, and limitations.
```

## 4. XAI Features

The XAI layer explains features such as:

- abstract detail
- paper length
- section structure
- citation coverage
- method section
- experiments/evaluation section
- results section
- baseline comparisons
- ablation studies
- reproducibility evidence
- novelty framing
- limitations discussion
- ethics/broader impact discussion
- quantitative evidence
- sentence complexity

## 5. Updated Dashboard Behavior

Dashboard model selection:

```text
XAI Local Review
XAI + OpenAI Detailed Review
```

Default:

```text
XAI Local Review
```

The paper-by-paper table now includes:

- `xai_focus`
- `suggestion_1`
- `suggestion_2`
- `suggestion_3`
- `suggestions`

This avoids repeated generic suggestions and shows paper-specific XAI reasons.

## 6. Updated System Architecture

The system architecture figure has been updated:

```text
reports/poster_figures/SYSTEM_ARCHITECTURE.svg
```

Updated flow:

```text
OpenReview Data
-> Dataset Cleaning
-> Double-Blind Prep
-> Local Screening
-> XAI Explanation
-> Section Summaries
-> Confidential Modes
-> Final Outputs
```

Optional branch:

```text
XAI Explanation
-> Optional OpenAI
-> SFT / LoRA Setup
```

## 7. XAI vs OpenAI

XAI:

- default method
- local
- no API key
- no internet required
- proposal-aligned
- explains model decisions using feature evidence

OpenAI:

- optional
- requires API key
- external service
- useful for longer natural-language suggestions
- used only with selected confidentiality mode and user consent

## 8. Supervisor Explanation

Use this explanation:

```text
The proposal-aligned default explanation method is XAI. The system first runs a local ML screening model, then uses a local XAI module to identify the most influential paper features behind the prediction. Those XAI factors are converted into reviewer-style suggestions. OpenAI is retained only as an optional secondary layer for more detailed natural-language feedback, not as the default explanation method.
```

## 9. Commands

Regenerate reports:

```bash
.venv312/bin/python paper_decisions.py
```

Regenerate poster/system architecture figures:

```bash
.venv312/bin/python poster_figures.py
```

Run dashboard:

```bash
.venv312/bin/streamlit run dashboard_app.py
```
