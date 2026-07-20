# AI-Based Blind Review System for Research Papers

## 1. Project Overview

Me project eke aim eka research paper ekak submission karanna kalin preliminary blind-review evaluation ekak karanna system ekak hadana eka.

System eka paper ekak balala:

- paper eka accept-ready da
- modify karanna one da
- reject-risk da
- paper eke hoda than monawada
- paper eke adu than monawada
- section-wise modify karanna one dewal monawada
- accept level ekata geniyanna action plan eka mokakda

kiyana dewal output karanna design karala thiyenawa.

## 2. Why This System Is Needed

Research paper ekak conference/journal ekakata submit karanna kalin authors lata objective feedback one wenawa.

Normal manual review process eka:

- time-consuming
- reviewer dependent
- late feedback labena process ekak
- paper eka submit karata passe thamai major weaknesses penenne

Me system eka pre-submission stage eke early warning system ekak wage use karanna puluwan.

## 3. Data Used

Project eke data folder eke main parts dekak thiyenawa:

```text
data/Analyse Critaria/
data/Research Papers/
```

### 3.1 Analyse Criteria

Analyse criteria folder eke metadata and reviewer feedback thiyenawa:

```text
data/Analyse Critaria/metadata.json
data/Analyse Critaria/metadata.csv
```

Meke include wenawa:

- paper id
- paper title
- venue/source
- decision
- reviewer rating
- reviewer confidence
- soundness score
- presentation score
- contribution score
- reviewer strengths
- reviewer weaknesses
- reviewer questions

### 3.2 Research Papers

Research paper files markdown format eken thiyenawa:

```text
data/Research Papers/*.md
```

Me files walin paper content and review details extract karanawa.

## 4. Important Dataset Limitation

Dataset eke direct rejected papers labels naha. Dataset eke decisions mostly accepted papers:

- Accept poster
- Accept spotlight
- Accept oral

E nisa direct binary accept/reject classifier ekak hadana eka scientifically weak wenawa.

Eka handle karanna reviewer ratings and sub-scores use karala derived classes haduwa:

- `Accept`
- `Modify`
- `Reject-risk`

Methana `Reject-risk` kiyanne actual conference reject label ekak newei. Eka reviewer score pattern walin paper eka high-risk category ekak kiyala identify karana derived class ekak.

## 5. System Architecture

System eka layers dekak use karanawa.

### 5.1 Local Criteria-Based Model

Local model eka dataset eke analyse criteria and paper structure walin decision estimate karanawa.

Me model eka fast, low-cost, and repeatable.

Local model output:

- Accept
- Modify
- Reject
- quality score
- accept probability
- modify probability
- reject probability

### 5.2 OpenAI AI Review Layer

Local model eka decision estimate karata, detailed suggestions rule-based widiyata denna madi.

E nisa OpenAI API layer ekak add karala thiyenawa.

OpenAI layer eka paper content balala:

- good points
- weak points
- must modify sections
- section-wise suggestions
- acceptance plan
- supervisor-friendly explanation

generate karanawa.

Me layer eka manually hard-code karapu suggestion system ekak newei. AI model eka paper content and local model output dekama consider karala feedback denawa.

## 6. Local Model Details

Local model eka paper content walin structural and review-related features extract karanawa.

Extract karana features:

- word count
- abstract word count
- section count
- citation count
- figure count
- table count
- equation count
- numeric results count
- abstract/introduction/method/experiments/results/conclusion sections thiyenawada
- baseline terms
- ablation terms
- reproducibility terms
- novelty terms
- limitation terms
- evaluation terms
- average sentence length

Model eka pure Python probabilistic classifier ekak widiyata implement karala thiyenawa.

Reason:

- Python 3.13 alpha environment eke NumPy binary issues awa
- dependency complexity avoid karanna
- system eka lightweight karanna

## 7. Current Local Results

Available paper files `5,868` review kala.

Local model result:

```text
Accept: 554
Modify: 5,164
Reject-risk: 150
Total: 5,868
```

Interpretation:

- Most papers need modification before submission
- Small set of papers are accept-ready
- Small set of papers have high reject risk

## 8. Paper-by-Paper Report

Paper-by-paper report eka generate karala thiyenawa.

Files:

```text
reports/paper_decisions.html
reports/paper_decisions.csv
reports/paper_decisions.json
```

Each row includes:

- paper id
- paper title
- actual dataset decision
- predicted decision
- quality score
- accept probability
- modify probability
- reject probability
- local suggestions

## 9. Why OpenAI Layer Is Added

Local model eke suggestions generic/structural nature ekak thiyenawa.

Example:

- add stronger baselines
- add ablations
- clarify method
- add limitations

But research paper review ekakata meka madi. Supervisor ta pennanna and real author feedback denna paper-specific comments one.

E nisa OpenAI API use karanawa.

OpenAI eken expected feedback:

- “This paper is strong because…”
- “Main weakness is limited baseline comparison in section X…”
- “Modify experiments section by adding…”
- “Contribution claim should be rewritten because…”
- “Reject-risk reason is…”

Me feedback eka paper-by-paper generate wenawa.

## 10. OpenAI API Configuration

OpenAI API key `.env` file eke thiyenawa.

Current configured model:

```text
gpt-4.1-mini
```

API connectivity test passed:

```text
model= gpt-4.1-mini
OpenAI API OK
```

## 11. Cost-Controlled AI Review Strategy

OpenAI paid API ekak nisa all `5,868` papers full review karana eka expensive.

E nisa controlled strategy ekak propose karanawa:

1. Local model eken all papers classify karanawa.
2. Each class eken top papers select karanawa.
3. OpenAI detailed review run karanne selected papers walata witharai.

Current planned selection:

```text
Best 5 Accept papers
Best 5 Modify papers
Best 5 Reject-risk papers
```

Total:

```text
15 papers
```

Me approach eka cost-effective and supervisor presentation walata suitable.

## 12. How Top Papers Are Selected

### 12.1 Best 5 Accept Papers

Local model `Accept` kiyala predict karapu papers walin highest quality score papers 5 select karanawa.

Purpose:

- system eka strong papers identify karanna puluwanda kiyala pennanna
- accept-ready characteristics explain karanna

### 12.2 Best 5 Modify Papers

Local model `Modify` kiyala predict karapu papers walin highest quality score papers 5 select karanawa.

Reason:

- me papers reject-risk newei
- modifications kaloth accept level ekata geniyanna puluwan
- supervisor ta practical improvement suggestions pennanna hoda category eka

### 12.3 Best 5 Reject-Risk Papers

Local model `Reject` kiyala predict karapu papers walin highest reject probability papers 5 select karanawa.

Purpose:

- major weaknesses identify karanna
- why reject-risk kiyala explain karanna
- accept level ekata enna major revision path eka denna

## 13. Advanced AI Report

Advanced AI report eka generate karanna script ekak add karala thiyenawa.

Main files:

```text
top_papers_openai.py
research_review/top_papers_openai_report.py
```

Run command:

```bash
.venv312/bin/python top_papers_openai.py --per-group 5 --max-chars 14000
```

Lower-risk abstract-only command:

```bash
.venv312/bin/python top_papers_openai.py --per-group 5 --abstract-only
```

Outputs:

```text
reports/advanced_ai_reviews.html
reports/advanced_ai_reviews.csv
reports/advanced_ai_reviews.json
```

## 14. Advanced AI Report Content

Each selected paper ekata AI report eke include wenawa:

- paper id
- title
- local decision
- AI decision
- AI confidence
- local quality score
- short paper summary
- good points
- weak points
- must modify items
- section-wise modification suggestions
- acceptance plan
- supervisor note

Example structure:

```text
Paper: X
Local Decision: Modify
AI Decision: MODIFY
Good Points:
  - ...
Weak Points:
  - ...
Must Modify:
  - Experiments: add stronger baselines...
  - Related Work: compare with closest method...
Acceptance Plan:
  1. ...
  2. ...
Supervisor Note:
  ...
```

## 15. Poster / Supervisor Presentation Outputs

Supervisor presentation ekata charts generate karala thiyenawa.

Folder:

```text
reports/poster_figures/
```

Current charts:

```text
01_local_decision_distribution.svg
02_quality_score_distribution.svg
03_decision_share.svg
```

Charts now have:

- white background
- clear labels
- larger canvas
- stronger colors
- gridlines
- poster-friendly style

After OpenAI advanced/comparison report run kalama additional charts generate karanna puluwan:

```text
04_local_vs_openai_decisions.svg
05_agreement_matrix.svg
```

## 16. Recommended Supervisor Explanation

Supervisor ta explain karanna puluwan main story eka:

1. Historical review data and review criteria use karala local decision model ekak train kala.
2. Local model eka all available papers classify kala.
3. Output classes: Accept, Modify, Reject-risk.
4. Local model fast and cheap, but suggestions generic wenna puluwan.
5. E nisa OpenAI AI review layer add kala.
6. Paid API cost control karanna all papers newei, representative top 5 papers from each class use karanawa.
7. OpenAI layer paper-specific strengths, weaknesses, modifications, and acceptance plan generate karanawa.
8. Local model and AI review results compare karala best workflow eka decide karanna puluwan.

## 17. Best Workflow

Recommended practical workflow:

```text
Step 1: Local model all papers classify karanawa.
Step 2: Accept/Modify/Reject categories identify karanawa.
Step 3: Top 5 from each category select karanawa.
Step 4: OpenAI eken detailed review generate karanawa.
Step 5: Local vs AI decisions compare karanawa.
Step 6: Final submission guidance generate karanawa.
```

## 18. Why This Is Better Than Only Local Model

Only local model:

- fast
- cheap
- scalable
- but suggestions generic

OpenAI layer:

- paper-specific feedback
- section-level suggestions
- stronger explanation
- supervisor-friendly output
- better for real author guidance

Combined approach:

- local model handles scale
- OpenAI handles deep feedback
- cost remains controlled

## 19. Limitations

Current limitations:

- dataset has no direct rejected paper labels
- reject category is derived from reviewer risk signals
- local model depends on paper markdown quality
- OpenAI review has API cost
- OpenAI review sends paper text to external API
- AI feedback should be treated as decision support, not final conference decision

## 20. Next Improvements

Future improvements:

1. Add real rejected papers to dataset.
2. Retrain model with actual accept/reject labels.
3. Add PDF upload pipeline.
4. Add UI/dashboard for uploading new papers.
5. Generate final author revision checklist.
6. Add plagiarism/related work matching.
7. Add reviewer-style score prediction.
8. Add OpenAI review caching to avoid repeated API cost.

## 21. Final Summary

Me system eka now two-stage AI-assisted blind review platform ekak:

```text
Local model -> fast decision prediction
OpenAI model -> detailed paper-specific review and suggestions
```

Current final target:

```text
Best 5 Accept papers
Best 5 Modify papers
Best 5 Reject-risk papers
```

Me selected 15 papers walata OpenAI eken:

- good points
- weak points
- required modifications
- acceptance plan
- supervisor notes

generate karanawa.

Me output eka supervisor presentation ekata and project evaluation ekata use karanna suitable.
