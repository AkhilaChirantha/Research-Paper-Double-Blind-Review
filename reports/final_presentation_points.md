# Presentation Points

1. Problem: peer review is overloaded, inconsistent, and difficult for authors to anticipate.
2. Goal: build a pre-submission AI screening framework for research papers.
3. Dataset: 5,868 cleaned papers with real reviewer evidence.
4. Method: local reviewer-derived risk model plus LLM-based feedback generation.
5. SFT dataset: 5,868 double-blind chat-format examples.
6. Confidentiality: local-only, abstract-only, section-summary-only, and full-paper-with-consent modes.
7. Output: Accept / Modify / Reject-risk decision, strengths, weaknesses, and modification plan.
8. Limitation: current dataset has accepted papers only, so reject is treated as reviewer-derived risk.
9. Future work: add true rejected papers and complete PEFT fine-tuning on GPU.
