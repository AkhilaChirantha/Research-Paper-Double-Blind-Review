# SFT Dataset Creation Report

This dataset is prepared for supervised fine-tuning of a double-blind research-paper reviewer model.

## Output Files

- `data/sft/sft_double_blind_reviews.jsonl`
- `data/sft/train.jsonl`
- `data/sft/validation.jsonl`

## Counts

- Total examples: 5,868
- Training examples: 5,282
- Validation examples: 586
- Max paper characters per prompt: 8,000

## Format

Each line is a JSON object with `messages` in chat fine-tuning format.

## Note

The assistant target is created by aggregating all anonymized reviewer reviews of a paper into one double-blind feedback object.
