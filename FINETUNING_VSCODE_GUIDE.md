# VS Code Fine-Tuning Guide

This step fine-tunes a small instruction model with LoRA using the SFT dataset already created in:

```text
data/sft/train.jsonl
data/sft/validation.jsonl
```

## 1. Install Fine-Tuning Dependencies

Use the Python 3.12 environment:

```bash
.venv312/bin/python -m pip install -r requirements-finetune.txt
```

If pip needs internet access, allow it when prompted.

## 2. Dry Run First

Run a tiny validation pass before training:

```bash
.venv312/bin/python scripts/train_lora.py --dry-run --limit 20
```

This checks:

- JSONL format
- chat message structure
- tokenizer compatibility
- sequence truncation

## 3. Small Local Test

If your machine has enough memory, run a tiny LoRA test:

```bash
.venv312/bin/python scripts/train_lora.py --limit 20 --epochs 0.1 --max-seq-length 1024
```

This is only a pipeline test, not the final research training.

## 4. Real Training

Recommended model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

Run:

```bash
.venv312/bin/python scripts/train_lora.py --epochs 1 --max-seq-length 2048
```

Output:

```text
outputs/lora/double_blind_reviewer/
```

## 5. Practical Note

VS Code can run the code, but fine-tuning is GPU-heavy.

Recommended workflow:

```text
VS Code -> develop and dry-run
GPU/Colab -> full training
```

This still satisfies the project requirement because the training pipeline is part of this repository and can run locally when GPU resources are available.

