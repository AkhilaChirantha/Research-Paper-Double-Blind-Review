from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path


def require_packages() -> None:
    missing = []
    for module, package in [
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("peft", "peft"),
    ]:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        joined = " ".join(missing)
        raise SystemExit(
            "Missing fine-tuning dependencies.\n"
            "Install them with:\n\n"
            f"  .venv312/bin/python -m pip install -r requirements-finetune.txt\n\n"
            f"Missing: {joined}"
        )


def read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def validate_rows(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows found in {path}")
    for index, row in enumerate(rows[:20], start=1):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"Row {index} in {path} does not contain a valid messages list")
        roles = [message.get("role") for message in messages]
        if roles[:3] != ["system", "user", "assistant"]:
            raise ValueError(f"Row {index} roles should start with system/user/assistant, got {roles[:3]}")


def render_chat(tokenizer, messages: list[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}:\n{message['content']}")
    return "\n\n".join(rendered)


def make_dataset(train_file: Path, validation_file: Path, tokenizer, max_seq_length: int, limit: int | None):
    from datasets import Dataset, DatasetDict

    train_rows = read_jsonl(train_file, limit)
    val_rows = read_jsonl(validation_file, max(1, limit // 10) if limit else None)
    validate_rows(train_rows, train_file)
    validate_rows(val_rows, validation_file)

    def encode(row: dict) -> dict:
        text = render_chat(tokenizer, row["messages"])
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return DatasetDict(
        {
            "train": Dataset.from_list(train_rows).map(encode, remove_columns=list(train_rows[0].keys())),
            "validation": Dataset.from_list(val_rows).map(encode, remove_columns=list(val_rows[0].keys())),
        }
    )


def detect_torch_dtype(torch):
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a double-blind reviewer model with LoRA.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train-file", default="data/sft/train.jsonl")
    parser.add_argument("--validation-file", default="data/sft/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/lora/double_blind_reviewer")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--limit", type=int, help="Use a small number of examples for local testing")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and tokenizer without training")
    args = parser.parse_args()

    require_packages()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    train_file = Path(args.train_file)
    validation_file = Path(args.validation_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = make_dataset(train_file, validation_file, tokenizer, args.max_seq_length, args.limit)
    print(dataset)
    print("Train example tokens:", len(dataset["train"][0]["input_ids"]))

    if args.dry_run:
        sample = tokenizer.decode(dataset["train"][0]["input_ids"][:400])
        print("\nDry run OK. Sample decoded text:\n")
        print(sample)
        return

    dtype = detect_torch_dtype(torch)
    print("Torch dtype:", dtype)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS available:", bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        trust_remote_code=False,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "logging_steps": 10,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_steps": 200,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": torch.cuda.is_available() and dtype == torch.float16,
        "bf16": torch.cuda.is_available() and dtype == torch.bfloat16,
        "gradient_checkpointing": True,
    }
    supported_args = inspect.signature(TrainingArguments.__init__).parameters
    training_kwargs = {key: value for key, value in training_kwargs.items() if key in supported_args}
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved LoRA adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
