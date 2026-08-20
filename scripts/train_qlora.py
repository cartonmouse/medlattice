"""Train a small QLoRA adapter with assistant-only causal-language loss."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))


class TokenizedDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class CausalLMDataCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return config


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def tokenize_rows(rows: list[dict[str, Any]], tokenizer: Any, max_seq_length: int) -> TokenizedDataset:
    tokenized: list[dict[str, list[int]]] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"{row.get('id', '<unknown>')} must contain 3 chat messages")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"{row.get('id', '<unknown>')} must end with an assistant message")

        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            messages[:2],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) >= len(full_ids):
            raise ValueError(f"{row.get('id', '<unknown>')} has no assistant tokens")
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{row.get('id', '<unknown>')} prompt is not a full-sequence prefix")
        if len(full_ids) > max_seq_length:
            raise ValueError(
                f"{row.get('id', '<unknown>')} requires {len(full_ids)} tokens, "
                f"exceeding max_seq_length={max_seq_length}"
            )

        tokenized.append(
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :],
            }
        )
    return TokenizedDataset(tokenized)


def build_training_arguments(config: dict[str, Any], train_examples: int) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": config["output_dir"],
        "seed": int(config["seed"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "logging_steps": int(config["logging_steps"]),
        "save_steps": int(config["save_steps"]),
        "eval_steps": int(config["eval_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "fp16": bool(config["fp16"]),
        "bf16": bool(config["bf16"]),
        "optim": config.get("optim", "paged_adamw_8bit"),
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "report_to": [],
        "remove_unused_columns": False,
        "do_train": True,
        "do_eval": True,
    }
    training_argument_names = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in training_argument_names:
        kwargs["warmup_ratio"] = float(config["warmup_ratio"])
    else:
        effective_batch_size = int(config["per_device_train_batch_size"]) * int(
            config["gradient_accumulation_steps"]
        )
        updates_per_epoch = max(1, math.ceil(train_examples / effective_batch_size))
        total_steps = math.ceil(float(config["num_train_epochs"]) * updates_per_epoch)
        kwargs["warmup_steps"] = math.ceil(float(config["warmup_ratio"]) * total_steps)
    kwargs["eval_strategy"] = "steps"
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)

    import torch
    from transformers import AutoTokenizer

    random.seed(int(config["seed"]))
    torch.manual_seed(int(config["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config["seed"]))

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = load_jsonl(Path(config["train_file"]), args.max_train_samples)
    eval_rows = load_jsonl(Path(config["validation_file"]), args.max_eval_samples)
    max_seq_length = int(config["max_seq_length"])
    train_dataset = tokenize_rows(train_rows, tokenizer, max_seq_length)
    eval_dataset = tokenize_rows(eval_rows, tokenizer, max_seq_length)

    lengths = [len(row["input_ids"]) for row in train_dataset.rows + eval_dataset.rows]
    summary = {
        "model_name": config["model_name"],
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "max_sequence_length": max(lengths),
        "mean_sequence_length": round(sum(lengths) / len(lengths), 2),
        "assistant_only_loss": True,
        "cuda_available": torch.cuda.is_available(),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU; use --dry-run for CPU-side validation")

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer

    quantization = config["quantization"]
    compute_dtype = getattr(torch, str(quantization["bnb_4bit_compute_dtype"]))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    if bool(config["gradient_checkpointing"]):
        model.enable_input_require_grads()

    lora = config["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = build_training_arguments(config, len(train_dataset))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMDataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary.update(
        {
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "output_dir": str(output_dir),
        }
    )
    if torch.cuda.is_available():
        summary["peak_cuda_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        summary["peak_cuda_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20, 2)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
