"""Train a QLoRA DPO adapter from the existing CMB SFT adapter."""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.dpo_data import validate_preference_record  # noqa: E402
from qwen_medical_qa.dpo_training import render_preference_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dpo.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--sft-adapter", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
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
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            validate_preference_record(row)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def _set_if_supported(
    kwargs: dict[str, Any], supported: set[str], name: str, value: Any
) -> None:
    if name in supported:
        kwargs[name] = value


def build_training_args(config: dict[str, Any], train_examples: int) -> Any:
    """Build DPOConfig while tolerating minor TRL/Transformers API differences."""

    from trl import DPOConfig

    supported = set(inspect.signature(DPOConfig).parameters)
    kwargs: dict[str, Any] = {}
    values = {
        "output_dir": config["output_dir"],
        "seed": int(config["seed"]),
        "num_train_epochs": float(config["num_train_epochs"]),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": int(config["per_device_eval_batch_size"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "optim": config.get("optim", "paged_adamw_8bit"),
        "logging_steps": int(config["logging_steps"]),
        "save_steps": int(config["save_steps"]),
        "eval_steps": int(config["eval_steps"]),
        "save_total_limit": int(config["save_total_limit"]),
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "fp16": bool(config["fp16"]),
        "bf16": bool(config["bf16"]),
        "report_to": [],
        "remove_unused_columns": False,
        "do_train": True,
        "do_eval": True,
        "beta": float(config["beta"]),
        "loss_type": config.get("loss_type", "sigmoid"),
        "max_length": int(config["max_length"]),
        "max_prompt_length": int(config["max_prompt_length"]),
    }
    for name, value in values.items():
        _set_if_supported(kwargs, supported, name, value)

    if "warmup_ratio" in supported:
        kwargs["warmup_ratio"] = float(config["warmup_ratio"])
    if "eval_strategy" in supported:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in supported:
        kwargs["evaluation_strategy"] = "steps"
    _set_if_supported(kwargs, supported, "save_strategy", "steps")
    _set_if_supported(kwargs, supported, "logging_strategy", "steps")
    if "warmup_ratio" not in supported and "warmup_steps" in supported:
        effective_batch = int(config["per_device_train_batch_size"]) * int(
            config["gradient_accumulation_steps"]
        )
        updates = max(1, (train_examples + effective_batch - 1) // effective_batch)
        total_steps = max(1, int(float(config["num_train_epochs"]) * updates))
        kwargs["warmup_steps"] = round(float(config["warmup_ratio"]) * total_steps)
    return DPOConfig(**kwargs)


def build_summary(
    config: dict[str, Any], train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "model_name": config["model_name"],
        "policy_initialization": "existing_sft_qlora_adapter",
        "sft_adapter_dir": config["sft_adapter_dir"],
        "reference_policy": "frozen_copy_of_sft_adapter_via_peft",
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "preference_source": config.get("preference_source", "unspecified"),
        "dataset_id": config.get("dataset_id"),
        "dataset_revision": config.get("dataset_revision"),
        "external_preference_eval_file": config.get("external_preference_eval_file"),
        "final_cmb_val_is_not_used_for_dpo": True,
        "cuda_available": None,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    if args.model is not None:
        config["model_name"] = args.model
    if args.sft_adapter is not None:
        config["sft_adapter_dir"] = str(args.sft_adapter)
    if args.local_files_only:
        config["local_files_only"] = True

    train_rows = load_jsonl(Path(config["train_file"]), args.max_train_samples)
    eval_rows = load_jsonl(Path(config["validation_file"]), args.max_eval_samples)
    summary = build_summary(config, train_rows, eval_rows)
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    import torch

    summary["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA-DPO requires a CUDA GPU; use --dry-run for CPU-side validation")

    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOTrainer

    local_files_only = bool(config.get("local_files_only", False))
    tokenizer_path = config["sft_adapter_dir"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = Dataset.from_list(render_preference_rows(train_rows, tokenizer))
    eval_dataset = Dataset.from_list(render_preference_rows(eval_rows, tokenizer))

    quantization = config["quantization"]
    compute_dtype = getattr(torch, str(quantization["bnb_4bit_compute_dtype"]))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=bool(quantization["load_in_4bit"]),
        bnb_4bit_quant_type=quantization["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(quantization["bnb_4bit_use_double_quant"]),
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(
        base_model,
        config["sft_adapter_dir"],
        is_trainable=True,
        local_files_only=local_files_only,
    )
    model.config.use_cache = False
    if bool(config["gradient_checkpointing"]):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    training_args = build_training_args(config, len(train_dataset))
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }
    trainer_parameters = set(inspect.signature(DPOTrainer.__init__).parameters)
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)

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
            "transformers_version": getattr(__import__("transformers"), "__version__", None),
            "trl_version": getattr(__import__("trl"), "__version__", None),
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
