"""Create CMB hard-negative preference pairs using an SFT adapter's option scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.dpo_data import split_preference_pairs  # noqa: E402
from qwen_medical_qa.hard_negative_data import (  # noqa: E402
    build_hard_negative_record,
    option_keys,
)
from qwen_medical_qa.prompting import build_messages  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/cmb-exam-v1/train.jsonl"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/cmb-hard-negative-v1")
    )
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=Path, default=Path("outputs/qlora-cmb-v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-size", type=int, default=200)
    parser.add_argument("--max-input-samples", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain JSON objects")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_chat_template(tokenizer: Any, row: dict[str, Any]) -> str:
    messages = build_messages(row)
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str) or not rendered:
        raise ValueError(f"{row.get('id', '<unknown>')} has an empty rendered prompt")
    return rendered


def option_token_ids(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = sorted({key for row in rows for key in option_keys(row)})
    token_ids: dict[str, int] = {}
    for key in keys:
        ids = tokenizer(key, add_special_tokens=False)["input_ids"]
        if not isinstance(ids, list) or len(ids) != 1:
            raise ValueError(
                f"option {key!r} must tokenize to one token for fast hard-negative scoring; got {ids}"
            )
        token_ids[key] = int(ids[0])
    return token_ids


def score_option_batches(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    *,
    batch_size: int,
    max_input_tokens: int,
    device: Any,
) -> list[dict[str, float]]:
    import torch

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    token_ids = option_token_ids(tokenizer, rows)
    tokenizer.padding_side = "left"
    scores: list[dict[str, float]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [apply_chat_template(tokenizer, row) for row in batch_rows]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded)
            next_token_logprobs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
        for row_index, row in enumerate(batch_rows):
            keys = option_keys(row)
            scores.append(
                {
                    key: float(next_token_logprobs[row_index, token_ids[key]].item())
                    for key in keys
                }
            )
    return scores


def select_pairs(pairs: list[dict[str, Any]], max_pairs: int | None, seed: int) -> list[dict[str, Any]]:
    if max_pairs is None or max_pairs >= len(pairs):
        return list(pairs)
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    indices = sorted(random.Random(seed).sample(range(len(pairs)), max_pairs))
    return [pairs[index] for index in indices]


def main() -> None:
    args = parse_args()
    if args.eval_size <= 0:
        raise ValueError("eval-size must be positive")
    rows = load_jsonl(args.input, args.max_input_samples)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        is_trainable=False,
        local_files_only=args.local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device
    option_scores = score_option_batches(
        rows,
        tokenizer,
        model,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        device=device,
    )
    pairs = [
        build_hard_negative_record(
            row,
            scores,
            generator_model=args.model,
            generator_adapter=str(args.adapter),
        )
        for row, scores in zip(rows, option_scores, strict=True)
    ]
    pairs = select_pairs(pairs, args.max_pairs, args.seed)
    if args.eval_size >= len(pairs):
        raise ValueError(f"eval-size must be less than selected pairs ({len(pairs)})")
    train_pairs, eval_pairs = split_preference_pairs(pairs, args.eval_size, args.seed)

    output_dir = args.output_dir
    all_path = output_dir / "hard_negative_pairs.jsonl"
    train_path = output_dir / "dpo_train.jsonl"
    eval_path = output_dir / "dpo_eval.jsonl"
    write_jsonl(all_path, pairs)
    train_count = write_jsonl(train_path, train_pairs)
    eval_count = write_jsonl(eval_path, eval_pairs)
    margins = [float(pair["reference_minus_hard_negative"]) for pair in pairs]
    summary: dict[str, Any] = {
        "format": "conversational_explicit_prompt_preference",
        "preference_source": "sft_hard_negative_from_option_logprob",
        "negative_strategy": "highest_scoring_wrong_option",
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "input_rows": len(rows),
        "selected_pairs": len(pairs),
        "dpo_train_rows": train_count,
        "dpo_eval_rows": eval_count,
        "seed": args.seed,
        "generator_model": args.model,
        "generator_adapter": str(args.adapter),
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "final_cmb_val_is_not_used_for_generation_or_training": True,
        "model_top_choice_counts": dict(
            sorted(Counter(str(pair["model_top_choice"]) for pair in pairs).items())
        ),
        "model_prediction_correct": sum(bool(pair["model_prediction_correct"]) for pair in pairs),
        "model_prediction_wrong": sum(not bool(pair["model_prediction_correct"]) for pair in pairs),
        "hard_negative_is_model_top_choice": sum(
            pair["model_top_choice"] == pair["rejected_answer"] for pair in pairs
        ),
        "reference_minus_hard_negative": {
            "mean": statistics.mean(margins),
            "median": statistics.median(margins),
            "min": min(margins),
            "max": max(margins),
        },
        "raw_data_committed_to_github": False,
        "outputs": {
            "all": str(all_path),
            "train": str(train_path),
            "eval": str(eval_path),
        },
        "sha256": {
            "all": sha256_file(all_path),
            "train": sha256_file(train_path),
            "eval": sha256_file(eval_path),
        },
    }
    metadata_path = output_dir / "hard_negative_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
