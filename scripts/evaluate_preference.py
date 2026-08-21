"""Score how often a model prefers chosen over rejected public responses."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.dpo_data import validate_preference_record  # noqa: E402
from qwen_medical_qa.dpo_training import _apply_chat_template, _assistant_content  # noqa: E402
from qwen_medical_qa.preference_eval import mean_completion_logprob  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=384)
    parser.add_argument("--max-samples", type=int)
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
            validate_preference_record(row)
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("tokenizer returned empty input_ids")
    return ids


def _build_input(
    tokenizer: Any,
    row: dict[str, Any],
    completion: str,
    *,
    max_length: int,
    max_prompt_length: int,
) -> tuple[list[int], int, bool]:
    prompt_text = _apply_chat_template(tokenizer, row["prompt"])
    prompt_ids = _token_ids(tokenizer, prompt_text)
    prompt_truncated = len(prompt_ids) > max_prompt_length
    if prompt_truncated:
        prompt_ids = prompt_ids[-max_prompt_length:]
    completion_ids = _token_ids(tokenizer, completion)
    completion_budget = max_length - len(prompt_ids)
    if completion_budget <= 0:
        raise ValueError(f"{row.get('id', '<unknown>')} leaves no completion budget")
    original_completion_length = len(completion_ids)
    if original_completion_length > completion_budget:
        completion_ids = completion_ids[:completion_budget]
    input_ids = prompt_ids + completion_ids
    return input_ids, len(prompt_ids), prompt_truncated or len(completion_ids) < original_completion_length


def _score_input(model: Any, input_ids: list[int], prompt_length: int, device: Any) -> float:
    import torch

    tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(input_ids=tensor)
    return mean_completion_logprob(outputs.logits, tensor, prompt_length)


def score_rows(
    rows: list[dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    max_length: int,
    max_prompt_length: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    margins: list[float] = []
    for row in rows:
        row_id = str(row["id"])
        chosen = _assistant_content(row["chosen"], row_id, "chosen")
        rejected = _assistant_content(row["rejected"], row_id, "rejected")
        chosen_ids, chosen_prompt_length, chosen_truncated = _build_input(
            tokenizer,
            row,
            chosen,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )
        rejected_ids, rejected_prompt_length, rejected_truncated = _build_input(
            tokenizer,
            row,
            rejected,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )
        chosen_logprob = _score_input(model, chosen_ids, chosen_prompt_length, device)
        rejected_logprob = _score_input(model, rejected_ids, rejected_prompt_length, device)
        margin = chosen_logprob - rejected_logprob
        margins.append(margin)
        results.append(
            {
                "id": row_id,
                "label_type": row.get("label_type"),
                "source_split": row.get("source_split"),
                "chosen_avg_logprob": chosen_logprob,
                "rejected_avg_logprob": rejected_logprob,
                "chosen_minus_rejected": margin,
                "model_prefers_chosen": margin > 0,
                "chosen_truncated": chosen_truncated,
                "rejected_truncated": rejected_truncated,
            }
        )
    positive = sum(result["model_prefers_chosen"] for result in results)
    return {
        "rows": results,
        "summary": {
            "total": len(results),
            "model_prefers_chosen": positive,
            "pairwise_preference_accuracy": positive / len(results),
            "mean_chosen_minus_rejected": mean(margins),
            "median_chosen_minus_rejected": median(margins),
            "chosen_truncated": sum(result["chosen_truncated"] for result in results),
            "rejected_truncated": sum(result["rejected_truncated"] for result in results),
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_length <= args.max_prompt_length:
        raise ValueError("max-length must be greater than max-prompt-length")
    rows = load_jsonl(args.input, args.max_samples)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("preference scoring requires a CUDA GPU")
    local_files_only = args.local_files_only
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter,
        is_trainable=False,
        local_files_only=local_files_only,
    )
    model.eval()
    device = next(model.parameters()).device
    scored = score_rows(
        rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    scored["metadata"] = {
        "input": str(args.input),
        "adapter": str(args.adapter),
        "model": args.model,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "transformers_version": getattr(__import__("transformers"), "__version__", None),
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(scored, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(scored["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
