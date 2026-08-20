"""Run deterministic-format baseline generation and record per-sample metrics."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"id", "question"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def apply_chat_template(tokenizer: Any, question: str, thinking: bool) -> str:
    messages = [
        {
            "role": "system",
            "content": "你是一个用于研究评测的中文医疗问答模型。回答仅供学习演示，不构成医疗建议。",
        },
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        # Keep the script compatible with model/template versions that do not expose
        # the enable_thinking keyword.
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = load_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc).isoformat()
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            prompt_text = apply_chat_template(tokenizer, record["question"], args.thinking)
            inputs = tokenizer(
                [prompt_text],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": not args.greedy,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if not args.greedy:
                generation_kwargs.update(
                    {
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                    }
                )
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            input_length = inputs["input_ids"].shape[-1]
            output_ids = generated[0][input_length:]
            answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            output_tokens = int(output_ids.shape[-1])
            peak_cuda_allocated_mb = None
            peak_cuda_reserved_mb = None
            if torch.cuda.is_available():
                peak_cuda_allocated_mb = round(torch.cuda.max_memory_allocated() / 2**20, 2)
                peak_cuda_reserved_mb = round(torch.cuda.max_memory_reserved() / 2**20, 2)
            result = {
                "id": record["id"],
                "question": record["question"],
                "reference_answer": record.get("reference_answer"),
                "model": args.model,
                "thinking": args.thinking,
                "seed": args.seed,
                "prompt_tokens": int(input_length),
                "output_tokens": output_tokens,
                "latency_ms": round(elapsed * 1000, 2),
                "tokens_per_second": round(output_tokens / elapsed, 2) if elapsed else None,
                "peak_cuda_allocated_mb": peak_cuda_allocated_mb,
                "peak_cuda_reserved_mb": peak_cuda_reserved_mb,
                "answer": answer,
                "run_started_utc": run_started,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"{record['id']}: {result['latency_ms']} ms, "
                f"{result['output_tokens']} new tokens"
            )


if __name__ == "__main__":
    main()
