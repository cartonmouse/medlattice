"""Run model-in-the-loop Constitutional AI-inspired v1 locally."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.constitutional import critique_answer  # noqa: E402
from qwen_medical_qa.constitutional_v1 import (  # noqa: E402
    apply_rule_hard_gate,
    build_candidate_messages,
    build_critic_messages,
    build_revision_messages,
    parse_critic_response,
    score_v1_details,
)
from qwen_medical_qa.reproducibility import configure_reproducibility  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/constitutional/benchmark-v1.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/constitutional-v1/results.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/constitutional-v1/summary.json"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--candidate-source",
        choices=("benchmark", "model"),
        default="benchmark",
        help="use curated benchmark answers or ask Qwen to generate candidates",
    )
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--candidate-max-new-tokens", type=int, default=128)
    parser.add_argument("--critic-max-new-tokens", type=int, default=256)
    parser.add_argument("--revision-max-new-tokens", type=int, default=128)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="request deterministic torch/CUDA algorithms and record the setting",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="load the base model with 4-bit NF4 quantization",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not row.get("id") or not row.get("question"):
                raise ValueError(f"{path}:{line_number} requires id and question")
            rows.append(row)
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": "auto",
        "device_map": "auto",
        "local_files_only": args.local_files_only,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    return model, tokenizer, torch


def generate_text(
    model: Any,
    tokenizer: Any,
    torch: Any,
    messages: list[dict[str, str]],
    *,
    thinking: bool,
    max_new_tokens: int,
    max_input_tokens: int,
    greedy: bool,
) -> dict[str, Any]:
    prompt_text = apply_chat_template(tokenizer, messages, thinking)
    inputs = tokenizer(
        [prompt_text],
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    ).to(model.device)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if not greedy:
        generation_kwargs.update({"temperature": 0.2, "top_p": 0.9})
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    elapsed = time.perf_counter() - started
    input_length = inputs["input_ids"].shape[-1]
    output_ids = generated[0][input_length:]
    text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return {
        "text": text,
        "prompt_tokens": int(input_length),
        "output_tokens": int(output_ids.shape[-1]),
        "latency_ms": round(elapsed * 1000, 2),
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    reproducibility = configure_reproducibility(args.seed, args.deterministic)
    rows = read_rows(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]
    model, tokenizer, torch = load_model(args)
    details: list[dict[str, Any]] = []
    run_started = datetime.now(timezone.utc).isoformat()

    for row in rows:
        context = row.get("context") or []
        requires_context = bool(row.get("requires_context", False))
        candidate_generation = None
        if args.candidate_source == "model":
            candidate_generation = generate_text(
                model,
                tokenizer,
                torch,
                build_candidate_messages(row["question"], context=context),
                thinking=args.thinking,
                max_new_tokens=args.candidate_max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                greedy=args.greedy,
            )
            candidate_answer = candidate_generation["text"]
        else:
            candidate_answer = str(row.get("answer") or "").strip()

        initial_rule_critique = critique_answer(
            row["question"],
            candidate_answer,
            context=context,
            requires_context=requires_context,
        )
        critic_generation = generate_text(
            model,
            tokenizer,
            torch,
            build_critic_messages(
                row["question"],
                candidate_answer,
                context=context,
                requires_context=requires_context,
            ),
            thinking=args.thinking,
            max_new_tokens=args.critic_max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            greedy=args.greedy,
        )
        model_critique = parse_critic_response(critic_generation["text"])
        revision_generation = generate_text(
            model,
            tokenizer,
            torch,
            build_revision_messages(
                row["question"],
                candidate_answer,
                model_critique,
                context=context,
                requires_context=requires_context,
            ),
            thinking=args.thinking,
            max_new_tokens=args.revision_max_new_tokens,
            max_input_tokens=args.max_input_tokens,
            greedy=args.greedy,
        )
        model_revision = revision_generation["text"]
        gate = apply_rule_hard_gate(
            row["question"],
            candidate_answer,
            model_revision,
            context=context,
            requires_context=requires_context,
        )
        detail = {
            "id": row["id"],
            "category": row.get("category"),
            "split": row.get("split"),
            "family_id": row.get("family_id"),
            "question": row["question"],
            "expected_initial_violation": (
                row.get("expected_initial_violation")
                if args.candidate_source == "benchmark"
                else None
            ),
            "benchmark_expected_initial_violation": row.get(
                "expected_initial_violation"
            ),
            "expected_final_safe": row.get("expected_final_safe"),
            "model": args.model,
            "adapter": str(args.adapter) if args.adapter else None,
            "candidate_source": args.candidate_source,
            "thinking": args.thinking,
            "greedy": args.greedy,
            "seed": args.seed,
            "reproducibility": reproducibility,
            "candidate_answer": candidate_answer,
            "candidate_generation": candidate_generation,
            "initial_rule_critique": initial_rule_critique,
            "critic_generation": critic_generation,
            "model_critique": model_critique,
            "model_revision": model_revision,
            "revision_generation": revision_generation,
            "model_revision_applied": model_revision.strip() != candidate_answer.strip(),
            "model_revision_critique": gate["model_revision_critique"],
            "hard_gate_fallback_applied": gate["fallback_applied"],
            "final_answer": gate["answer"],
            "final_critique": gate["final_critique"],
            "run_started_utc": run_started,
        }
        details.append(detail)
        print(
            f"{row['id']}: source={args.candidate_source}, "
            f"critic_passed={model_critique['passed']}, "
            f"fallback={gate['fallback_applied']}"
        )

    summary = score_v1_details(details)
    summary.update(
        {
            "input": str(args.input),
            "output": str(args.output),
            "summary": str(args.summary),
            "model": args.model,
            "adapter": str(args.adapter) if args.adapter else None,
            "candidate_source": args.candidate_source,
            "thinking": args.thinking,
            "greedy": args.greedy,
            "local_files_only": args.local_files_only,
            "load_in_4bit": args.load_in_4bit,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
