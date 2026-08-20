"""Run closed-domain CMB-Exam RAG generation with retrieved solved examples."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.reproducibility import configure_reproducibility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--input", type=Path, default=Path("data/processed/cmb-rag-v1/qa_benchmark.jsonl"))
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="request deterministic torch/CUDA algorithms and record the setting",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def build_exam_rag_prompt(question: str, retrieved: list[dict[str, Any]]) -> str:
    rendered_examples = []
    for item in retrieved:
        rendered_examples.append(
            f"[{item.get('chunk_id', '')}] {item.get('title', '')}\n"
            f"{item.get('text', '')}\n"
            "来源：CMB-Exam train split"
        )
    context = "\n\n".join(rendered_examples) if rendered_examples else "[无相关训练例题]"
    return (
        "你是一个用于研究评测的中文医学考试问答模型。"
        "回答仅供学习演示，不构成医疗建议。\n"
        "下面是从 CMB-Exam train split 检索到的相似已解答例题。"
        "例题答案只用于辅助推理，不能直接当作当前问题答案。\n\n"
        f"检索到的例题：\n{context}\n\n"
        f"当前问题：\n{question}\n\n"
        "请只输出当前问题的一个选项字母，不要输出解释、前缀或其他文字。"
    )


def apply_chat_template(tokenizer: Any, prompt: str, thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
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


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0 or args.max_input_tokens <= 0:
        raise ValueError("max-new-tokens and max-input-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    reproducibility = configure_reproducibility(args.seed, args.deterministic)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = read_jsonl(args.input)
    retrieval_rows = read_jsonl(args.retrieval)
    retrieval_by_id = {str(row.get("id")): row for row in retrieval_rows}
    if len(retrieval_by_id) != len(retrieval_rows):
        raise ValueError("retrieval file contains duplicate ids")
    if args.limit is not None:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc).isoformat()
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = str(row.get("id") or "")
            retrieval = retrieval_by_id.get(row_id)
            if retrieval is None:
                raise ValueError(f"missing retrieval row for {row_id}")
            retrieved = retrieval.get("retrieved") or []
            rag_prompt = build_exam_rag_prompt(str(row["question"]), retrieved)
            prompt_text = apply_chat_template(tokenizer, rag_prompt, args.thinking)
            inputs = tokenizer(
                [prompt_text],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            started = time.perf_counter()
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": not args.greedy,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if not args.greedy:
                generation_kwargs.update({"temperature": 0.7, "top_p": 0.8, "top_k": 20})
            with torch.inference_mode():
                generated = model.generate(**inputs, **generation_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            input_length = inputs["input_ids"].shape[-1]
            output_ids = generated[0][input_length:]
            answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            peak_allocated = None
            peak_reserved = None
            if torch.cuda.is_available():
                peak_allocated = round(torch.cuda.max_memory_allocated() / 2**20, 2)
                peak_reserved = round(torch.cuda.max_memory_reserved() / 2**20, 2)
            result = {
                "id": row_id,
                "question": row["question"],
                "choices": row.get("choices", {}),
                "reference_answer": row.get("reference_answer"),
                "task_type": row.get("task_type", "single_choice"),
                "model": args.model,
                "adapter": None,
                "thinking": args.thinking,
                "seed": args.seed,
                "reproducibility": reproducibility,
                "retriever": retrieval.get("retriever", "sqlite-dense"),
                "reranker": retrieval.get("reranker"),
                "embedding_model": retrieval.get("model_name"),
                "retrieved": retrieved,
                "prompt": rag_prompt,
                "answer": answer,
                "prompt_tokens": int(input_length),
                "output_tokens": int(output_ids.shape[-1]),
                "latency_ms": round(elapsed * 1000, 2),
                "tokens_per_second": round(int(output_ids.shape[-1]) / elapsed, 2)
                if elapsed
                else None,
                "peak_cuda_allocated_mb": peak_allocated,
                "peak_cuda_reserved_mb": peak_reserved,
                "run_started_utc": run_started,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{row_id}: {result['latency_ms']} ms, answer={answer!r}")


if __name__ == "__main__":
    main()
