"""Run dense retrieval followed by Qwen generation with chunk citations."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_medical_qa.rag import build_rag_prompt
from qwen_medical_qa.rag_embedding import DenseRetriever, TransformerTextEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--index", type=Path, default=Path("outputs/rag-embedding-v1/index.json"))
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rag_demo/qa_benchmark.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_queries(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        query_id = str(row.get("id") or "").strip()
        question = str(row.get("question") or "").strip()
        if not query_id or not question:
            raise ValueError(f"query at line {line_number} needs id and question")
        row["id"] = query_id
        row["question"] = question
        rows.append(row)
    if not rows:
        raise ValueError(f"no queries found in {path}")
    return rows


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
    if args.top_k <= 0 or args.max_new_tokens <= 0:
        raise ValueError("top-k and max-new-tokens must be positive")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    retriever = DenseRetriever.load(args.index)
    rows = read_queries(args.input)
    encoder = TransformerTextEncoder(
        model_name=retriever.model_name,
        device=args.device,
        max_length=retriever.max_length,
        batch_size=args.embedding_batch_size,
        query_instruction=retriever.query_instruction,
        local_files_only=args.local_files_only,
    )
    query_embeddings = encoder.encode([row["question"] for row in rows], is_query=True)
    retrieval_results = [
        retriever.search(embedding, top_k=args.top_k, min_score=args.min_score)
        for embedding in query_embeddings
    ]
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_started = datetime.now(timezone.utc).isoformat()
    with args.output.open("w", encoding="utf-8") as handle:
        for row, results in zip(rows, retrieval_results):
            rag_prompt = build_rag_prompt(row["question"], results)
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
                "id": row["id"],
                "question": row["question"],
                "expected_answer": row.get("expected_answer"),
                "relevant_doc_ids": row.get("relevant_doc_ids", []),
                "model": args.model,
                "adapter": str(args.adapter) if args.adapter else None,
                "thinking": args.thinking,
                "retriever": "dense-cosine",
                "embedding_model": retriever.model_name,
                "retrieved": [result.to_dict() for result in results],
                "prompt": rag_prompt,
                "answer": answer,
                "prompt_tokens": int(input_length),
                "output_tokens": int(output_ids.shape[-1]),
                "latency_ms": round(elapsed * 1000, 2),
                "tokens_per_second": round(int(output_ids.shape[-1]) / elapsed, 2) if elapsed else None,
                "peak_cuda_allocated_mb": peak_allocated,
                "peak_cuda_reserved_mb": peak_reserved,
                "run_started_utc": run_started,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{row['id']}: {result['latency_ms']} ms, answer={answer!r}")


if __name__ == "__main__":
    main()
