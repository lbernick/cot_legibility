import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from src.inference.providers import get_provider
from src.utils.io import append_jsonl, read_jsonl, write_json


def split_solution_into_chunks(text: str, min_chunk_length: int = 10, replace_newlines: bool=False) -> list[str]:
    if not text:
        return []
    if "<think>" in text:
        text = text.split("<think>")[1]
    if "</think>" in text:
        text = text.split("</think>")[0]
    text = text.strip()

    text = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL>\2", text)
    text = re.sub(r"\n(\d)\.(\s)", r"\n\1<DECIMAL>\2", text)

    sentences = re.split(r"([!?:\n]|(?<!\n\d)\.)", text)
    chunks = []
    for i in range(0, len(sentences) - 1, 2):
        if replace_newlines:
            chunks.append((sentences[i] + sentences[i + 1]).replace("\n", " "))
        else:
            chunks.append((sentences[i] + sentences[i + 1]))

    chunks = [re.sub(r"<DECIMAL>", ".", c) for c in chunks]

    if not chunks:
        return []
    merged = [chunks[0]]
    for c in chunks[1:]:
        if len(merged[-1]) < min_chunk_length:
            merged[-1] += c
        else:
            merged.append(c)
    return [c.strip() for c in merged if c.strip()]


def process_rollout(
    result: dict,
    chunk_index: int,
    rollout_index: int,
    chunks: list[str],
    model_config: dict,
    provider,
    force_answer: bool,
) -> dict:
    base = {
        "question_id": result["question_id"],
        "run": result.get("run"),
        "dataset": result.get("dataset"),
        "sample_index": result.get("sample_index"),
        "question": result["question"],
        "correct_answer": result.get("correct_answer"),
        "original_answer": result.get("answer"),
        "original_correctness": result.get("correctness", {}).get("correctness"),
        "chunk_index": chunk_index,
        "rollout_index": rollout_index,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    try:
        if chunk_index == -1:
            prefill = "<think>\n" + ("</think>" if force_answer else "")
        else:
            prefill = f"<think>\n{' '.join(chunks[: chunk_index + 1])}" + (
                "\n</think>" if force_answer else ""
            )

        response = provider.generate(result["question"], model_config, prefill=prefill)

        metadata = {
            "duration_ms": response["duration_ms"],
            "tokens": response.get("tokens"),
        }
        for field in [
            "provider_model",
            "openrouter_provider",
            "stream_complete",
            "error",
        ]:
            if field in response:
                metadata[field] = response[field]

        return {
            **base,
            "prefill": prefill,
            "prefill_answer": response["answer"],
            "prefill_reasoning": response.get("reasoning"),
            "metadata": metadata,
        }
    except Exception as e:
        return {**base, "prefill_error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="chinese_correct_results.json")
    parser.add_argument("--output", default="chinese_rollout_results.json")
    parser.add_argument("--samples_per_chunk", type=int, required=True)
    parser.add_argument("--chunk_size", type=int, required=True)
    parser.add_argument(
        "--force_answer", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max_workers", type=int, default=20)
    args = parser.parse_args()

    results = json.loads(Path(args.input).read_text())

    inference_config = results[0]["inference_config"]
    model_name = results[0]["model"]
    model_config = next(
        m for m in inference_config["models"] if m["name"] == model_name
    )
    model_config = {**model_config, "include_reasoning": False}

    provider = get_provider(model_config["provider"])

    tasks = []
    for result in results:
        chunks = split_solution_into_chunks(
            result.get("reasoning", ""), min_chunk_length=args.chunk_size
        )
        for chunk_idx in [-1] + list(range(len(chunks))):
            for rollout_idx in range(args.samples_per_chunk):
                tasks.append((result, chunk_idx, rollout_idx, chunks))

    print(
        f"Running {len(tasks)} rollouts ({len(results)} samples, avg {len(tasks) / len(results):.1f} chunks/sample)"
    )

    tmp_path = Path(args.output).with_suffix(".jsonl")
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(
                process_rollout,
                r,
                ci,
                ri,
                chunks,
                model_config,
                provider,
                args.force_answer,
            )
            for r, ci, ri, chunks in tasks
        ]
        for future in tqdm(
            as_completed(futures), total=len(tasks), desc="Rollout Inference"
        ):
            item = future.result()
            append_jsonl(tmp_path, item)
            if "prefill_error" in item:
                print(
                    f"Error on {item.get('question_id')} chunk {item.get('chunk_index')}: {item['prefill_error']}",
                    file=sys.stderr,
                )

    all_results = list(read_jsonl(tmp_path))
    write_json(
        args.output,
        {
            "config": vars(args),
            "model_config": model_config,
            "results": all_results,
        },
    )
    tmp_path.unlink()
    print(f"Done. {len(all_results)} results written to {args.output}")


if __name__ == "__main__":
    main()
