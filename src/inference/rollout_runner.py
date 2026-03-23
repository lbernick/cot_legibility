from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from .providers import get_provider
from ..utils.io import append_jsonl, read_json, read_jsonl, write_json


def _skip_result(result: dict, chunk_index: int, rollout_index: int, reason: str, message: str) -> dict:
    return {
        "question_id": result["question_id"],
        "question": result["question"],
        "original_answer": result.get("answer"),
        "original_correctness": result.get("correctness"),
        "chunk_index": chunk_index,
        "rollout_index": rollout_index,
        "prefill_skip_reason": reason,
        "prefill_skip_message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def process_rollout(result: dict, chunk_index: int, rollout_index: int, model_config: dict, provider, force_answer: bool) -> dict:
    try:
        reasoning = result.get("reasoning", "")
        legibility_chunks = result.get("legibility_chunks", [])

        if not reasoning:
            return _skip_result(result, chunk_index, rollout_index, "no_reasoning", "No reasoning available")
        if not legibility_chunks or chunk_index >= len(legibility_chunks):
            return _skip_result(result, chunk_index, rollout_index, "no_chunks", "No legibility chunks available")

        chunk = legibility_chunks[chunk_index]
        extracted = reasoning[:chunk.get("end_pos", 0)]
        last_newline = extracted.rfind('\n')
        if last_newline != -1:
            extracted = extracted[:last_newline]

        if not extracted.strip():
            return _skip_result(result, chunk_index, rollout_index, "empty_after_extraction", "Extraction resulted in empty reasoning")

        prefill = f"<think>\n{extracted}\n</think>" if force_answer else f"<think>\n{extracted}"

        response = provider.generate(result["question"], model_config, prefill=prefill)

        metadata = {"duration_ms": response["duration_ms"], "tokens": response.get("tokens")}
        for field in ["provider_model", "openrouter_provider", "stream_complete", "error"]:
            if field in response:
                metadata[field] = response[field]

        output = {
            "question_id": result["question_id"],
            "question": result["question"],
            "original_answer": result.get("answer"),
            "original_correctness": result.get("correctness"),
            "legibility_chunks": legibility_chunks,
            "chunk_index": chunk_index,
            "rollout_index": rollout_index,
            "prefill": prefill,
            "prefill_answer": response["answer"],
            "prefill_reasoning": response.get("reasoning"),
            "prefill_reasoning_length": len(prefill),
            "prefill_include_reasoning": model_config.get("include_reasoning", False),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata,
        }
        for field in ["correct_answer", "dataset", "model", "temperature"]:
            if field in result:
                output[field] = result[field]
        return output
    except Exception as e:
        return {
            "question_id": result["question_id"],
            "question": result["question"],
            "original_answer": result.get("answer"),
            "original_correctness": result.get("correctness"),
            "chunk_index": chunk_index,
            "rollout_index": rollout_index,
            "prefill_error": str(e),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


def run_rollout_stage(config: dict, output_dir: Path, logger) -> None:
    rollout_config = config["rollouts"]
    samples_per_chunk = rollout_config["samples_per_chunk"]
    force_answer = rollout_config.get("force_answer", True)
    include_reasoning = rollout_config.get("include_reasoning", False)
    max_workers = rollout_config.get("max_workers", config.get("prefill", {}).get("max_workers", 10))

    evaluation_file = output_dir / "evaluation.json"
    if not evaluation_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {evaluation_file}")

    logger.info(f"Loading evaluation results from {evaluation_file}")
    evaluation = read_json(evaluation_file)
    eval_results = evaluation.get("results", [])

    inference_file = evaluation.get("metadata", {}).get("inference_file")
    inference_file = Path(inference_file) if inference_file else output_dir / "inference.json"
    if not inference_file.exists():
        raise FileNotFoundError(f"Inference file not found: {inference_file}")

    logger.info(f"Loading full reasoning from {inference_file}")
    inference_results = read_json(inference_file)
    reasoning_map = {
        (item["question_id"], item.get("sample_index", 0)): item.get("reasoning", "")
        for item in inference_results
    }

    legibility_threshold = rollout_config.get("legibility_threshold")
    correct_only = rollout_config.get("correct_only", False)

    results = []
    skipped = 0
    for eval_result in eval_results:
        if legibility_threshold is not None:
            score = (eval_result.get("legibility") or {}).get("score")
            if score is None or score < legibility_threshold:
                skipped += 1
                continue
        if correct_only:
            correctness = (eval_result.get("correctness") or {}).get("correctness")
            if correctness != "correct":
                skipped += 1
                continue
        full_reasoning = reasoning_map.get((eval_result["question_id"], eval_result.get("sample_index", 0)), "")
        results.append({**eval_result, "reasoning": full_reasoning})

    if skipped:
        logger.info(f"Filtered out {skipped} results (legibility_threshold={legibility_threshold}, correct_only={correct_only})")

    if not results:
        logger.warning("No results remain after filtering")
        return

    model_name = results[0].get("model")
    model_config = None
    for m in config.get("inference", {}).get("models", []):
        if m["name"] == model_name:
            model_config = m
            break
    if not model_config:
        raise ValueError(f"Model config not found for model: {model_name}")

    model_config = {**model_config, "include_reasoning": include_reasoning}

    logger.info(f"Samples per chunk: {samples_per_chunk}, force_answer: {force_answer}")
    logger.info(f"Initializing provider: {model_config['provider']}")
    provider = get_provider(model_config["provider"])

    tasks = [
        (result, chunk_idx, rollout_idx)
        for result in results
        for chunk_idx in range(len(result.get("legibility_chunks", [])))
        for rollout_idx in range(samples_per_chunk)
    ]

    logger.info(f"Running {len(tasks)} rollout inferences with {max_workers} workers")
    rollout_jsonl = output_dir / "rollout_inference.jsonl"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_rollout, r, ci, ri, model_config, provider, force_answer)
            for r, ci, ri in tasks
        ]
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Rollout Inference"):
            item = future.result()
            append_jsonl(rollout_jsonl, item)
            if "prefill_error" in item:
                logger.warning(f"Error on {item.get('question_id')} chunk {item.get('chunk_index')}: {item['prefill_error']}")

    logger.info("Converting results to JSON format")
    all_results = list(read_jsonl(rollout_jsonl))
    json_file = output_dir / "rollout_inference.json"
    write_json(json_file, {"results": all_results})
    rollout_jsonl.unlink()
    logger.info(f"Rollout inference complete. {len(all_results)} results written to {json_file}")
