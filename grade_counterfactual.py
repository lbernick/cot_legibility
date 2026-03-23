#!/usr/bin/env python3
"""Grade prefill_answer correctness for counterfactual rollout results."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluation.grader import Grader
from src.utils.io import read_json, write_json

GRADER_MODEL = "gpt-4o"
MAX_WORKERS = 10


def grade_item(item: dict, grader: Grader) -> dict:
    result = {k: item.get(k) for k in (
        "question_id", "sample_index", "dataset", "question",
        "correct_answer", "original_answer", "original_correctness",
        "prefill_answer", "chunk_index", "rollout_index", "timestamp", "metadata", "chunk"
    )}

    predicted = item.get("prefill_answer", "")
    actual = item.get("correct_answer", "")
    if predicted and actual:
        result["prefill_correctness"] = grader.grade_correctness(predicted, actual, item["question"])

    return result


def compute_stats(results: list[dict]) -> dict:
    prefill = [r["prefill_correctness"]["correctness"] for r in results if "prefill_correctness" in r]
    original = [r["original_correctness"] for r in results if isinstance(r.get("original_correctness"), str)]

    def pct_stats(grades):
        total = len(grades)
        correct = grades.count("correct")
        partial = grades.count("partially_correct")
        incorrect = grades.count("incorrect")
        return {
            "correct": correct, "partially_correct": partial, "incorrect": incorrect, "total": total,
            "correct_pct": round(correct / total * 100, 1) if total else 0,
            "partially_pct": round(partial / total * 100, 1) if total else 0,
            "incorrect_pct": round(incorrect / total * 100, 1) if total else 0,
        }

    stats = {}
    if prefill:
        stats["prefill_correctness"] = pct_stats(prefill)
    if original:
        stats["original_correctness"] = pct_stats(original)
    if prefill and original:
        diff = stats["prefill_correctness"]["correct_pct"] - stats["original_correctness"]["correct_pct"]
        stats["comparison"] = {
            "prefill_correct_pct": stats["prefill_correctness"]["correct_pct"],
            "original_correct_pct": stats["original_correctness"]["correct_pct"],
            "difference_pct": round(diff, 1),
        }
    return stats


def main(input_file: Path, output_file: Path):
    data = read_json(input_file)
    items = data["results"]
    print(f"Loaded {len(items)} items")

    grader = Grader(GRADER_MODEL)
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(grade_item, item, grader): item for item in items}
        for future in tqdm(as_completed(futures), total=len(items), desc="Grading"):
            try:
                results.append(future.result())
            except Exception as e:
                item = futures[future]
                print(f"Error grading {item['question_id']}: {e}")

    stats = compute_stats(results)
    write_json(output_file, {"metadata": {"grader_model": GRADER_MODEL}, "results": results, "statistics": stats})

    print(f"\nResults written to {output_file}")
    if "prefill_correctness" in stats:
        s = stats["prefill_correctness"]
        print(f"Prefill: {s['correct_pct']}% correct, {s['partially_pct']}% partial (n={s['total']})")
    if "original_correctness" in stats:
        s = stats["original_correctness"]
        print(f"Original: {s['correct_pct']}% correct, {s['partially_pct']}% partial (n={s['total']})")
    if "comparison" in stats:
        diff = stats["comparison"]["difference_pct"]
        sign = "+" if diff > 0 else ""
        print(f"Difference: {sign}{diff}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grade counterfactual rollout results")
    parser.add_argument("--input", help="Input JSON file")
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = input_path.parent / (input_path.stem + "_graded" + input_path.suffix)
    main(input_path, output_path)
