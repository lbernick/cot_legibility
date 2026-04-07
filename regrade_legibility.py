"""Re-grade legibility for R1 gpqa runs using the same code path as the original evaluation."""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from src.evaluation.grader import Grader, grade_item, compute_statistics
from src.utils.io import read_json, write_json

RUNS = [
    "streamlit_runs/20251012_225607_R1_gpqa",
    "streamlit_runs/20251014_190506_R1_gpqa",
    "streamlit_runs/20251014_201056_R1_gpqa",
    "streamlit_runs/20251022_003910_R1-Distill-Qwen-32B_gpqa",
    "streamlit_runs/20251022_012813_R1-Distill-Qwen-14B_gpqa",
    "streamlit_runs/20251022_013133_R1-Distill-Qwen-14B_gpqa",
    #"streamlit_runs/20251024_054931_R1-Zero_gpqa",
    "streamlit_runs/20251024_155133_R1-Distill-Qwen-14B_gpqa",
    "streamlit_runs/20251024_155559_R1-Distill-Qwen-32B_gpqa",
    #"streamlit_runs/20251024_165858_R1-Zero_gpqa",
]

EVAL_CONFIG = {
    "grade_legibility": True,
    "grade_correctness": False,
    "grade_legibility_chunks": False,
    "max_chars_legibility": 5000,
    "grader_model": "gpt-4o",
}

MAX_WORKERS = 20


def regrade_run(run_dir: str, grader: Grader):
    run_path = Path(run_dir)
    inference_file = run_path / "inference.json"
    output_file = run_path / "evaluation_regrade.json"

    if not inference_file.exists():
        print(f"SKIP {run_dir}: no inference.json")
        return

    if output_file.exists():
        print(f"SKIP {run_dir}: evaluation_regrade.json already exists")
        return

    items = read_json(inference_file)
    print(f"\n{'=' * 60}\n{run_dir}: {len(items)} items")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(grade_item, item, grader, EVAL_CONFIG): item
            for item in items
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=run_dir.split("/")[-1]
        ):
            try:
                results.append(future.result())
            except Exception as e:
                item = futures[future]
                print(f"  ERROR {item['question_id']}: {e}")
                results.append(
                    {
                        "question_id": item["question_id"],
                        "sample_index": item.get("sample_index", 0),
                        "errors": [f"Fatal: {e}"],
                    }
                )

    statistics = compute_statistics(results)
    output = {
        "metadata": {
            "inference_file": str(inference_file),
            "grader_model": EVAL_CONFIG["grader_model"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "purpose": "regrade to check for grader model drift",
        },
        "results": results,
        "statistics": statistics,
    }

    write_json(output_file, output)
    leg = statistics.get("legibility", {})
    print(f"  Saved {output_file}")
    if leg:
        print(
            f"  mean={leg['mean']:.2f} std={leg['std']:.2f} median={leg['median']:.1f} n={leg['count']}"
        )


def main():
    runs = RUNS
    if len(sys.argv) > 1:
        runs = [r for r in RUNS if any(arg in r for arg in sys.argv[1:])]
        print(f"Filtered to {len(runs)} runs")

    grader = Grader("gpt-4o")
    for run_dir in runs:
        regrade_run(run_dir, grader)


if __name__ == "__main__":
    main()
