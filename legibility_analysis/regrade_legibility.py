"""Re-grade legibility using multiple grader models to assess inter-grader consistency."""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm.auto import tqdm

from src.evaluation.grader import Grader, grade_item, compute_statistics
from src.utils.io import read_json, write_json

REASONING_MODELS = {
    "R1",
    "o3-mini",
    "qwq",
    "R1-Distill-Llama-70B",
    "R1-Distill-Qwen-32B",
    "R1-Distill-Qwen-14B",
}

RUNS = [
    # "streamlit_runs/20251012_225607_R1_gpqa",
    # "streamlit_runs/20251014_190506_R1_gpqa",
    # "streamlit_runs/20251014_201056_R1_gpqa",
    # "streamlit_runs/20251022_003910_R1-Distill-Qwen-32B_gpqa",
    # "streamlit_runs/20251022_012813_R1-Distill-Qwen-14B_gpqa",
    # "streamlit_runs/20251022_013133_R1-Distill-Qwen-14B_gpqa",
    # "streamlit_runs/20251024_155133_R1-Distill-Qwen-14B_gpqa",
    # "streamlit_runs/20251024_155559_R1-Distill-Qwen-32B_gpqa",
    "streamlit_runs/20260406_174237_qwq_gpqa"
]

MAX_WORKERS = 20


def make_eval_config(grader_model: str) -> dict:
    return {
        "grade_legibility": True,
        "grade_correctness": False,
        "grade_legibility_chunks": False,
        "max_chars_legibility": 5000,
        "grader_model": grader_model,
    }


def output_filename(grader_model: str) -> str:
    slug = grader_model.lower().replace(" ", "-")
    return f"evaluation_regrade_{slug}.json"


def regrade_run(run_dir: str, grader: Grader, eval_config: dict):
    run_path = Path(run_dir)
    inference_file = run_path / "inference.json"
    out_file = run_path / output_filename(eval_config["grader_model"])

    if not inference_file.exists():
        print(f"SKIP {run_dir}: no inference.json")
        return

    if out_file.exists():
        print(f"SKIP {run_dir}: {out_file.name} already exists")
        return

    items = read_json(inference_file)
    print(
        f"\n{'=' * 60}\n{run_dir} [{eval_config['grader_model']}]: {len(items)} items"
    )

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(grade_item, item, grader, eval_config): item
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
            "grader_model": eval_config["grader_model"],
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "purpose": "regrade for inter-grader consistency analysis",
        },
        "results": results,
        "statistics": statistics,
    }

    write_json(out_file, output)
    leg = statistics.get("legibility", {})
    print(f"  Saved {out_file}")
    if leg:
        print(
            f"  mean={leg['mean']:.2f} std={leg['std']:.2f} median={leg['median']:.1f} n={leg['count']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grader-models",
        nargs="+",
        default=["gpt-4o", "claude-sonnet-4-5", "o3-mini"],
    )
    parser.add_argument("filters", nargs="*", help="Substrings to filter RUNS")
    args = parser.parse_args()

    runs = RUNS
    if args.filters:
        runs = [r for r in RUNS if any(f in r for f in args.filters)]
        print(f"Filtered to {len(runs)} runs")

    for model_name in args.grader_models:
        print(f"\n{'#' * 60}\nGrader model: {model_name}\n{'#' * 60}")
        eval_config = make_eval_config(model_name)
        grader = Grader(model_name)
        if model_name in REASONING_MODELS:
            grader.model_config["include_reasoning"] = True
        for run_dir in runs:
            regrade_run(run_dir, grader, eval_config)


if __name__ == "__main__":
    main()
