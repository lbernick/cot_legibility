#!/usr/bin/env python3
import json
import re
from pathlib import Path

import yaml

RUNS_DIR = Path("streamlit_runs")
MODEL_FILTER = "qwq"
OUTPUT_FILE = Path("chinese_correct_results.json")

CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def chinese_character_count(text: str) -> int:
    if not text:
        return 0
    return len(CHINESE_RE.findall(text))


def load_run(run_dir: Path) -> list[dict]:
    inference_path = run_dir / "inference.json"
    evaluation_path = run_dir / "evaluation.json"
    config_path = run_dir / "config.yaml"

    if not all(p.exists() for p in [inference_path, evaluation_path, config_path]):
        return []

    with open(inference_path) as f:
        inference_data = json.load(f)

    with open(evaluation_path) as f:
        eval_data = json.load(f)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    eval_by_key = {
        (r["question_id"], r["sample_index"]): r
        for r in eval_data["results"]
    }

    results = []
    for item in inference_data:
        key = (item["question_id"], item["sample_index"])
        eval_item = eval_by_key.get(key)
        if not eval_item:
            continue

        correctness = eval_item.get("correctness", {})
        if correctness.get("correctness") != "correct":
            continue

        char_cnt = chinese_character_count(item.get("reasoning", ""))
        if char_cnt <= 5:
            continue

        results.append({
            **item,
            "correctness": correctness,
            "run": run_dir.name,
            "inference_config": config.get("inference"),
            "evaluation_config": config.get("evaluation"),
        })

    return results


def main():
    all_results = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        parts = run_dir.name.split("_")
        if len(parts) < 3:
            continue
        model = parts[2]
        if model != MODEL_FILTER:
            continue

        results = load_run(run_dir)
        all_results.extend(results)
        if results:
            print(f"{run_dir.name}: {len(results)} matches")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nTotal: {len(all_results)} results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
