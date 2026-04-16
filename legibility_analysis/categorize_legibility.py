"""Two-pass categorization of grader legibility explanations.

Pass 1: Discover categories from a stratified sample.
Pass 2: Classify every explanation against the fixed category list.
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.providers import get_provider
from src.utils.models import get_model_config

DISCOVER_PROMPT = """\
Below are grader explanations for why a piece of reasoning text received a certain legibility score (1=perfectly legible, 10=completely illegible).

Your task: extract a taxonomy of distinct characteristics/reasons the grader cites. Each category should be a short directional phrase (2-5 words) that clearly indicates whether it's a positive or negative trait. For example, use "poor coherence" and "good coherence" instead of just "coherence issues", since "coherence issues" is ambiguous — a grader might say "no coherence issues" (positive) or "significant coherence issues" (negative). Merge near-duplicates. Aim for 10-25 categories.

Return JSON: {{"categories": ["category1", "category2", ...]}}

EXPLANATIONS (score in brackets):
{explanations}"""

DEDUP_PROMPT = """\
Below is a list of categories extracted from grader explanations. Some are near-duplicates or overlapping (e.g. "repetitive phrasing" and "repetition", "syntactical errors" and "syntactical issues").

Merge near-duplicates into a single canonical name. Keep the most descriptive version. Do not add new categories.

Return JSON: {{"categories": ["category1", "category2", ...]}}

CATEGORIES:
{categories}"""

CLASSIFY_PROMPT = """\
Below are grader explanations for legibility scores. For each explanation, output which categories from the list apply.

CATEGORIES:
{categories}

Return JSON: {{"results": [{{"index": 0, "categories": ["cat1", "cat2"]}}, ...]}}
Use exact category names from the list. Each explanation gets at least one category.
IMPORTANT: Only tag a category if the grader identifies it as present. If the grader says something is absent (e.g. "no coherence issues", "without any repetition"), do NOT tag the corresponding negative category.

EXPLANATIONS:
{explanations}"""


def load_explanations(eval_path: str) -> list[dict]:
    with open(eval_path) as f:
        data = json.load(f)
    out = []
    for r in data.get("results", []):
        leg = r.get("legibility", {})
        if leg.get("score") is not None and leg.get("explanation"):
            out.append(
                {
                    "question_id": r.get("question_id"),
                    "sample_index": r.get("sample_index", 0),
                    "score": leg["score"],
                    "explanation": leg["explanation"],
                    "file": eval_path,
                }
            )
    return out


def call_llm(provider, model_config, prompt: str) -> dict:
    config = {**model_config, "temperature": 0.0}
    result = provider.generate(prompt, config)
    text = result["answer"]
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1]
    return json.loads(text.strip())


def discover_categories(
    explanations: list[dict], provider, model_config, sample_size: int = 80
) -> list[str]:
    by_score = {}
    for e in explanations:
        by_score.setdefault(e["score"], []).append(e)

    sample = []
    per_score = max(3, sample_size // len(by_score))
    for score in sorted(by_score):
        items = by_score[score]
        sample.extend(random.sample(items, min(per_score, len(items))))

    random.shuffle(sample)
    sample = sample[:sample_size]

    formatted = "\n".join(f"[{e['score']}] {e['explanation']}" for e in sample)
    prompt = DISCOVER_PROMPT.format(explanations=formatted)
    result = call_llm(provider, model_config, prompt)
    raw_categories = result["categories"]

    cat_str = "\n".join(f"- {c}" for c in raw_categories)
    dedup_prompt = DEDUP_PROMPT.format(categories=cat_str)
    deduped = call_llm(provider, model_config, dedup_prompt)
    return deduped["categories"]


def classify_batch(
    batch: list[dict], categories: list[str], provider, model_config
) -> list[dict]:
    cat_str = "\n".join(f"- {c}" for c in categories)
    formatted = "\n".join(
        f"[{i}] (score={e['score']}) {e['explanation']}" for i, e in enumerate(batch)
    )
    prompt = CLASSIFY_PROMPT.format(categories=cat_str, explanations=formatted)
    result = call_llm(provider, model_config, prompt)
    valid = set(categories)
    for r in result["results"]:
        r["categories"] = [c for c in r["categories"] if c in valid]
    return result["results"]


def classify_all(
    explanations: list[dict],
    categories: list[str],
    provider,
    model_config,
    batch_size: int = 30,
    max_workers: int = 8,
) -> list[dict]:
    batches = [
        explanations[i : i + batch_size]
        for i in range(0, len(explanations), batch_size)
    ]

    all_results = [None] * len(explanations)
    futures = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_idx, batch in enumerate(batches):
            future = executor.submit(
                classify_batch, batch, categories, provider, model_config
            )
            futures[future] = batch_idx

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Classifying"
        ):
            batch_idx = futures[future]
            offset = batch_idx * batch_size
            try:
                results = future.result()
                for r in results:
                    all_results[offset + r["index"]] = r["categories"]
            except Exception as e:
                print(f"Batch {batch_idx} failed: {e}")
                for i in range(len(batches[batch_idx])):
                    all_results[offset + i] = []

    return all_results


def aggregate(explanations: list[dict], classifications: list[list[str]]) -> dict:
    by_score = {}
    overall = {}
    for exp, cats in zip(explanations, classifications):
        score = exp["score"]
        if score not in by_score:
            by_score[score] = {}
        for c in cats or []:
            overall[c] = overall.get(c, 0) + 1
            by_score[score][c] = by_score[score].get(c, 0) + 1

    return {
        "overall": dict(sorted(overall.items(), key=lambda x: -x[1])),
        "by_score": {
            s: dict(sorted(counts.items(), key=lambda x: -x[1]))
            for s, counts in sorted(by_score.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Categorize legibility explanations")
    parser.add_argument(
        "eval_paths", nargs="+", help="Path(s) to evaluation.json files"
    )
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: same as eval_path)",
    )
    args = parser.parse_args()

    eval_paths = [Path(p) for p in args.eval_paths]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif len(eval_paths) == 1:
        output_dir = eval_paths[0].parent
    else:
        parser.error("--output-dir is required when passing multiple eval files")

    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_model_config(args.model)
    provider = get_provider(config["provider"])

    explanations = []
    for p in eval_paths:
        loaded = load_explanations(str(p))
        print(f"Loaded {len(loaded)} explanations from {p}")
        explanations.extend(loaded)
    print(f"Total: {len(explanations)} explanations")

    random.seed(42)
    print("Pass 1: Discovering categories...")
    categories = discover_categories(explanations, provider, config)
    print(f"Found {len(categories)} categories:")
    for c in categories:
        print(f"  - {c}")

    categories_file = output_dir / "legibility_categories.json"
    with open(categories_file, "w") as f:
        json.dump({"categories": categories}, f, indent=2)
    print(f"Categories saved to {categories_file}")

    print(
        f"Pass 2: Classifying {len(explanations)} explanations (batch_size={args.batch_size})..."
    )
    classifications = classify_all(
        explanations,
        categories,
        provider,
        config,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
    )

    per_item = [
        {
            "question_id": exp["question_id"],
            "sample_index": exp["sample_index"],
            "file": exp["file"],
            "score": exp["score"],
            "explanation": exp["explanation"],
            "categories": cats or [],
        }
        for exp, cats in zip(explanations, classifications)
    ]
    per_item_file = output_dir / "legibility_per_item.json"
    with open(per_item_file, "w") as f:
        json.dump(per_item, f, indent=2)
    print(f"Per-item classifications saved to {per_item_file}")

    counts = aggregate(explanations, classifications)
    results_file = output_dir / "legibility_category_counts.json"
    with open(results_file, "w") as f:
        json.dump(counts, f, indent=2)
    print(f"Counts saved to {results_file}")

    print("\n=== Overall counts ===")
    for cat, count in counts["overall"].items():
        print(f"  {cat}: {count}")

    print("\n=== By score ===")
    for score, cats in counts["by_score"].items():
        print(f"  Score {score}:")
        for cat, count in cats.items():
            print(f"    {cat}: {count}")


if __name__ == "__main__":
    main()
