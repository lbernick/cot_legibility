"""Compare legibility scores across grader models to assess inter-grader consistency."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import read_json

RUNS = ["streamlit_runs/20260406_174237_qwq_gpqa"]

DISPLAY_NAMES = {
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4-turbo": "GPT-4 Turbo",
    "claude-sonnet-4-5": "Claude Sonnet 4.5",
    "claude-sonnet-4": "Claude Sonnet 4",
    "claude-opus-4": "Claude Opus 4",
    "claude-opus-4-1": "Claude Opus 4.1",
    "claude-haiku-4-5": "Claude Haiku 4.5",
    "claude-3-7-sonnet-latest": "Claude 3.7 Sonnet",
    "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet",
    "claude-3-opus-latest": "Claude 3 Opus",
    "r1": "DeepSeek R1",
    "o3-mini": "o3-mini",
    "qwq": "QwQ-32B",
    "qwen-max": "Qwen Max",
    "qwen_max": "Qwen Max",
    "qwen3-235b": "Qwen3 235B",
}


def discover_grader_models(runs: list[str]) -> list[str]:
    """Find all grader models that have regrade files across runs."""
    models = set()
    for run_dir in runs:
        for f in Path(run_dir).glob("evaluation_regrade_*.json"):
            slug = f.stem.removeprefix("evaluation_regrade_")
            models.add(slug)
    return sorted(models)


def load_scores(run_dir: str, grader_slug: str) -> dict[tuple, float]:
    path = Path(run_dir) / f"evaluation_regrade_{grader_slug}.json"
    if not path.exists():
        return {}
    data = read_json(path)
    scores = {}
    for r in data["results"]:
        if "legibility" in r and isinstance(r["legibility"].get("score"), (int, float)):
            key = (r["question_id"], r.get("sample_index", 0))
            scores[key] = r["legibility"]["score"]
    return scores


def display_name(slug: str) -> str:
    return DISPLAY_NAMES.get(slug, slug)


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def plot_distributions(runs: list[str], models: list[str]):
    PLOTS_DIR.mkdir(exist_ok=True)
    bins = np.arange(0.5, 10.5, 1)
    centers = np.arange(1, 10)

    model_scores = {}
    for model in models:
        scores = []
        for run_dir in runs:
            scores.extend(load_scores(run_dir, model).values())
        if scores:
            model_scores[model] = np.array(scores)

    sorted_models = sorted(model_scores, key=lambda m: np.mean(model_scores[m]))
    width = 0.8 / len(sorted_models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(sorted_models):
        arr = model_scores[model]
        counts = np.histogram(arr, bins=bins)[0]
        offset = (i - len(sorted_models) / 2 + 0.5) * width
        name = display_name(model)
        ax.bar(
            centers + offset,
            counts,
            width,
            alpha=0.8,
            label=f"{name} (n={len(arr)}, \u03bc={np.mean(arr):.2f}, \u03c3={np.std(arr):.2f})",
        )

    ax.set_xlabel("Illegibility score")
    ax.set_ylabel("Count")
    ax.set_title("Score distributions by grader model")
    ax.set_xticks(centers)
    ax.legend()
    fig.tight_layout()
    out = PLOTS_DIR / "intergrader_distributions.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def main():
    runs = RUNS
    if len(sys.argv) > 1:
        runs = [r for r in RUNS if any(arg in r for arg in sys.argv[1:])]

    models = discover_grader_models(runs)
    if len(models) < 2:
        print(f"Need at least 2 grader models, found: {models}")
        return

    print(f"Grader models found: {models}")
    plot_distributions(runs, models)


if __name__ == "__main__":
    main()
