"""Compare illegibility score distributions between original and new inference runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import read_json

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

MODELS = {
    "R1": {
        "original": [
            "streamlit_runs/20251012_225607_R1_gpqa",
            "streamlit_runs/20251014_190506_R1_gpqa",
            "streamlit_runs/20251014_201056_R1_gpqa",
        ],
        "new": ["streamlit_runs/20260405_085558_r1_gpqa"],
    },
    "QwQ": {
        "original": [
            f"streamlit_runs/{d}"
            for d in [
                "20251016_011742_qwq_gpqa",
                "20251017_172954_qwq_gpqa",
                "20251017_192243_qwq_gpqa",
                "20251017_224816_qwq_gpqa",
                "20251017_230836_qwq_gpqa",
                "20251018_034705_qwq_gpqa",
                "20251018_145227_qwq_gpqa",
                "20251018_175940_qwq_gpqa",
                "20251019_184002_qwq_gpqa",
                "20251019_185849_qwq_gpqa",
                "20251019_191941_qwq_gpqa",
                "20251019_195028_qwq_gpqa",
                "20251019_202534_qwq_gpqa",
                "20251020_185640_qwq_gpqa",
                "20251020_221724_qwq_gpqa",
                "20251020_232429_qwq_gpqa",
            ]
        ],
        "new": ["streamlit_runs/20260406_174237_qwq_gpqa"],
    },
}


def collect_scores(run_dirs: list[str]) -> np.ndarray:
    scores = []
    for run_dir in run_dirs:
        eval_file = Path(run_dir) / "evaluation.json"
        if not eval_file.exists():
            print(f"SKIP {run_dir}: no evaluation.json")
            continue
        data = read_json(eval_file)
        for r in data["results"]:
            if "legibility" in r and isinstance(
                r["legibility"].get("score"), (int, float)
            ):
                scores.append(r["legibility"]["score"])
    return np.array(scores)


def plot_model(model_name: str, original_dirs: list[str], new_dirs: list[str]):
    old_scores = collect_scores(original_dirs)
    new_scores = collect_scores(new_dirs)

    bins = np.arange(0.5, 10.5, 1)
    centers = np.arange(1, 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    old_counts = np.histogram(old_scores, bins=bins)[0]
    ax1.bar(centers, old_counts, width=0.7, alpha=0.8)
    ax1.set_xlabel("Illegibility score")
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"Original ({model_name})\n"
        f"n={len(old_scores)}, mean={np.mean(old_scores):.2f}, std={np.std(old_scores):.2f}"
    )
    ax1.set_xticks(centers)

    new_counts = np.histogram(new_scores, bins=bins)[0]
    ax2.bar(centers, new_counts, width=0.7, alpha=0.8)
    ax2.set_xlabel("Illegibility score")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"New ({model_name})\n"
        f"n={len(new_scores)}, mean={np.mean(new_scores):.2f}, std={np.std(new_scores):.2f}"
    )
    ax2.set_xticks(centers)

    fig.tight_layout()
    PLOTS_DIR.mkdir(exist_ok=True)
    out = PLOTS_DIR / f"generation_compare_{model_name.lower()}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def main():
    for model_name, dirs in MODELS.items():
        plot_model(model_name, dirs["original"], dirs["new"])


if __name__ == "__main__":
    main()
