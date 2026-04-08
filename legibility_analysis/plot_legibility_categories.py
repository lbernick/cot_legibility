"""Plot legibility category counts as a normalized heatmap and overall bar chart."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_heatmap(data: dict, output_path: Path):
    scores = sorted(data["by_score"].keys(), key=int)
    all_cats = list(data["overall"].keys())

    # Number of explanations per score (each can have multiple categories,
    # so we need the actual count from evaluation.json; approximate by using
    # the max single-category count as a lower bound — but better to normalize
    # by row sum divided by avg categories per explanation).
    # Instead, just show raw counts in heatmap and note it's count-based.

    top_cats = sorted(
        [c for c in all_cats if data["overall"][c] >= 5],
        key=lambda c: -data["overall"][c],
    )

    raw = np.zeros((len(top_cats), len(scores)))
    pct = np.zeros((len(top_cats), len(scores)))
    for j, s in enumerate(scores):
        score_counts = data["by_score"][s]
        total = sum(score_counts.values())
        for i, cat in enumerate(top_cats):
            raw[i, j] = score_counts.get(cat, 0)
            pct[i, j] = raw[i, j] / total * 100 if total else 0

    annot_labels = np.array([[f"{int(v)}" for v in row] for row in raw])

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        pct,
        xticklabels=[f"Score {s}" for s in scores],
        yticklabels=top_cats,
        annot=annot_labels,
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": "% of tags at this score"},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Legibility Category Prevalence by Score", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Heatmap saved to {output_path}")


def plot_bars(data: dict, output_path: Path):
    cats = list(data["overall"].keys())
    cats = [c for c in cats if data["overall"][c] >= 5]
    counts = [data["overall"][c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(cats))
    bars = ax.barh(y, counts, color="#3498db", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(cats)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Overall Legibility Category Counts", fontsize=14)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Bar chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("counts_path", help="Path to legibility_category_counts.json")
    parser.add_argument(
        "--output-dir", help="Output directory (default: same as input)"
    )
    args = parser.parse_args()

    counts_path = Path(args.counts_path)
    output_dir = Path(args.output_dir) if args.output_dir else counts_path.parent

    with open(counts_path) as f:
        data = json.load(f)

    plot_heatmap(data, output_dir / "legibility_categories_heatmap.png")
    plot_bars(data, output_dir / "legibility_categories_bars.png")


if __name__ == "__main__":
    main()
