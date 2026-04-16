"""Compare original vs regraded legibility scores to detect grader model drift."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.utils.io import read_json

RUNS = [
    "streamlit_runs/20251012_225607_R1_gpqa",
    "streamlit_runs/20251014_190506_R1_gpqa",
    "streamlit_runs/20251014_201056_R1_gpqa",
    "streamlit_runs/20251022_003910_R1-Distill-Qwen-32B_gpqa",
    "streamlit_runs/20251022_012813_R1-Distill-Qwen-14B_gpqa",
    "streamlit_runs/20251022_013133_R1-Distill-Qwen-14B_gpqa",
    "streamlit_runs/20251024_155133_R1-Distill-Qwen-14B_gpqa",
    "streamlit_runs/20251024_155559_R1-Distill-Qwen-32B_gpqa",
]


def compare_run(run_dir: str) -> dict | None:
    run_path = Path(run_dir)
    old_file = run_path / "evaluation.json"
    new_file = run_path / "evaluation_regrade_gpt-4o.json"

    if not new_file.exists():
        print(f"SKIP {run_dir}: no evaluation_regrade_gpt-4o.json")
        return None

    old_data = read_json(old_file)
    new_data = read_json(new_file)

    old_scores = {}
    for r in old_data["results"]:
        if "legibility" in r and isinstance(r["legibility"].get("score"), (int, float)):
            key = (r["question_id"], r.get("sample_index", 0))
            old_scores[key] = r["legibility"]["score"]

    new_scores = {}
    for r in new_data["results"]:
        if "legibility" in r and isinstance(r["legibility"].get("score"), (int, float)):
            key = (r["question_id"], r.get("sample_index", 0))
            new_scores[key] = r["legibility"]["score"]

    common_keys = sorted(set(old_scores) & set(new_scores))
    if not common_keys:
        print(f"SKIP {run_dir}: no overlapping legibility scores")
        return None

    old_arr = np.array([old_scores[k] for k in common_keys])
    new_arr = np.array([new_scores[k] for k in common_keys])
    diffs = new_arr - old_arr

    corr = np.corrcoef(old_arr, new_arr)[0, 1]
    exact_match = np.sum(diffs == 0)
    within_1 = np.sum(np.abs(diffs) <= 1)
    big_change = np.sum(np.abs(diffs) >= 3)

    run_name = run_dir.split("/")[-1]
    stats = {
        "run": run_name,
        "n": len(common_keys),
        "old_mean": float(np.mean(old_arr)),
        "new_mean": float(np.mean(new_arr)),
        "mean_diff": float(np.mean(diffs)),
        "mean_abs_diff": float(np.mean(np.abs(diffs))),
        "correlation": float(corr),
        "exact_match": int(exact_match),
        "within_1": int(within_1),
        "big_change_3plus": int(big_change),
    }
    return stats


def main():
    runs = RUNS
    if len(sys.argv) > 1:
        runs = [r for r in RUNS if any(arg in r for arg in sys.argv[1:])]

    all_stats = []
    for run_dir in runs:
        stats = compare_run(run_dir)
        if stats:
            all_stats.append(stats)

    if not all_stats:
        print("No runs to compare.")
        return

    print(f"\n{'=' * 100}")
    print(
        f"{'Run':<45} {'n':>4} {'old':>6} {'new':>6} {'diff':>6} {'|diff|':>6} {'corr':>6} {'exact':>6} {'<=1':>5} {'>=3':>5}"
    )
    print(f"{'-' * 100}")

    for s in all_stats:
        print(
            f"{s['run']:<45} {s['n']:>4} {s['old_mean']:>6.2f} {s['new_mean']:>6.2f} "
            f"{s['mean_diff']:>+6.2f} {s['mean_abs_diff']:>6.2f} {s['correlation']:>6.3f} "
            f"{s['exact_match']:>5}/{s['n']:<1} {s['within_1']:>4}/{s['n']:<1} {s['big_change_3plus']:>4}/{s['n']:<1}"
        )

    # Aggregate per model
    model_scores: dict[str, tuple[list, list]] = {}
    for run_dir in runs:
        run_path = Path(run_dir)
        new_file = run_path / "evaluation_regrade_gpt-4o.json"
        if not new_file.exists():
            continue
        old_data = read_json(run_path / "evaluation.json")
        new_data = read_json(new_file)
        model = get_model_name(run_dir)

        old_map = {}
        for r in old_data["results"]:
            if "legibility" in r and isinstance(
                r["legibility"].get("score"), (int, float)
            ):
                old_map[(r["question_id"], r.get("sample_index", 0))] = r["legibility"][
                    "score"
                ]

        if model not in model_scores:
            model_scores[model] = ([], [])
        for r in new_data["results"]:
            if "legibility" in r and isinstance(
                r["legibility"].get("score"), (int, float)
            ):
                key = (r["question_id"], r.get("sample_index", 0))
                if key in old_map:
                    model_scores[model][0].append(old_map[key])
                    model_scores[model][1].append(r["legibility"]["score"])

    all_old = np.concatenate([np.array(v[0]) for v in model_scores.values()])
    all_new = np.concatenate([np.array(v[1]) for v in model_scores.values()])
    all_diffs = all_new - all_old
    corr = np.corrcoef(all_old, all_new)[0, 1]

    print(f"{'-' * 100}")
    print(
        f"{'TOTAL':<45} {len(all_old):>4} {np.mean(all_old):>6.2f} {np.mean(all_new):>6.2f} "
        f"{np.mean(all_diffs):>+6.2f} {np.mean(np.abs(all_diffs)):>6.2f} {corr:>6.3f} "
        f"{np.sum(all_diffs == 0):>5}/{len(all_old):<1} {np.sum(np.abs(all_diffs) <= 1):>4}/{len(all_old):<1} {np.sum(np.abs(all_diffs) >= 3):>4}/{len(all_old):<1}"
    )

    plot_scatter_per_model(model_scores)
    plot_histogram_per_model(model_scores)


MODEL_ORDER = ["R1-Distill-Qwen-32B", "R1-Distill-Qwen-14B", "R1-Zero", "R1"]


def get_model_name(run_dir: str) -> str:
    name = run_dir.split("/")[-1]
    for model in MODEL_ORDER:
        if model in name:
            return model
    return name


def plot_scatter_per_model(model_scores: dict[str, tuple[list, list]]):
    models = [m for m in MODEL_ORDER if m in model_scores]
    rng = np.random.default_rng(42)
    jitter = 0.15
    Path("plots").mkdir(exist_ok=True)

    for model in models:
        old = np.array(model_scores[model][0])
        new = np.array(model_scores[model][1])
        corr = np.corrcoef(old, new)[0, 1] if len(old) > 1 else 0.0

        old_j = old + rng.uniform(-jitter, jitter, len(old))
        new_j = new + rng.uniform(-jitter, jitter, len(new))

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(old_j, new_j, alpha=0.3, s=12, edgecolors="none")
        ax.plot([0, 10], [0, 10], "k--", alpha=0.3, lw=1)
        ax.set_xlabel("Original illegibility score")
        ax.set_ylabel("Regraded illegibility score")
        ax.set_title(
            f"Illegibility score consistency ({model})\nr={corr:.3f}, n={len(old)}"
        )
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0.5, 9.5)
        ax.set_xticks(range(1, 10))
        ax.set_yticks(range(1, 10))
        ax.set_aspect("equal")
        fig.tight_layout()

        slug = model.lower().replace(" ", "_")
        out = f"plots/regrade_scatter_{slug}.png"
        fig.savefig(out, dpi=150)
        print(f"Scatter plot saved to {out}")
        plt.close(fig)


def plot_histogram_per_model(model_scores: dict[str, tuple[list, list]]):
    models = [m for m in MODEL_ORDER if m in model_scores]
    Path("plots").mkdir(exist_ok=True)
    bins = np.arange(0.5, 10.5, 1)

    for model in models:
        old = np.array(model_scores[model][0])
        new = np.array(model_scores[model][1])

        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.35
        centers = np.arange(1, 10)
        old_counts = np.histogram(old, bins=bins)[0]
        new_counts = np.histogram(new, bins=bins)[0]

        ax.bar(centers - width / 2, old_counts, width, label="Original", alpha=0.8)
        ax.bar(centers + width / 2, new_counts, width, label="Regraded", alpha=0.8)
        ax.set_xlabel("Illegibility score")
        ax.set_ylabel("Count")
        ax.set_title(f"Illegibility score distribution ({model})\nn={len(old)}")
        ax.set_xticks(centers)
        ax.legend()
        fig.tight_layout()

        slug = model.lower().replace(" ", "_")
        out = f"plots/regrade_hist_{slug}.png"
        fig.savefig(out, dpi=150)
        print(f"Histogram saved to {out}")
        plt.close(fig)


if __name__ == "__main__":
    main()
