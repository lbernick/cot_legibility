"""Analyze how grader citations of repetition relate to repetition metrics in reasoning."""

import json
import re
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STREAMLIT_RUNS = Path(__file__).resolve().parent.parent / "streamlit_runs"
OUTPUT_DIR = Path(__file__).resolve().parent / "qwq_gpqa_combined"

# Grader explanation: mentions of repetition as a legibility factor
# Strict: match "repetition", "repetitive phrasing/phrases", not "repeated corrections" etc.
GRADER_REPETITION_RE = re.compile(
    r"repetition|repetitive\s+phras",
    re.IGNORECASE,
)


def ngram_repetition_ratio(text: str, n: int) -> float:
    """Fraction of word n-grams that duplicate an earlier n-gram."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    seen: set[tuple[str, ...]] = set()
    dupes = 0
    for g in ngrams:
        if g in seen:
            dupes += 1
        else:
            seen.add(g)
    return dupes / len(ngrams)


def compression_ratio(text: str) -> float:
    """Ratio of compressed size to original size. Lower = more repetitive."""
    raw = text.encode("utf-8")
    if len(raw) == 0:
        return 1.0
    return len(zlib.compress(raw)) / len(raw)


def load_data() -> list[dict]:
    rows = []
    for eval_path in sorted(STREAMLIT_RUNS.glob("*qwq*/evaluation.json")):
        inf_path = eval_path.parent / "inference.json"
        if not inf_path.exists():
            continue

        with open(eval_path) as f:
            ev = json.load(f)
        with open(inf_path) as f:
            inf = json.load(f)

        inf_lookup = {}
        for item in inf:
            key = (item["question_id"], item.get("sample_index", 0))
            inf_lookup[key] = item

        for r in ev["results"]:
            leg = r.get("legibility", {})
            explanation = leg.get("explanation", "")
            score = leg.get("score")
            if not explanation or score is None:
                continue

            key = (r["question_id"], r.get("sample_index", 0))
            reasoning = (inf_lookup.get(key) or {}).get("reasoning") or ""

            rows.append(
                {
                    "score": score,
                    "grader_mentioned": bool(GRADER_REPETITION_RE.search(explanation)),
                    "rep_3": ngram_repetition_ratio(reasoning, 3),
                    "rep_4": ngram_repetition_ratio(reasoning, 4),
                    "rep_5": ngram_repetition_ratio(reasoning, 5),
                    "compression_ratio": compression_ratio(reasoning),
                }
            )
    return rows


def _scatter(ax, scores: np.ndarray, values: np.ndarray, title: str, ylabel: str):
    jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(scores))
    ax.scatter(
        scores + jitter, values, alpha=0.15, s=8, color="#4c72b0", edgecolors="none"
    )
    unique_scores = sorted(set(scores))
    means = [values[scores == s].mean() for s in unique_scores]
    ax.plot(unique_scores, means, "o-", color="#c44e52", ms=6, lw=2, label="mean")
    ax.set_xlabel("Legibility score")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


def plot_ngram_vs_score(rows: list[dict], output_dir: Path):
    """Scatterplot: n-gram repetition ratio vs legibility score for n=3,4,5."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    scores = np.array([r["score"] for r in rows])
    for ax, n, key in zip(axes, [3, 4, 5], ["rep_3", "rep_4", "rep_5"]):
        values = np.array([r[key] for r in rows])
        _scatter(ax, scores, values, f"{n}-gram repetition ratio", "")
    axes[0].set_ylabel("Repetition ratio (fraction duplicate n-grams)")
    fig.suptitle("N-gram repetition ratio vs. legibility score", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "ngram_repetition_vs_score.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'ngram_repetition_vs_score.png'}")


def plot_compression_vs_score(rows: list[dict], output_dir: Path):
    """Scatterplot: compression ratio vs legibility score."""
    fig, ax = plt.subplots(figsize=(8, 5))
    scores = np.array([r["score"] for r in rows])
    values = np.array([r["compression_ratio"] for r in rows])
    _scatter(
        ax,
        scores,
        values,
        "Compression ratio vs. legibility score",
        "Compression ratio (lower = more repetitive)",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "compression_ratio_vs_score.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'compression_ratio_vs_score.png'}")


def plot_grader_mention_vs_score(rows: list[dict], output_dir: Path):
    """Two-panel: fraction citing repetition per score + sample count."""
    by_score: dict[int, dict] = {}
    for r in rows:
        s = r["score"]
        if s not in by_score:
            by_score[s] = {"total": 0, "mentioned": 0}
        by_score[s]["total"] += 1
        if r["grader_mentioned"]:
            by_score[s]["mentioned"] += 1

    scores = sorted(by_score)
    totals = [by_score[s]["total"] for s in scores]
    fractions = [by_score[s]["mentioned"] / by_score[s]["total"] for s in scores]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    x = np.arange(len(scores))
    bars = ax_top.bar(x, fractions, color="#8e6bb0", edgecolor="white")
    for bar, frac in zip(bars, fractions):
        if frac > 0:
            ax_top.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{frac:.1%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax_top.set_ylabel("Fraction citing\nrepetition/redundancy")
    ax_top.set_title("Grader mentions of repetition by legibility score")

    ax_bot.bar(x, totals, color="#aaaaaa", edgecolor="white")
    for i, t in enumerate(totals):
        ax_bot.text(
            i, t + max(totals) * 0.02, str(t), ha="center", va="bottom", fontsize=8
        )
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([str(s) for s in scores])
    ax_bot.set_xlabel("Legibility score")
    ax_bot.set_ylabel("Sample count")
    ax_bot.set_ylim(0, max(totals) * 1.25)

    fig.tight_layout()
    fig.savefig(output_dir / "grader_repetition_vs_score.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'grader_repetition_vs_score.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} samples")
    print(f"Grader cited repetition: {sum(r['grader_mentioned'] for r in rows)}")
    print(
        f"Mean repetition ratios — "
        f"3-gram: {np.mean([r['rep_3'] for r in rows]):.3f}, "
        f"4-gram: {np.mean([r['rep_4'] for r in rows]):.3f}, "
        f"5-gram: {np.mean([r['rep_5'] for r in rows]):.3f}"
    )
    comp = [r["compression_ratio"] for r in rows]
    print(f"Compression ratio — mean: {np.mean(comp):.3f}, std: {np.std(comp):.3f}")

    plot_ngram_vs_score(rows, OUTPUT_DIR)
    plot_compression_vs_score(rows, OUTPUT_DIR)
    plot_grader_mention_vs_score(rows, OUTPUT_DIR)


if __name__ == "__main__":
    main()
