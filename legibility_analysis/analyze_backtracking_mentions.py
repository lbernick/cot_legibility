"""Analyze how grader citations of backtracking/uncertainty relate to reasoning markers and scores."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STREAMLIT_RUNS = Path(__file__).resolve().parent.parent / "streamlit_runs"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Grader explanation: mentions of backtracking, hedging, self-correction, confusion
# Excludes "repetitive/redundant" which is about verbosity, not uncertainty
GRADER_BACKTRACK_RE = re.compile(
    r"backtrack|self[- ]correct|re-?evaluat|revis(?:it|e)|reconsider"
    r"|hesitat|uncertain|confus|indeci|circular"
    r"|frequent use of .{0,20}(?:wait|maybe|perhaps|actually|alternatively|hmm)",
    re.IGNORECASE,
)

# Reasoning text: hedging/backtracking tokens produced by the model
REASONING_HEDGE_RE = re.compile(
    r"\bwait\b|\bhmm\b|\bactually,|\balternatively[,.]"
    r"|\bmaybe I|\bperhaps I|\bon second thought"
    r"|\blet me (?:reconsider|think|re-?evaluate)"
    r"|\bI'?m not sure|\bno,? that'?s (?:not right|wrong)"
    r"|\bhold on|\bwait,",
    re.IGNORECASE,
)


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

            hedge_count = len(REASONING_HEDGE_RE.findall(reasoning))
            grader_mentioned = bool(GRADER_BACKTRACK_RE.search(explanation))

            rows.append(
                {
                    "score": score,
                    "hedge_count": hedge_count,
                    "grader_mentioned": grader_mentioned,
                }
            )
    return rows


def bucket_hedges(count: int) -> str:
    if count <= 10:
        return "0-10"
    elif count <= 50:
        return "11-50"
    elif count <= 150:
        return "51-150"
    else:
        return "150+"


def plot_grader_vs_hedge_count(rows: list[dict], output_dir: Path):
    """Fraction of grader explanations citing backtracking, by hedge-token bucket."""
    bucket_order = ["0-10", "11-50", "51-150", "150+"]
    buckets = {b: {"total": 0, "mentioned": 0} for b in bucket_order}

    for r in rows:
        b = bucket_hedges(r["hedge_count"])
        buckets[b]["total"] += 1
        if r["grader_mentioned"]:
            buckets[b]["mentioned"] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bucket_order))
    totals = [buckets[b]["total"] for b in bucket_order]
    fractions = [
        buckets[b]["mentioned"] / buckets[b]["total"] if buckets[b]["total"] else 0
        for b in bucket_order
    ]

    bars = ax.bar(x, fractions, color="#6a994e", edgecolor="white")
    for bar, frac, tot in zip(bars, fractions, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{frac:.1%}\n(n={tot})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} tokens" for b in bucket_order])
    ax.set_xlabel("Hedge/backtracking tokens in reasoning")
    ax.set_ylabel("Fraction where grader cites backtracking")
    ax.set_title("Grader mentions of backtracking vs. hedge tokens in reasoning")
    ax.set_ylim(0, min(1.0, max(fractions) * 1.4))
    fig.tight_layout()
    fig.savefig(output_dir / "grader_backtrack_vs_hedge_count.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'grader_backtrack_vs_hedge_count.png'}")


def plot_grader_backtrack_vs_score(rows: list[dict], output_dir: Path):
    """Two-panel: fraction citing backtracking per score + sample count."""
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
    fractions = [
        by_score[s]["mentioned"] / by_score[s]["total"] if by_score[s]["total"] else 0
        for s in scores
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    x = np.arange(len(scores))
    bars = ax_top.bar(x, fractions, color="#6a994e", edgecolor="white")
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
    ax_top.set_ylabel("Fraction citing\nbacktracking/uncertainty")
    ax_top.set_title("Grader mentions of backtracking by illegibility score")

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
    fig.savefig(output_dir / "grader_backtrack_vs_score.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'grader_backtrack_vs_score.png'}")


def plot_score_lift_by_hedges(rows: list[dict], output_dir: Path):
    """Lift = P(score | hedge bucket) / P(score) for each bucket."""
    bucket_order = ["0-10", "11-50", "51-150", "150+"]
    total = len(rows)

    score_counts: dict[int, int] = {}
    for r in rows:
        score_counts[r["score"]] = score_counts.get(r["score"], 0) + 1
    scores = sorted(s for s in score_counts if s <= 7)
    prior = {s: score_counts[s] / total for s in scores}

    bucket_totals = {b: 0 for b in bucket_order}
    bucket_score_counts = {b: {s: 0 for s in scores} for b in bucket_order}
    score_set = set(scores)
    for r in rows:
        if r["score"] not in score_set:
            continue
        b = bucket_hedges(r["hedge_count"])
        bucket_totals[b] += 1
        bucket_score_counts[b][r["score"]] += 1

    lifts = {}
    for b in bucket_order:
        lifts[b] = [
            (bucket_score_counts[b][s] / bucket_totals[b]) / prior[s]
            if bucket_totals[b] and prior[s]
            else 0
            for s in scores
        ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scores))
    n_buckets = len(bucket_order)
    width = 0.8 / n_buckets
    colors = ["#a7c957", "#6a994e", "#386641", "#bc4749"]

    for i, b in enumerate(bucket_order):
        offset = (i - n_buckets / 2 + 0.5) * width
        ax.bar(
            x + offset,
            lifts[b],
            width,
            label=f"{b} tokens (n={bucket_totals[b]})",
            color=colors[i],
            edgecolor="white",
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scores])
    ax.set_xlabel("Illegibility score")
    ax.set_ylabel("Lift: P(score | hedge bucket) / P(score)")
    ax.set_title("Score distribution lift by hedge/backtracking token count")
    ax.legend(title="Hedge tokens in reasoning")
    fig.tight_layout()
    fig.savefig(output_dir / "score_lift_by_hedges.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'score_lift_by_hedges.png'}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} samples")
    print(f"Grader cited backtracking: {sum(r['grader_mentioned'] for r in rows)}")
    print(f"Samples with hedge tokens: {sum(r['hedge_count'] > 0 for r in rows)}")

    plot_grader_vs_hedge_count(rows, PLOTS_DIR)
    plot_grader_backtrack_vs_score(rows, PLOTS_DIR)
    plot_score_lift_by_hedges(rows, PLOTS_DIR)


if __name__ == "__main__":
    main()
