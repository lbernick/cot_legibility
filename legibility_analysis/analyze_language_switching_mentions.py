"""Analyze how often grader explanations cite language switching as a legibility factor."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STREAMLIT_RUNS = Path(__file__).resolve().parent.parent / "streamlit_runs"
OUTPUT_DIR = Path(__file__).resolve().parent / "qwq_gpqa_combined"

# CJK unicode range (Chinese characters)
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

# Patterns the grader uses to mention language switching
LANG_SWITCH_RE = re.compile(
    r"non-?english|chinese|mandarin|korean|language switch|mixed language"
    r"|switches? between.*(?:english|language|chinese)",
    re.IGNORECASE,
)


def load_data() -> list[dict]:
    """Load all QwQ eval+inference pairs, joining on (question_id, sample_index)."""
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

            cjk_count = len(CJK_RE.findall(reasoning))
            lang_mentioned = bool(LANG_SWITCH_RE.search(explanation))

            rows.append(
                {
                    "score": score,
                    "cjk_count": cjk_count,
                    "lang_mentioned": lang_mentioned,
                }
            )
    return rows


def bucket_cjk(count: int) -> str:
    if count == 0:
        return "0"
    elif count <= 10:
        return "1-10"
    elif count <= 100:
        return "11-100"
    else:
        return "100+"


def plot_lang_mention_vs_cjk(rows: list[dict], output_dir: Path):
    """Bar chart: for each CJK bucket, what fraction mentions language switching."""
    bucket_order = ["0", "1-10", "11-100", "100+"]
    buckets = {b: {"total": 0, "mentioned": 0} for b in bucket_order}

    for r in rows:
        b = bucket_cjk(r["cjk_count"])
        buckets[b]["total"] += 1
        if r["lang_mentioned"]:
            buckets[b]["mentioned"] += 1

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bucket_order))
    totals = [buckets[b]["total"] for b in bucket_order]
    mentioned = [buckets[b]["mentioned"] for b in bucket_order]
    fractions = [m / t if t > 0 else 0 for m, t in zip(mentioned, totals)]

    bars = ax.bar(x, fractions, color="#4c72b0", edgecolor="white")
    for i, (bar, frac, tot) in enumerate(zip(bars, fractions, totals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{frac:.1%}\n(n={tot})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} chars" for b in bucket_order])
    ax.set_xlabel("Chinese characters in reasoning")
    ax.set_ylabel("Fraction mentioning language switching")
    ax.set_title("Grader mentions of language switching vs. Chinese character count")
    ax.set_ylim(0, min(1.0, max(fractions) * 1.4))
    fig.tight_layout()
    fig.savefig(output_dir / "lang_mention_vs_cjk.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'lang_mention_vs_cjk.png'}")


def plot_lang_mention_vs_score(rows: list[dict], output_dir: Path):
    """Two-panel plot: top = fraction mentioning language switching per score,
    bottom = sample count per score."""
    by_score = {}
    for r in rows:
        s = r["score"]
        if s not in by_score:
            by_score[s] = {"total": 0, "mentioned": 0}
        by_score[s]["total"] += 1
        if r["lang_mentioned"]:
            by_score[s]["mentioned"] += 1

    scores = sorted(by_score.keys())
    totals = [by_score[s]["total"] for s in scores]
    fractions = [
        by_score[s]["mentioned"] / by_score[s]["total"]
        if by_score[s]["total"] > 0
        else 0
        for s in scores
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    x = np.arange(len(scores))
    bars = ax_top.bar(x, fractions, color="#4c72b0", edgecolor="white")
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
    ax_top.set_ylabel("Fraction mentioning\nlanguage switching")
    ax_top.set_title("Grader mentions of language switching by illegibility score")

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
    fig.savefig(output_dir / "lang_mention_vs_score.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'lang_mention_vs_score.png'}")


def plot_score_lift_by_cjk(rows: list[dict], output_dir: Path):
    """Grouped bar chart: lift = P(score | CJK bucket) / P(score) for each bucket."""
    bucket_order = ["0", "1-10", "11-100", "100+"]
    total = len(rows)

    # P(score) — prior
    score_counts = {}
    for r in rows:
        score_counts[r["score"]] = score_counts.get(r["score"], 0) + 1
    scores = sorted(s for s in score_counts if s <= 7)
    prior = {s: score_counts[s] / total for s in scores}

    # P(score | bucket)
    bucket_totals = {b: 0 for b in bucket_order}
    bucket_score_counts = {b: {s: 0 for s in scores} for b in bucket_order}
    score_set = set(scores)
    for r in rows:
        if r["score"] not in score_set:
            continue
        b = bucket_cjk(r["cjk_count"])
        bucket_totals[b] += 1
        bucket_score_counts[b][r["score"]] += 1

    # Compute lift
    lifts = {}
    for b in bucket_order:
        lifts[b] = []
        for s in scores:
            p_cond = (
                bucket_score_counts[b][s] / bucket_totals[b] if bucket_totals[b] else 0
            )
            lifts[b].append(p_cond / prior[s] if prior[s] > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(scores))
    n_buckets = len(bucket_order)
    width = 0.8 / n_buckets
    colors = ["#a1c9f4", "#4c72b0", "#d4a373", "#c44e52"]

    for i, b in enumerate(bucket_order):
        offset = (i - n_buckets / 2 + 0.5) * width
        ax.bar(
            x + offset,
            lifts[b],
            width,
            label=f"{b} chars (n={bucket_totals[b]})",
            color=colors[i],
            edgecolor="white",
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scores])
    ax.set_xlabel("Illegibility score")
    ax.set_ylabel("Lift: P(score | Chinese char bucket) / P(score)")
    ax.set_title("Score distribution lift by Chinese character count")
    ax.legend(title="Chinese chars in reasoning")
    fig.tight_layout()
    fig.savefig(output_dir / "score_lift_by_cjk.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'score_lift_by_cjk.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} samples")
    print(f"Language switching mentioned: {sum(r['lang_mentioned'] for r in rows)}")
    print(f"Samples with CJK: {sum(r['cjk_count'] > 0 for r in rows)}")

    plot_lang_mention_vs_cjk(rows, OUTPUT_DIR)
    plot_lang_mention_vs_score(rows, OUTPUT_DIR)
    plot_score_lift_by_cjk(rows, OUTPUT_DIR)


if __name__ == "__main__":
    main()
