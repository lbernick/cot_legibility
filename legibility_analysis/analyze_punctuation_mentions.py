"""Analyze how grader citations of punctuation/formatting relate to reasoning anomalies and scores."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STREAMLIT_RUNS = Path(__file__).resolve().parent.parent / "streamlit_runs"
OUTPUT_DIR = Path(__file__).resolve().parent / "qwq_gpqa_combined"

# Grader explanation: mentions of punctuation or formatting issues
GRADER_PUNCTUATION_RE = re.compile(
    r"punctuation|formatting (?:issue|inconsistenc)"
    r"|inconsistent (?:formatting|spacing|punctuation)"
    r"|excessive punctuation",
    re.IGNORECASE,
)

# Reasoning text: isolated punctuation on its own line, repeated 3+ times
ISOLATED_PUNCT_RE = re.compile(r"(?:^[ \t]*[,;.!?]{1,3}[ \t]*\n){3,}", re.MULTILINE)

# Reasoning text: 4+ consecutive newlines
EXCESS_NEWLINES_RE = re.compile(r"\n{4,}")


def count_punct_anomalies(text: str) -> int:
    return len(ISOLATED_PUNCT_RE.findall(text)) + len(EXCESS_NEWLINES_RE.findall(text))


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

            anomaly_count = count_punct_anomalies(reasoning)
            grader_mentioned = bool(GRADER_PUNCTUATION_RE.search(explanation))

            rows.append(
                {
                    "score": score,
                    "anomaly_count": anomaly_count,
                    "grader_mentioned": grader_mentioned,
                }
            )
    return rows


def bucket_anomalies(count: int) -> str:
    if count == 0:
        return "0"
    elif count <= 3:
        return "1-3"
    elif count <= 10:
        return "4-10"
    else:
        return "10+"


def plot_grader_vs_anomaly_count(rows: list[dict], output_dir: Path):
    bucket_order = ["0", "1-3", "4-10", "10+"]
    buckets = {b: {"total": 0, "mentioned": 0} for b in bucket_order}

    for r in rows:
        b = bucket_anomalies(r["anomaly_count"])
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

    bars = ax.bar(x, fractions, color="#e07a5f", edgecolor="white")
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
    ax.set_xticklabels([f"{b} anomalies" for b in bucket_order])
    ax.set_xlabel("Punctuation anomalies in reasoning")
    ax.set_ylabel("Fraction where grader cites punctuation/formatting")
    ax.set_title("Grader mentions of punctuation vs. anomaly count in reasoning")
    ax.set_ylim(0, min(1.0, max(fractions) * 1.4))
    fig.tight_layout()
    fig.savefig(output_dir / "grader_punct_vs_anomaly_count.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'grader_punct_vs_anomaly_count.png'}")


def plot_grader_punct_vs_score(rows: list[dict], output_dir: Path):
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
    bars = ax_top.bar(x, fractions, color="#e07a5f", edgecolor="white")
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
    ax_top.set_ylabel("Fraction citing\npunctuation/formatting")
    ax_top.set_title("Grader mentions of punctuation by illegibility score")

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
    fig.savefig(output_dir / "grader_punct_vs_score.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'grader_punct_vs_score.png'}")


def plot_score_lift_by_anomalies(rows: list[dict], output_dir: Path):
    bucket_order = ["0", "1-3", "4-10", "10+"]
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
        b = bucket_anomalies(r["anomaly_count"])
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
    colors = ["#f2cc8f", "#e07a5f", "#81b29a", "#3d405b"]

    for i, b in enumerate(bucket_order):
        offset = (i - n_buckets / 2 + 0.5) * width
        ax.bar(
            x + offset,
            lifts[b],
            width,
            label=f"{b} anomalies (n={bucket_totals[b]})",
            color=colors[i],
            edgecolor="white",
        )

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in scores])
    ax.set_xlabel("Illegibility score")
    ax.set_ylabel("Lift: P(score | anomaly bucket) / P(score)")
    ax.set_title("Score distribution lift by punctuation anomaly count")
    ax.legend(title="Punctuation anomalies")
    fig.tight_layout()
    fig.savefig(output_dir / "score_lift_by_punct.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir / 'score_lift_by_punct.png'}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_data()
    print(f"Loaded {len(rows)} samples")
    print(
        f"Grader cited punctuation/formatting: {sum(r['grader_mentioned'] for r in rows)}"
    )
    print(f"Samples with anomalies: {sum(r['anomaly_count'] > 0 for r in rows)}")

    plot_grader_vs_anomaly_count(rows, OUTPUT_DIR)
    plot_grader_punct_vs_score(rows, OUTPUT_DIR)
    plot_score_lift_by_anomalies(rows, OUTPUT_DIR)


if __name__ == "__main__":
    main()
