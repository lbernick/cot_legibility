"""Plot language switching coherence analysis results."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_coherence_pie(data: dict, output_path: Path):
    results = [r for r in data["results"] if r.get("summary") != "error"]

    coherent = 0
    incoherent = 0
    for r in results:
        for seg in r.get("segments", []):
            lang = seg.get("language", "").lower()
            if "english" in lang or "mathematical" in lang:
                continue
            if seg.get("coherent"):
                coherent += 1
            else:
                incoherent += 1

    labels = ["Coherent", "Incoherent"]
    sizes = [coherent, incoherent]
    colors = ["#2ecc71", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p * sum(sizes) / 100))})",
        startangle=90,
        textprops={"fontsize": 12},
    )
    for t in autotexts:
        t.set_fontsize(11)

    total = coherent + incoherent
    ax.set_title(
        f"Coherence of Non-English Segments\n({total} segments from {len(results)} samples)",
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Pie chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "analysis_path", help="Path to language_switching_analysis.json"
    )
    parser.add_argument(
        "--output-dir", help="Output directory (default: same as input)"
    )
    args = parser.parse_args()

    analysis_path = Path(args.analysis_path)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_path.parent

    with open(analysis_path) as f:
        data = json.load(f)

    plot_coherence_pie(data, output_dir / "language_switching_coherence.png")


if __name__ == "__main__":
    main()
