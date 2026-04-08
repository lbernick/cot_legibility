"""Analyze whether non-English segments in reasoning are contextually coherent or nonsensical."""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference.providers import get_provider
from src.utils.models import get_model_config

CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
KOREAN_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
NON_LATIN_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u0400-\u04ff\uac00-\ud7af\u1100-\u11ff]+"
)

CLASSIFY_PROMPT = """\
Below is an excerpt from a model's chain-of-thought reasoning (in English) that contains non-English text segments. For each non-English segment, determine:

1. What language it is
2. What it means (translate it)
3. Whether it is **coherent** (meaningful in context — e.g. a discourse marker, technical term, or continuation of the reasoning in another language) or **incoherent** (nonsensical, out of context, or garbled)

Return JSON:
{{"segments": [{{"text": "non-English text", "language": "...", "translation": "...", "coherent": true/false}}, ...], "summary": "coherent" or "incoherent" or "mixed"}}

The summary should be "coherent" if all/nearly all segments are coherent, "incoherent" if all/nearly all are incoherent, "mixed" otherwise.

REASONING EXCERPT:
{excerpt}"""


def extract_windows(reasoning: str, window: int = 200) -> list[str]:
    """Extract text windows around each non-Latin span, merging overlapping ones."""
    spans = []
    for m in NON_LATIN_RE.finditer(reasoning):
        start = max(0, m.start() - window)
        end = min(len(reasoning), m.end() + window)
        spans.append((start, end))

    if not spans:
        return []

    merged = [spans[0]]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return [reasoning[s:e] for s, e in merged]


def has_non_latin(text: str) -> bool:
    return bool(NON_LATIN_RE.search(text))


def load_samples(inference_paths: list[str]) -> list[dict]:
    samples = []
    for path in inference_paths:
        with open(path) as f:
            items = json.load(f)
        for item in items:
            reasoning = item.get("reasoning") or ""
            if has_non_latin(reasoning):
                samples.append(
                    {
                        "question_id": item.get("question_id"),
                        "file": path,
                        "reasoning": reasoning,
                    }
                )
    return samples


def call_llm(provider, model_config, prompt: str) -> dict:
    config = {**model_config, "temperature": 0.0}
    result = provider.generate(prompt, config)
    text = result["answer"]
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1]
    return json.loads(text.strip())


def analyze_sample(sample: dict, provider, model_config) -> dict:
    windows = extract_windows(sample["reasoning"])
    # Combine all windows into one excerpt, separated by markers
    excerpt = "\n[...]\n".join(windows)
    # Cap at ~4000 chars to stay within reasonable token limits
    if len(excerpt) > 4000:
        excerpt = excerpt[:4000]

    result = call_llm(provider, model_config, CLASSIFY_PROMPT.format(excerpt=excerpt))
    return {
        "question_id": sample["question_id"],
        "file": sample["file"],
        "num_windows": len(windows),
        "excerpt": excerpt,
        "summary": result["summary"],
        "segments": result["segments"],
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze language switching coherence")
    parser.add_argument("inference_paths", nargs="+", help="Path(s) to inference.json")
    parser.add_argument("--sample-size", type=int, default=75)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_model_config(args.model)
    provider = get_provider(config["provider"])

    print(f"Loading from {len(args.inference_paths)} files...")
    all_samples = load_samples(args.inference_paths)
    print(f"Found {len(all_samples)} samples with non-Latin characters")

    random.seed(42)
    samples = random.sample(all_samples, min(args.sample_size, len(all_samples)))
    print(f"Sampled {len(samples)} for analysis")

    results = []
    futures = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for sample in samples:
            future = executor.submit(analyze_sample, sample, provider, config)
            futures[future] = sample

        for future in tqdm(as_completed(futures), total=len(futures), desc="Analyzing"):
            try:
                results.append(future.result())
            except Exception as e:
                sample = futures[future]
                print(f"Failed {sample['question_id']}: {e}")
                results.append(
                    {
                        "question_id": sample["question_id"],
                        "file": sample["file"],
                        "summary": "error",
                        "error": str(e),
                    }
                )

    summaries = [r["summary"] for r in results]
    counts = {}
    for s in summaries:
        counts[s] = counts.get(s, 0) + 1

    total_segments = sum(len(r.get("segments", [])) for r in results)
    coherent_segments = sum(
        sum(1 for s in r.get("segments", []) if s.get("coherent")) for r in results
    )

    languages = {}
    for r in results:
        for seg in r.get("segments", []):
            lang = seg.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1

    output = {
        "sample_size": len(samples),
        "total_with_non_latin": len(all_samples),
        "summary_counts": dict(sorted(counts.items(), key=lambda x: -x[1])),
        "segment_stats": {
            "total": total_segments,
            "coherent": coherent_segments,
            "incoherent": total_segments - coherent_segments,
            "coherent_pct": round(coherent_segments / total_segments * 100, 1)
            if total_segments
            else 0,
        },
        "languages": dict(sorted(languages.items(), key=lambda x: -x[1])),
        "results": results,
    }

    output_file = output_dir / "language_switching_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_file}")
    print(f"\n=== Summary ({len(samples)} samples, {total_segments} segments) ===")
    print(f"Sample-level: {counts}")
    print(
        f"Segment-level: {coherent_segments}/{total_segments} coherent ({output['segment_stats']['coherent_pct']}%)"
    )
    print(f"Languages: {languages}")


if __name__ == "__main__":
    main()
