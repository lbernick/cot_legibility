# Legibility Analysis

Extended analysis scripts for investigating the reliability and structure of legibility scores. These scripts build on top of the core pipeline (`src/`) and operate on its outputs (`streamlit_runs/`).

All scripts should be run from the **project root**.

## Scripts

### `regrade_legibility.py`

Re-runs legibility grading on existing inference outputs to check whether the grader model (GPT-4o) produces consistent scores over time. Saves results as `evaluation_regrade.json` alongside the original `evaluation.json`.

```bash
python legibility_analysis/regrade_legibility.py              # all configured runs
python legibility_analysis/regrade_legibility.py R1-Distill   # filter by substring
```

### `compare_regrade.py`

Compares original vs. regraded legibility scores to quantify grader drift. Prints per-run and aggregate statistics (correlation, exact match rate, mean absolute difference) and generates scatter + histogram plots.

```bash
python legibility_analysis/compare_regrade.py
```

### `compare_generations.py`

Compares legibility score distributions between original inference runs and new runs of the same model, to check whether the model's legibility characteristics have changed.

```bash
python legibility_analysis/compare_generations.py
```

### `categorize_legibility.py`

Two-pass LLM pipeline that (1) discovers a taxonomy of legibility characteristics from a stratified sample of grader explanations, then (2) classifies every explanation against that taxonomy. Outputs category lists and per-score counts.

```bash
python legibility_analysis/categorize_legibility.py streamlit_runs/.../evaluation.json
python legibility_analysis/categorize_legibility.py streamlit_runs/.../evaluation.json --model gpt-4o --batch-size 30
```

### `plot_legibility_categories.py`

Plots the output of `categorize_legibility.py` as a heatmap (raw counts colored by per-score percentage) and a horizontal bar chart (overall counts). Categories are sorted by overall frequency.

```bash
python legibility_analysis/plot_legibility_categories.py streamlit_runs/.../legibility_category_counts.json
```

### `analyze_language_switching.py`

Samples reasoning texts that contain non-Latin characters (CJK, Cyrillic, Korean), extracts windows around each switch point, and uses an LLM to classify each non-English segment as contextually coherent or incoherent (with translations).

```bash
python legibility_analysis/analyze_language_switching.py \
  streamlit_runs/*qwq_gpqa/inference.json \
  --output-dir legibility_analysis/qwq_gpqa_combined
```

### `analyze_language_switching_mentions.py`

Analyzes how often grader explanations cite language switching as a legibility factor, and cross-references with actual non-Latin character detection in the reasoning text.

```bash
python legibility_analysis/analyze_language_switching_mentions.py
```

### `plot_language_switching.py`

Plots the output of `analyze_language_switching.py` as a pie chart showing the proportion of coherent vs incoherent non-English segments (filtering out English and error results).

```bash
python legibility_analysis/plot_language_switching.py legibility_analysis/qwq_gpqa_combined/language_switching_analysis.json
```

## Why these exist

The core pipeline grades legibility with a single LLM call per sample. These scripts address follow-up questions:

- **Is the grader stable?** `regrade_legibility.py` + `compare_regrade.py` test whether re-running the same grader on the same inputs produces the same scores (grader drift).
- **Is the model stable?** `compare_generations.py` tests whether re-running inference produces similar legibility distributions (generation drift).
- **What drives the scores?** `categorize_legibility.py` + `plot_legibility_categories.py` extract and visualize the specific characteristics the grader cites in its explanations.
- **Is language switching meaningful?** `analyze_language_switching.py` + `plot_language_switching.py` determine whether non-English segments in reasoning are contextually coherent code-switching or nonsensical output. `analyze_language_switching_mentions.py` cross-references grader mentions of language switching with actual detection.
