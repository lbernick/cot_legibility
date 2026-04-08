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

Plots the output of `categorize_legibility.py` as a normalized heatmap (category prevalence by score) and a horizontal bar chart (overall counts).

```bash
python legibility_analysis/plot_legibility_categories.py streamlit_runs/.../legibility_category_counts.json
```

## Why these exist

The core pipeline grades legibility with a single LLM call per sample. These scripts address follow-up questions:

- **Is the grader stable?** `regrade_legibility.py` + `compare_regrade.py` test whether re-running the same grader on the same inputs produces the same scores (grader drift).
- **Is the model stable?** `compare_generations.py` tests whether re-running inference produces similar legibility distributions (generation drift).
- **What drives the scores?** `categorize_legibility.py` + `plot_legibility_categories.py` extract and visualize the specific characteristics the grader cites in its explanations.
