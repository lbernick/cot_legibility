#%%
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
original_df = pd.read_parquet("qwq_gpqa.parquet")
# %%
with open("selected_qwq/forced_answer_graded.json") as f:
    data = json.load(f)

prefill_df = pd.DataFrame(data["results"])
prefill_df["prefill_correctness"] = prefill_df["prefill_correctness"].apply(lambda x: x["correctness"] if isinstance(x, dict) else None)
prefill_df.head()

# %%
original_df.head()
# %%
original_correctness = original_df[original_df["correctness"] != "unknown"].groupby("question_id").agg(
    correct_pct=("correctness", lambda x: (x == "correct").mean() * 100),
)
original_correctness.head()

# %%
new_correctness = prefill_df[prefill_df["prefill_correctness"] != "unknown"].groupby(["question_id", "sample_index"]).agg(
    correct_pct=("prefill_correctness", lambda x: (x == "correct").mean() * 100),
)

# %%
orig = prefill_df.groupby(["question_id", "sample_index"])["original_correctness"].first().map({"correct": 100.0, "incorrect": 0.0})

delta_df = new_correctness.join(orig).assign(delta=lambda d: d["correct_pct"] - d["original_correctness"])

# %%
augmented_df = original_df.join(
    new_correctness["correct_pct"].rename("prefill_correct_pct"),
    on=["question_id", "sample_index"],
)

# %%
plot_df = augmented_df.dropna(subset=["prefill_correct_pct"]).copy()
plot_df["orig_correct_pct"] = (plot_df["correctness"] == "correct").astype(float) * 100
plot_df["delta"] = plot_df["prefill_correct_pct"] - plot_df["orig_correct_pct"]

x_axes = ["legibility_score", "reasoning_non_latin_chars", "reasoning_perplexity", "avg_entropy"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, col in zip(axes, x_axes):
    mask = plot_df[col].notna() & np.isfinite(plot_df[col])
    if col == "reasoning_non_latin_chars":
        mask &= plot_df[col] <= plot_df[col].quantile(0.97)
    x, y = plot_df.loc[mask, col], plot_df.loc[mask, "delta"]
    r = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, alpha=0.4, s=20)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(col)
    ax.set_ylabel("prefill - original correct %")
    ax.set_title(f"r = {r:.2f}")

plt.tight_layout()
plt.savefig("plots/prefill_delta_scatter.png", dpi=150)
plt.show()

# %%
