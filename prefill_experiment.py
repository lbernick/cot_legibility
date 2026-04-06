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
with open("selected_qwq/original_answer.json") as f:
    selected_df = pd.DataFrame(json.load(f)).set_index(["question_id", "sample_index"])

plot_df = selected_df.join(new_correctness["correct_pct"].rename("prefill_correct_pct")).dropna(subset=["prefill_correct_pct"]).copy()
plot_df["orig_correct_pct"] = (plot_df["correctness"] == "correct").astype(float) * 100
plot_df["delta"] = plot_df["prefill_correct_pct"] - plot_df["orig_correct_pct"]
print(f"{len(plot_df)} data points")

x_axes = ["legibility_score", "reasoning_non_latin_chars", "reasoning_perplexity", "avg_entropy"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, col in zip(axes, x_axes):
    mask = plot_df[col].notna() & np.isfinite(plot_df[col])
    if col == "reasoning_non_latin_chars":
        mask &= plot_df[col] <= plot_df[col].quantile(0.95)
    x, y = plot_df.loc[mask, col], plot_df.loc[mask, "delta"]
    r = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, alpha=0.4, s=20)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(col)
    ax.set_ylabel("Change in correctness")
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

fig.suptitle("Change in correctness from prefill")
plt.tight_layout()
plt.savefig("plots/prefill_delta_scatter.png", dpi=150)
plt.show()

# %%
diff_plot = plot_df.join(original_correctness["correct_pct"].rename("orig_q_correct_pct"), on="question_id")

x, y = diff_plot["orig_q_correct_pct"], diff_plot["delta"]
r = np.corrcoef(x, y)[0, 1]
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(x, y, alpha=0.4, s=20)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Baseline correctness")
ax.set_ylabel("Change in correctness")
ax.set_title("Baseline correctness vs uplift from prefill")
ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
plt.tight_layout()
plt.savefig("plots/prefill_delta_vs_difficulty.png", dpi=150)
plt.show()

# %%
groups = [
    plot_df.loc[plot_df["reasoning_non_latin_chars"] <= 5, "delta"],
    plot_df.loc[plot_df["reasoning_non_latin_chars"] > 5, "delta"],
]
fig, ax = plt.subplots(figsize=(5, 4))
ax.boxplot(groups, labels=["≤5 non-latin chars", ">5 non-latin chars"])
for i, g in enumerate(groups, 1):
    ax.scatter([i] * len(g), g, alpha=0.4, s=15, zorder=3)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_ylabel("Change in correctness")
ax.set_title("Correctness uplift from prefill")
fig.savefig("plots/prefill_delta_non_latin.png")

# %%
mask = plot_df["reasoning_tokens"].notna() & np.isfinite(plot_df["reasoning_tokens"])
x, y = plot_df.loc[mask, "reasoning_tokens"], plot_df.loc[mask, "delta"]
r = np.corrcoef(x, y)[0, 1]
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(x, y, alpha=0.4, s=20)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Reasoning length (tokens)")
ax.set_ylabel("Change in correctness")
ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
plt.tight_layout()
plt.savefig("plots/prefill_delta_vs_reasoning_len.png", dpi=150)
plt.show()

# %%
def residualize(x, y):
    coef = np.polyfit(x, y, 1)
    return y - np.polyval(coef, x)

cols = ["legibility_score", "reasoning_perplexity", "avg_entropy"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, col in zip(axes, cols):
    mask = plot_df[[col, "reasoning_len", "delta"]].notna().all(axis=1) & np.isfinite(plot_df[col])
    sub = plot_df.loc[mask]
    x_resid = residualize(sub["reasoning_len"], sub[col])
    y_resid = residualize(sub["reasoning_len"], sub["delta"])
    r = np.corrcoef(x_resid, y_resid)[0, 1]
    ax.scatter(x_resid, y_resid, alpha=0.4, s=20)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"{col} (residual)")
    ax.set_ylabel("delta (residual)")
    ax.text(0.05, 0.95, f"r = {r:.2f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

fig.suptitle("Uplift vs legibility metrics, controlling for reasoning length")
plt.tight_layout()
plt.savefig("plots/prefill_delta_partial_corr.png", dpi=150)
plt.show()

# %%
