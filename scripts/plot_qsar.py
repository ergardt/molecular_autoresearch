"""Plot QSAR pIC50 predictions per fine-tuned model."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

tsv = "experiments/exp_2_var_2/predictions.tsv"
df = pd.read_csv(tsv, sep="\t")

# Load experimental SYK pIC50 values
syk_df = pd.read_csv("data/syk_inhibitors.csv")
syk_col = next(c for c in syk_df.columns if "pic50" in c.lower())
syk_exp_pic50 = syk_df[syk_col].dropna()
syk_exp_mean = float(syk_exp_pic50.mean())
syk_exp_median = float(syk_exp_pic50.median())

# Parse model names
df["source"] = df["model"].apply(lambda x: "baseline" if x == "syk_baseline" else x.replace("syk_", "").split("_")[0])
df["fraction"] = df["model"].apply(lambda x: 0 if x == "syk_baseline" else float(x.split("_")[-1]) / 100)
df["label"] = df["model"].apply(lambda x: x.replace("syk_", ""))

source_colors = {"baseline": "#888888", "zinc": "#1f77b4", "coconut": "#2ca02c", "hmdb": "#ff7f0e"}
source_labels = {"baseline": "Baseline", "zinc": "ZINC", "coconut": "COCONUT", "hmdb": "HMDB"}

out_dir = "experiments/exp_2_var_2"

# 1. Box plot per model + SYK experimental
fig, ax = plt.subplots(figsize=(14, 5))
data_by_model = [df[df["model"] == m]["pIC50"].values for m in df["model"].unique()]
data_by_model.append(syk_exp_pic50.values)
labels = [m.replace("syk_", "") for m in df["model"].unique()] + ["SYK experimental"]
colors = [source_colors[df[df["model"] == m]["source"].iloc[0]] for m in df["model"].unique()] + ["#9467bd"]

bp = ax.boxplot(data_by_model, labels=labels, patch_artist=True, widths=0.5,
                medianprops=dict(color="white", linewidth=1.5),
                whiskerprops=dict(linewidth=1),
                capprops=dict(linewidth=1),
                flierprops=dict(markerfacecolor='xkcd:gray', marker='.', alpha=0.3, markersize=3))

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel("Predicted pIC50")
ax.set_title("SYK Fine-tuning — QSAR pIC50 Predictions", fontsize=14, fontweight="bold")
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
ax.grid(axis="y", alpha=0.3)

out = f"{out_dir}/qsar_boxplot.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# 2. Mean pIC50 per source as line plot (like phase plots)
summary = df.groupby(["source", "fraction", "label"])["pIC50"].agg(["mean", "median"]).reset_index()

fig, ax = plt.subplots(figsize=(7, 4.5))

# SYK experimental pIC50 reference line
ax.hlines(y=syk_exp_mean, xmin=-0.005, xmax=0.21, color="#9467bd", linewidth=2, alpha=0.6,
          label=f"SYK experimental mean ({syk_exp_mean:.2f})")

for key, name in [("zinc", "ZINC"), ("coconut", "COCONUT"), ("hmdb", "HMDB")]:
    sub = summary[summary["source"] == key].sort_values("fraction")
    ax.plot(sub["fraction"], sub["mean"], marker="o", linewidth=2,
            markerfacecolor="none", markeredgewidth=1.5, markersize=9,
            color=source_colors[key], label=name)
    ax.scatter(sub["fraction"], sub["mean"], color=source_colors[key], s=50, zorder=5)

# Baseline point
bl = summary[summary["source"] == "baseline"]
if len(bl) > 0:
    ax.hlines(y=bl["mean"].iloc[0], xmin=-0.005, xmax=0.21, color="#888888", linewidth=3, alpha=0.5,
              label=f"baseline ({bl['mean'].iloc[0]:.3f})")
    ax.plot(0, bl["mean"].iloc[0], "D", color="#888888", markersize=9, zorder=6)

ax.set_title("SYK Fine-tuning — Mean Predicted pIC50", fontsize=14, fontweight="bold")
ax.set_xlabel("Additive fraction (of 500K ChEMBL)")
ax.set_ylabel("Mean pIC50")
ax.set_xticks([0, 0.02, 0.05, 0.10, 0.20])
ax.set_xlim(-0.005, 0.21)
ax.legend(fontsize=10, loc="upper left")
ax.grid(alpha=0.3)

out = f"{out_dir}/qsar_mean_pic50.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# 3. Distribution of pIC50 per source + SYK experimental
fig, ax = plt.subplots(figsize=(7, 4.5))
for key, name in [("zinc", "ZINC"), ("coconut", "COCONUT"), ("hmdb", "HMDB"), ("baseline", "Baseline")]:
    sub = df[df["source"] == key]["pIC50"]
    ax.hist(sub, bins=100, alpha=0.3, density=True, label=name, color=source_colors[key], linewidth=0)
ax.hist(syk_exp_pic50, bins=100, alpha=0.6, density=True, label="SYK experimental", color="#9467bd", linewidth=0)

ax.set_title("SYK Fine-tuning — pIC50 Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted pIC50")
ax.legend(fontsize=10, loc="upper right")
ax.grid(alpha=0.3)

out = f"{out_dir}/qsar_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

# 4. Summary table
print("\nPer-model summary:")
summary_table = df.groupby("model").agg(
    count=("pIC50", "count"),
    mean=("pIC50", "mean"),
    median=("pIC50", "median"),
    std=("pIC50", "std"),
    max=("pIC50", "max"),
    min=("pIC50", "min"),
    pct_above_7=("pIC50", lambda x: (x >= 7.0).mean()),
    pct_above_8=("pIC50", lambda x: (x >= 8.0).mean()),
).sort_values("mean", ascending=False)
print(summary_table.to_string())
