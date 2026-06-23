"""Plot SYK fine-tuning results: lines per source with baseline as horizontal reference."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tsv = "experiments/exp_2_var_2/syk_results.tsv"
df = pd.read_csv(tsv, sep="\t")

baseline = df[df["model"] == "baseline"].iloc[0]

# SYK dataset intrinsic metrics
syk_data = {
    "score": 0.7304,
    "validity": 1.000,
    "mean_qed": 0.497,
    "pct_lipinski": 0.738,
    "scaffold_entropy": 0.879,
    "int_div": 0.831,
    "uniqueness": 0.9606,
    "novelty": 1.0000,
}

sources = {
    "zinc": "ZINC",
    "coconut": "COCONUT",
    "hmdb": "HMDB",
}

colors = {"ZINC": "#1f77b4", "COCONUT": "#2ca02c", "HMDB": "#ff7f0e"}
markers = {"ZINC": "o", "COCONUT": "s", "HMDB": "^"}

metrics = [
    ("score", "Score"),
    ("validity", "Validity"),
    ("mean_qed", "Mean QED"),
    ("pct_lipinski", "Pct Lipinski"),
    ("scaffold_entropy", "Scaffold Entropy"),
    ("int_div", "Int. Diversity"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
]

out_dir = "experiments/exp_2_var_2"

for col, title in metrics:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bl_val = baseline[col]

    # Baseline — horizontal line
    ax.hlines(y=bl_val, xmin=-0.005, xmax=0.21, color="#888888", linewidth=3, alpha=0.5,
              label=f"baseline ({bl_val:.3f})")
    ax.plot(0, bl_val, "D", color="#888888", markersize=9, zorder=6)

    for key, name in sources.items():
        sub = df[df["model"].str.startswith(key)].sort_values("model")
        sub = sub.copy()
        sub["fraction"] = sub["model"].apply(lambda x: float(x.split("_")[-1]) / 100)
        ax.plot(sub["fraction"], sub[col], marker=markers[name], linewidth=2,
                markerfacecolor="none", markeredgewidth=1.5, markersize=9,
                color=colors[name], label=name)
        ax.scatter(sub["fraction"], sub[col], color=colors[name], s=50, zorder=5)

    # SYK dataset — horizontal line
    syk_val = syk_data[col]
    ax.hlines(y=syk_val, xmin=-0.005, xmax=0.21, color="#9467bd", linewidth=2, alpha=0.6,
              label=f"SYK dataset ({syk_val:.3f})")

    ax.set_title(f"SYK Fine-tuning — {title}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Additive fraction (of 500K ChEMBL)")
    ax.set_xticks([0, 0.02, 0.05, 0.10, 0.20])
    ax.set_xlim(-0.005, 0.21)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)

    out = f"{out_dir}/syk_{col}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# Combined overview — 4 key metrics in 2x2
overview_metrics = [("score", "Score"), ("mean_qed", "Mean QED"), ("scaffold_entropy", "Scaffold Entropy"), ("int_div", "Int. Diversity")]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, (col, title) in zip(axes.flatten(), overview_metrics):
    bl_val = baseline[col]
    ax.hlines(y=bl_val, xmin=-0.005, xmax=0.21, color="#888888", linewidth=2, alpha=0.5,
              label=f"baseline ({bl_val:.3f})")
    ax.plot(0, bl_val, "D", color="#888888", markersize=7, zorder=6)

    # SYK dataset — horizontal line
    syk_val = syk_data[col]
    ax.hlines(y=syk_val, xmin=-0.005, xmax=0.21, color="#9467bd", linewidth=1.5, alpha=0.6,
              label=f"SYK dataset ({syk_val:.3f})")

    for key, name in sources.items():
        sub = df[df["model"].str.startswith(key)].sort_values("model").copy()
        sub["fraction"] = sub["model"].apply(lambda x: float(x.split("_")[-1]) / 100)
        ax.plot(sub["fraction"], sub[col], marker=markers[name], linewidth=1.5,
                markerfacecolor="none", markeredgewidth=1.2, markersize=7,
                color=colors[name], label=name)
        ax.scatter(sub["fraction"], sub[col], color=colors[name], s=40, zorder=5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([0, 0.02, 0.05, 0.10, 0.20])
    ax.set_xlim(-0.005, 0.21)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

fig.suptitle("SYK Fine-tuning — All Sources", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = f"{out_dir}/syk_overview.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
