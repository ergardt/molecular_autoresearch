"""Plot Phase B results: bar charts comparing ZINC strategies at 10%."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tsv = "experiments/exp_2_var_2/results.tsv"
df = pd.read_csv(tsv, sep="\t")

baseline = df[df["phase"] == "baseline"].iloc[0]

phase_b = df[df["phase"] == "B"].copy()
phase_b["strategy_label"] = phase_b["strategy"].str.replace("_", " ").str.title()
phase_b = phase_b.sort_values("score", ascending=False)

strategies = phase_b["strategy_label"].tolist()
colors_bar = []
for _, row in phase_b.iterrows():
    # Green if score >= baseline, red if below
    if row["score"] >= baseline["score"]:
        colors_bar.append("#2ca02c")
    else:
        colors_bar.append("#d62728")

metrics = [
    ("validity", "Validity"),
    ("mean_qed", "Mean QED"),
    ("scaffold_entropy", "Scaffold Entropy"),
    ("int_div", "Int. Diversity"),
    ("uniqueness", "Uniqueness"),
    ("novelty", "Novelty"),
    ("score", "Score"),
]

out_dir = "experiments/exp_2_var_2"

for col, title in metrics:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bl_val = baseline[col]
    values = phase_b[col].tolist()

    bars = ax.bar(strategies, values, color=colors_bar, width=0.6, edgecolor="white", linewidth=0.5)

    # Baseline line
    ax.axhline(y=bl_val, color="#333333", linewidth=2, linestyle="-", label=f"baseline ({bl_val:.3f})")
    ax.set_ylabel(title)
    ax.set_title(f"Phase B — ZINC 10% — {title}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(strategies, rotation=30, ha="right")

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (bl_val * 0.01),
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Set y limits to show range
    min_val = min(values + [bl_val]) * 0.95
    max_val = max(values + [bl_val]) * 1.05
    ax.set_ylim(min_val, max_val)

    out = f"{out_dir}/phase_b_{col}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# Combined overview — all 6 metrics in 2 rows (excluding score)
overview_metrics = [(c, t) for c, t in metrics if c != "score"]
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for ax, (col, title) in zip(axes.flatten(), overview_metrics):
    bl_val = baseline[col]
    values = phase_b[col].tolist()
    ax.bar(strategies, values, color=colors_bar, width=0.6, edgecolor="white")
    ax.axhline(y=bl_val, color="#333333", linewidth=2, label=f"baseline ({bl_val:.3f})")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    min_val = min(values + [bl_val]) * 0.95
    max_val = max(values + [bl_val]) * 1.05
    ax.set_ylim(min_val, max_val)

fig.suptitle("Phase B — ZINC 10% — All Metrics", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
out = f"{out_dir}/phase_b_overview.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
