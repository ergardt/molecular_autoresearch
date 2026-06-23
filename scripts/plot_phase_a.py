"""Plot Phase A results: 4 separate figures, one metric each."""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tsv = "experiments/exp_2_var_2/results.tsv"
df = pd.read_csv(tsv, sep="\t")

baseline = df[df["phase"] == "baseline"].iloc[0]

sources = {
    "zinc": "ZINC",
    "coconut": "COCONUT",
    "hmdb": "HMDB",
}

colors = {"ZINC": "#1f77b4", "COCONUT": "#2ca02c", "HMDB": "#ff7f0e"}
markers = {"ZINC": "o", "COCONUT": "s", "HMDB": "^"}

metrics = [
    ("validity", "Validity"),
    ("mean_qed", "Mean QED"),
    ("scaffold_entropy", "Scaffold Entropy"),
    ("int_div", "Int. Diversity"),
]

out_dir = "experiments/exp_2_var_2"

for col, title in metrics:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bl_val = baseline[col]

    # baseline plateau — horizontal line from x=0 across full range
    ax.hlines(y=bl_val, xmin=-0.005, xmax=0.21, color="#888888", linewidth=3, alpha=0.5,
              label=f"baseline ({bl_val:.3f})")
    ax.plot(0, bl_val, "o", color="#888888", markersize=10, zorder=6)

    for key, name in sources.items():
        sub = df[(df["dataset"].str.contains(key)) & (df["phase"] == "A")].sort_values("fraction")
        ax.plot(sub["fraction"], sub[col], marker=markers[name], linewidth=2,
                markerfacecolor="none", markeredgewidth=1.5, markersize=9,
                color=colors[name], label=name)
        ax.scatter(sub["fraction"], sub[col], color=colors[name], s=50, zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Additive fraction (of 500K ChEMBL)")
    ax.set_xticks([0, 0.02, 0.05, 0.10, 0.20])
    ax.set_xlim(-0.005, 0.21)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)

    out = f"{out_dir}/phase_a_{col}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
