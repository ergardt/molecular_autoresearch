"""
baseline_analytics.py — подробная аналитика baseline-сэмплов.

Метрики:
  - молекулярные дескрипторы (MW, LogP, HBA, HBD, TPSA, Rotatable Bonds, Rings, etc.)
  - распределения (гистограммы, box plots, KDE)
  - сравнение 10 сэмплов между собой
  - правила Lipinski и Veber
  - уникальность, diversity (Tanimoto на Morgan fingerprints)
  - atom/bond composition

Все результаты → data/baseline_samples/analytics/
"""

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

SAMPLES_DIR = Path("data/baseline_samples")
OUT_DIR = SAMPLES_DIR / "analytics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
SEED_FILES = sorted([SAMPLES_DIR / f"chembl_500k_seed{i}.csv" for i in range(N_SEEDS)], key=lambda p: int(p.stem.split("seed")[1]))

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.figsize": (8, 5),
    "figure.constrained_layout.use": True,
})

COLORS = plt.cm.Set1(np.linspace(0, 1, N_SEEDS))


def read_smiles(path: Path) -> list[str]:
    """Read comma-per-character SMILES and reconstruct."""
    smiles_list = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            smiles_list.append("".join(row))
    return smiles_list


def mol_props(smi: str) -> dict | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotBonds": Lipinski.NumRotatableBonds(mol),
        "Rings": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "Atoms": mol.GetNumAtoms(),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "Bonds": mol.GetNumBonds(),
        "SmilesLen": len(smi),
    }


# ============================================================
# 1. READ ALL SAMPLES
# ============================================================
print("[1/5] Чтение сэмплов...")
all_data = {}  # seed -> DataFrame
all_smiles = {}  # seed -> list of smiles

for fpath in SEED_FILES:
    seed = int(fpath.stem.split("seed")[1])
    print(f"      seed={seed}: чтение...")
    smiles = read_smiles(fpath)
    all_smiles[seed] = smiles

    props_list = []
    invalid = 0
    for smi in smiles:
        p = mol_props(smi)
        if p is None:
            invalid += 1
        else:
            props_list.append(p)

    df = pd.DataFrame(props_list)
    all_data[seed] = df
    print(f"         {len(df):,} валидных, {invalid} невалидных")


# ============================================================
# 2. SUMMARY TABLE
# ============================================================
print("[2/5] Сводная таблица...")

numeric_cols = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "Rings", "AromaticRings", "HeavyAtoms", "Atoms", "Bonds", "SmilesLen"]

summary_rows = []
for seed in range(N_SEEDS):
    df = all_data[seed]
    row = {"seed": seed, "N": len(df)}
    for col in numeric_cols:
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std()
        row[f"{col}_min"] = df[col].min()
        row[f"{col}_median"] = df[col].median()
        row[f"{col}_max"] = df[col].max()
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "summary_stats.csv"
summary_df.to_csv(summary_path, index=False)
print(f"      Сохранено: {summary_path}")

# Lipinski & Veber per sample
lipinski_path = OUT_DIR / "lipinski_veber.csv"
lip_rows = []
for seed in range(N_SEEDS):
    df = all_data[seed]
    # Lipinski: MW <= 500, LogP <= 5, HBA <= 10, HBD <= 5
    lip_mw = (df["MW"] <= 500).mean()
    lip_logp = (df["LogP"] <= 5).mean()
    lip_hba = (df["HBA"] <= 10).mean()
    lip_hbd = (df["HBD"] <= 5).mean()
    lip_all = ((df["MW"] <= 500) & (df["LogP"] <= 5) & (df["HBA"] <= 10) & (df["HBD"] <= 5)).mean()
    # Veber: RotBonds <= 10, TPSA <= 140
    veber_rot = (df["RotBonds"] <= 10).mean()
    veber_tpsa = (df["TPSA"] <= 140).mean()
    veber_all = ((df["RotBonds"] <= 10) & (df["TPSA"] <= 140)).mean()
    lip_rows.append({
        "seed": seed,
        "N": len(df),
        "lipinski_MW_ok(%)": lip_mw * 100,
        "lipinski_LogP_ok(%)": lip_logp * 100,
        "lipinski_HBA_ok(%)": lip_hba * 100,
        "lipinski_HBD_ok(%)": lip_hbd * 100,
        "lipinski_all_ok(%)": lip_all * 100,
        "veber_RotBonds_ok(%)": veber_rot * 100,
        "veber_TPSA_ok(%)": veber_tpsa * 100,
        "veber_all_ok(%)": veber_all * 100,
    })

lip_df = pd.DataFrame(lip_rows)
lip_df.to_csv(lipinski_path, index=False)
print(f"      Сохранено: {lipinski_path}")


# ============================================================
# 3. DISTRIBUTION PLOTS
# ============================================================
print("[3/5] Построение графиков распределений...")

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.ravel()

descriptor_labels = {
    "MW": "Молекулярный вес",
    "LogP": "LogP (октазол/вода)",
    "HBA": "H-acceptors",
    "HBD": "H-donors",
    "TPSA": "TPSA (Å²)",
    "RotBonds": "Вращающиеся связи",
    "Rings": "Кольца",
    "AromaticRings": "Ароматические кольца",
    "HeavyAtoms": "Тяжёлые атомы",
    "Atoms": "Все атомы",
    "Bonds": "Связи",
    "SmilesLen": "Длина SMILES",
}

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    for seed in range(N_SEEDS):
        ax.hist(all_data[seed][col], bins=60, alpha=0.35, density=True, label=f"seed={seed}", color=COLORS[seed])
    ax.set_title(f"{descriptor_labels[col]}", fontsize=9)
    ax.set_xlabel(col)
    ax.set_ylabel("Плотность")
    ax.legend(fontsize=5, ncol=2)

plt.savefig(OUT_DIR / "distributions.png", bbox_inches=False)
plt.close()
print("      distributions.png")

# KDE overlay
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    data = all_data[idx] if N_SEEDS == 1 else all_data[0]
    # Plot median line across all seeds
    medians = [all_data[s][col].median() for s in range(N_SEEDS)]
    for seed in range(N_SEEDS):
        vals = all_data[seed][col].dropna()
        if len(vals) == 0:
            continue
        kde = vals.plot.kde(ax=ax, label=f"seed={seed}", color=COLORS[seed], linewidth=1.2, alpha=0.8)
    ax.set_title(f"{descriptor_labels[col]}", fontsize=9)
    ax.set_xlabel(col)
    ax.set_ylabel("KDE")
    ax.legend(fontsize=5, ncol=2)
    ax.set_ylim(bottom=0)

plt.savefig(OUT_DIR / "kde_distributions.png", bbox_inches=False)
plt.close()
print("      kde_distributions.png")


# ============================================================
# 4. BOX PLOTS — cross-sample comparison
# ============================================================
print("      Box plots...")

# Combine into one DataFrame with seed column
combined = pd.concat(
    [all_data[s].assign(seed=s) for s in range(N_SEEDS)],
    ignore_index=True,
)

box_cols = ["MW", "LogP", "TPSA", "HeavyAtoms", "RotBonds", "Rings", "HBA", "HBD"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.ravel()

for idx, col in enumerate(box_cols):
    ax = axes[idx]
    data_by_seed = [combined.loc[combined["seed"] == s, col].values for s in range(N_SEEDS)]
    bp = ax.boxplot(data_by_seed, labels=[f"s{i}" for i in range(N_SEEDS)], patch_artist=True)
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f"{descriptor_labels[col]}", fontsize=9)
    ax.set_xlabel("Seed")
    ax.set_ylabel(col)
    ax.set_yscale("log" if col in ("TPSA",) else "linear")

plt.savefig(OUT_DIR / "boxplots.png", bbox_inches=False)
plt.close()


# ============================================================
# 5. SCATTER: MW vs LogP, colored by seed
# ============================================================
print("      Scatter plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

scatter_pairs = [
    ("MW", "LogP", "Молекулярный вес vs LogP"),
    ("HeavyAtoms", "TPSA", "Тяжёлые атомы vs TPSA"),
    ("RotBonds", "Rings", "Вращающиеся связи vs Кольца"),
    ("HBA", "HBD", "HBA vs HBD"),
]

for idx, (xcol, ycol, title) in enumerate(scatter_pairs):
    ax = axes[idx]
    for seed in range(N_SEEDS):
        df = all_data[seed]
        # subsample for performance
        sub = df.sample(n=min(20000, len(df)), random_state=42)
        ax.scatter(sub[xcol], sub[ycol], s=0.5, alpha=0.15, label=f"seed={seed}", color=COLORS[seed])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.legend(fontsize=6, loc="best")

plt.savefig(OUT_DIR / "scatter_plots.png", bbox_inches=False)
plt.close()


# ============================================================
# 6. ATOM & BOND COMPOSITION
# ============================================================
print("      Atom/bond composition...")

ATOM_SYMS = {"C", "H", "N", "O", "S", "F", "P", "Cl", "Br", "I"}

def count_atoms(mol):
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in ATOM_SYMS:
            counts[sym] = counts.get(sym, 0) + 1
    return counts

def bond_types(mol):
    from rdkit.Chem import BondType
    bt = {
        "single": 0, "double": 0, "triple": 0, "aromatic": 0
    }
    for bond in mol.GetBonds():
        if bond.GetBondType() == BondType.SINGLE:
            bt["single"] += 1
        elif bond.GetBondType() == BondType.DOUBLE:
            bt["double"] += 1
        elif bond.GetBondType() == BondType.TRIPLE:
            bt["triple"] += 1
        elif bond.GetBondType() == BondType.AROMATIC:
            bt["aromatic"] += 1
    return bt

# Sample 10k per seed for atom/bond stats
atom_data = {}
bond_data = {}
for seed in range(N_SEEDS):
    smiles_sub = all_smiles[seed][:10000]
    atom_counts = {sym: [] for sym in ATOM_SYMS}
    bond_counts = {"single": [], "double": [], "triple": [], "aromatic": []}

    for smi in smiles_sub:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ac = count_atoms(mol)
        for sym in ATOM_SYMS:
            atom_counts[sym].append(ac.get(sym, 0))
        bc = bond_types(mol)
        for bt in bond_counts:
            bond_counts[bt].append(bc[bt])

    atom_data[seed] = atom_counts
    bond_data[seed] = bond_counts

# Atom composition bar chart
fig, ax = plt.subplots(figsize=(10, 6))
sym_order = ["C", "H", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
x = np.arange(len(sym_order))
width = 0.7 / N_SEEDS
for i, seed in enumerate(range(N_SEEDS)):
    means = [np.mean(atom_data[seed][sym]) for sym in sym_order]
    ax.bar(x + i * width, means, width, label=f"seed={seed}", alpha=0.7, color=COLORS[i])

ax.set_xlabel("Элемент")
ax.set_ylabel("Среднее количество атомов")
ax.set_title("Средняя композиция по атомам (10k молекул/сэмпл)")
ax.set_xticks(x + width * (N_SEEDS - 1) / 2)
ax.set_xticklabels(sym_order)
ax.legend(fontsize=7, ncol=2)

plt.savefig(OUT_DIR / "atom_composition.png", bbox_inches=False)
plt.close()

# Bond type pie (average across seeds)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bond type distribution
bond_labels = ["Single", "Double", "Triple", "Aromatic"]
bond_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
avg_bonds = []
for bt in bond_labels:
    vals = []
    for seed in range(N_SEEDS):
        vals.extend(bond_data[seed][bt])
    avg_bonds.append(np.mean(vals))

axes[0].pie(avg_bonds, labels=bond_labels, colors=bond_colors, autopct="%1.1f%%", startangle=90)
axes[0].set_title("Распределение типов связей (среднее)")

# Element fraction
elem_totals = {}
for sym in sym_order:
    vals = []
    for seed in range(N_SEEDS):
        vals.extend(atom_data[seed][sym])
    elem_totals[sym] = np.mean(vals)

axes[1].pie(list(elem_totals.values()), labels=list(elem_totals.keys()), autopct="%1.1f%%", startangle=90,
            colors=plt.cm.tab20(np.linspace(0, 1, len(sym_order))))
axes[1].set_title("Доля элементов (среднее кол-во атомов)")

plt.savefig(OUT_DIR / "composition_pies.png", bbox_inches=False)
plt.close()


# ============================================================
# 7. DIVERSITY — Tanimoto similarity (Morgan fingerprints)
# ============================================================
print("      Diversity (Tanimoto similarity)...")

DIVERSITY_N = 5000  # subsample for pairwise

diversity_results = {}
for seed in range(N_SEEDS):
    fps = []
    for smi in all_smiles[seed][:DIVERSITY_N]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = rdMolDescriptors.GetMorganFingerprint(mol, radius=2)
        fps.append(fp)

    # Random pairwise comparisons
    rng = np.random.default_rng(seed)
    n_pairs = min(50000, len(fps) * (len(fps) - 1) // 2)
    tanimoto_scores = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(fps), size=2, replace=False)
        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
        tanimoto_scores.append(sim)

    diversity_results[seed] = tanimoto_scores
    print(f"         seed={seed}: {len(tanimoto_scores)} пар, mean_sim={np.mean(tanimoto_scores):.4f}")

# Tanimoto distribution
fig, ax = plt.subplots(figsize=(8, 5))
for seed in range(N_SEEDS):
    ax.hist(diversity_results[seed], bins=50, alpha=0.4, density=True, label=f"seed={seed}", color=COLORS[seed])
ax.set_title("Распределение Tanimoto-сходства (Morgan r=2, 50k пар)")
ax.set_xlabel("Tanimoto similarity")
ax.set_ylabel("Плотность")
ax.legend(fontsize=7)

plt.savefig(OUT_DIR / "tanimoto_diversity.png", bbox_inches=False)
plt.close()


# ============================================================
# 8. UNIQUENESS — self-similarity within each sample
# ============================================================
print("      Уникальность...")

uniqueness = {}
for seed in range(N_SEEDS):
    smiles = all_smiles[seed]
    unique_count = len(set(smiles))
    uniqueness[seed] = {
        "total": len(smiles),
        "unique": unique_count,
        "unique_pct": unique_count / len(smiles) * 100,
    }

uniq_df = pd.DataFrame(uniqueness).T
uniq_df.index.name = "seed"
uniq_df.to_csv(OUT_DIR / "uniqueness.csv")
print(f"      Сохранено: uniqueness.csv")


# ============================================================
# 9. QPLP RADAR (drug-likeness)
# ============================================================
print("      Drug-likeness (QED, SA Score)...")

from rdkit.Chem import QED, SimDivFilters

qed_data = {}
sa_data = {}
for seed in range(N_SEEDS):
    qeds = []
    sas = []
    for smi in all_smiles[seed]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        qeds.append(QED.qed(mol))
        sas.append(SimDivFilters.calcScore(mol))
    qed_data[seed] = qeds
    sa_data[seed] = sas

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for seed in range(N_SEEDS):
    axes[0].hist(qed_data[seed], bins=50, alpha=0.35, density=True, label=f"seed={seed}", color=COLORS[seed])
axes[0].set_title("QED (drug-likeness)")
axes[0].set_xlabel("QED score")
axes[0].set_ylabel("Плотность")
axes[0].legend(fontsize=7)

for seed in range(N_SEEDS):
    axes[1].hist(sa_data[seed], bins=50, alpha=0.35, density=True, label=f"seed={seed}", color=COLORS[seed])
axes[1].set_title("Synthetic Accessibility Score")
axes[1].set_xlabel("SA score (1-10)")
axes[1].set_ylabel("Плотность")
axes[1].legend(fontsize=7)

plt.savefig(OUT_DIR / "drug_likeness.png", bbox_inches=False)
plt.close()

# QED summary
qed_summary = []
for seed in range(N_SEEDS):
    qed_summary.append({
        "seed": seed,
        "QED_mean": np.mean(qed_data[seed]),
        "QED_std": np.std(qed_data[seed]),
        "QED_median": np.median(qed_data[seed]),
        "SA_mean": np.mean(sa_data[seed]),
        "SA_std": np.std(sa_data[seed]),
        "SA_median": np.median(sa_data[seed]),
    })
pd.DataFrame(qed_summary).to_csv(OUT_DIR / "drug_likeness_stats.csv", index=False)


# ============================================================
# 10. SMILES LENGTH DISTRIBUTION
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
for seed in range(N_SEEDS):
    ax.hist(all_data[seed]["SmilesLen"], bins=50, alpha=0.35, density=True, label=f"seed={seed}", color=COLORS[seed])
ax.set_title("Длина канонического SMILES")
ax.set_xlabel("Количество символов")
ax.set_ylabel("Плотность")
ax.legend(fontsize=7)

plt.savefig(OUT_DIR / "smiles_length.png", bbox_inches=False)
plt.close()


# ============================================================
# 11. CROSS-SEED OVERLAP — how many molecules shared?
# ============================================================
print("      Пересечение сэмплов...")

smiles_sets = {seed: set(all_smiles[seed]) for seed in range(N_SEEDS)}
overlap_matrix = np.zeros((N_SEEDS, N_SEEDS), dtype=int)
for i in range(N_SEEDS):
    for j in range(N_SEEDS):
        if i == j:
            overlap_matrix[i][j] = len(smiles_sets[i])
        elif j > i:
            overlap = len(smiles_sets[i] & smiles_sets[j])
            overlap_matrix[i][j] = overlap
            overlap_matrix[j][i] = overlap

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(overlap_matrix, cmap="YlOrRd", aspect="equal")
ax.set_xticks(range(N_SEEDS))
ax.set_yticks(range(N_SEEDS))
ax.set_xticklabels([f"s{i}" for i in range(N_SEEDS)])
ax.set_yticklabels([f"s{i}" for i in range(N_SEEDS)])
ax.set_title("Пересечение сэмплов (общее количество молекул)")
ax.set_xlabel("Seed j")
ax.set_ylabel("Seed i")
fig.colorbar(im, ax=ax)

# Annotate cells
for i in range(N_SEEDS):
    for j in range(N_SEEDS):
        val = overlap_matrix[i][j]
        ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=6,
                color="white" if val < overlap_matrix.max() * 0.7 else "black")

plt.savefig(OUT_DIR / "seed_overlap.png", bbox_inches=False)
plt.close()

# Overlap as %
overlap_pct = overlap_matrix / overlap_matrix.max() * 100
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(overlap_pct, cmap="YlOrRd", aspect="equal", vmin=0, vmax=100)
ax.set_xticks(range(N_SEEDS))
ax.set_yticks(range(N_SEEDS))
ax.set_xticklabels([f"s{i}" for i in range(N_SEEDS)])
ax.set_yticklabels([f"s{i}" for i in range(N_SEEDS)])
ax.set_title("Пересечение сэмплов (%)")
for i in range(N_SEEDS):
    for j in range(N_SEEDS):
        ax.text(j, i, f"{overlap_pct[i][j]:.0f}%", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, label="%")

plt.savefig(OUT_DIR / "seed_overlap_pct.png", bbox_inches=False)
plt.close()


# ============================================================
# 12. WRITE REPORT
# ============================================================
print("[4/5] Генерация отчёта...")

report_lines = []
report_lines.append("# Аналитика baseline-сэмплов ChEMBL\n")
report_lines.append(f"- Количество сэмплов: {N_SEEDS}")
report_lines.append(f"- Молекул на сэмпл: 500 000")
report_lines.append(f"- Источник: data/chembl.csv (2 854 815 сырых записей)\n")

report_lines.append("## Молекулярные дескрипторы\n")
report_lines.append(summary_df.to_markdown(index=False))

report_lines.append("\n## Правила Lipinski и Veber (% молекул, удовлетворяющих)\n")
report_lines.append(lip_df.to_markdown(index=False))

report_lines.append("\n## Drug-likeness (QED, SA Score)\n")
report_lines.append(pd.DataFrame(qed_summary).to_markdown(index=False))

report_lines.append("\n## Уникальность\n")
report_lines.append(uniq_df.to_markdown())

report_lines.append("\n## Пересечение сэмплов (количество общих молекул)\n")
overlap_df = pd.DataFrame(overlap_matrix, index=[f"seed{i}" for i in range(N_SEEDS)], columns=[f"seed{i}" for i in range(N_SEEDS)])
report_lines.append(overlap_df.to_markdown())

report_lines.append("\n## Графики\n")
report_lines.append("- `distributions.png` — гистограммы 12 дескрипторов для каждого seed")
report_lines.append("- `kde_distributions.png` — KDE-оценки распределений")
report_lines.append("- `boxplots.png` — box plots для сравнения сэмплов")
report_lines.append("- `scatter_plots.png` — scatter: MW vs LogP, HeavyAtoms vs TPSA, RotBonds vs Rings, HBA vs HBD")
report_lines.append("- `atom_composition.png` — средняя композиция по элементам")
report_lines.append("- `composition_pies.png` — pie charts: типы связей и элементы")
report_lines.append("- `tanimoto_diversity.png` — распределение Tanimoto-сходства")
report_lines.append("- `drug_likeness.png` — QED и SA Score")
report_lines.append("- `smiles_length.png` — длина SMILES")
report_lines.append("- `seed_overlap.png` / `seed_overlap_pct.png` — матрица пересечения сэмплов")

report = "\n".join(report_lines)
report_path = OUT_DIR / "report.md"
report_path.write_text(report)
print(f"      Сохранено: {report_path}")

print(f"\n[5/5] Готово. Все результаты в {OUT_DIR}/")
print(f"\nСодержимое:")
for f in sorted(OUT_DIR.iterdir()):
    size = f.stat().st_size / 1024
    print(f"  {f.name}: {size:.0f} KB")
