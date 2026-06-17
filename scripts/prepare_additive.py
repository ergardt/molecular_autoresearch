"""
prepare_additive.py — sample molecules from a non-ChEMBL source by strategy.

Usage:
    python prepare_additive.py --source data/zinc_vs_chembl_full_QED.csv --strategy random --fraction 0.02 --output zinc_random_02.csv
    python prepare_additive.py --source data/coconut_smiles.csv --strategy random --fraction 0.05 --output coconut_random_05.csv
    python prepare_additive.py --source data/hmdb_smiles.csv --strategy high_qed --fraction 0.05 --output hmdb_high_qed_05.csv
"""

import argparse
import csv
import random
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

SEED = 42
BASELINE_SIZE = 500_000


def load_smiles(path: str) -> list[str]:
    """Read SMILES from CSV, find the smiles column automatically."""
    df = pd.read_csv(path)
    col = None
    for c in df.columns:
        if "smiles" in c.lower():
            col = c
            break
    if col is None:
        print(f"ERROR: no SMILES column in {path}. Columns: {df.columns.tolist()}")
        sys.exit(1)
    return df[col].dropna().astype(str).tolist()


def compute_qed(smi: str) -> float:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return float("nan")
        return QED.qed(mol)
    except Exception:
        return float("nan")


def compute_mw(smi: str) -> float:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return float("nan")
        return Descriptors.MolWt(mol)
    except Exception:
        return float("nan")


def prepare(source_path: str, strategy: str, fraction: float, output_path: str):
    n_target = int(fraction * BASELINE_SIZE)
    print(f"Source: {source_path}")
    print(f"Strategy: {strategy}, Fraction: {fraction}, Target: {n_target:,} molecules")

    smiles = load_smiles(source_path)
    print(f"Loaded {len(smiles):,} molecules")

    if len(smiles) < n_target:
        print(f"WARNING: source has {len(smiles):,} molecules, need {n_target:,}. Taking all.")
        n_target = len(smiles)

    if strategy == "random":
        rng = random.Random(SEED)
        selected = rng.sample(smiles, n_target)

    elif strategy == "high_qed":
        scored = [(smi, compute_qed(smi)) for smi in smiles]
        scored = [(s, q) for s, q in scored if not np.isnan(q)]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in scored[:n_target]]

    elif strategy == "low_qed":
        scored = [(smi, compute_qed(smi)) for smi in smiles]
        scored = [(s, q) for s, q in scored if not np.isnan(q)]
        scored.sort(key=lambda x: x[1])
        selected = [s for s, _ in scored[:n_target]]

    elif strategy == "high_mw":
        scored = [(smi, compute_mw(smi)) for smi in smiles]
        scored = [(s, m) for s, m in scored if not np.isnan(m)]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [s for s, _ in scored[:n_target]]

    elif strategy == "low_mw":
        scored = [(smi, compute_mw(smi)) for smi in smiles]
        scored = [(s, m) for s, m in scored if not np.isnan(m)]
        scored.sort(key=lambda x: x[1])
        selected = [s for s, _ in scored[:n_target]]

    elif strategy == "zinc_close":
        # High tanimoto_sum — structurally similar to ChEMBL
        df = pd.read_csv(source_path)
        df = df.sort_values("tanimoto_sum", ascending=False)
        col = None
        for c in df.columns:
            if "smiles" in c.lower():
                col = c
                break
        selected = df[col].head(n_target).tolist()

    elif strategy == "zinc_far":
        # Low tanimoto_sum AND scaffold_sum = 0
        df = pd.read_csv(source_path)
        df = df[(df["tanimoto_sum"] == df["tanimoto_sum"].min()) | True]  # get all, then filter
        df = df.sort_values("tanimoto_sum")
        # Prioritize scaffold_sum = 0, then lowest tanimoto_sum
        df_zero = df[df["scaffold_sum"] == 0].sort_values("tanimoto_sum")
        if len(df_zero) >= n_target:
            df = df_zero
        else:
            # Fill remaining with lowest tanimoto_sum regardless of scaffold
            remaining = n_target - len(df_zero)
            df_rest = df[~df.index.isin(df_zero.index)].sort_values("tanimoto_sum")
            df = pd.concat([df_zero, df_rest.head(remaining)])
        col = None
        for c in df.columns:
            if "smiles" in c.lower():
                col = c
                break
        selected = df[col].head(n_target).tolist()

    else:
        print(f"ERROR: unknown strategy '{strategy}'")
        sys.exit(1)

    print(f"Selected {len(selected):,} molecules")

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])
        for smi in selected:
            writer.writerow([smi])

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--strategy", required=True,
                        choices=["random", "high_qed", "low_qed", "high_mw", "low_mw",
                                 "zinc_close", "zinc_far"])
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.strategy, args.fraction, args.output)


if __name__ == "__main__":
    main()
