"""Compute metrics on the SYK inhibitors dataset itself."""

import math
import pandas as pd
import numpy as np
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold


def main():
    df = pd.read_csv("data/syk_inhibitors.csv")
    col = next(c for c in df.columns if "smiles" in c.lower())
    smiles = df[col].dropna().astype(str).tolist()
    print(f"Molecules: {len(smiles)}")

    # Validity
    mols = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
    validity = len(mols) / len(smiles)
    print(f"Validity: {validity:.3f}")

    # Uniqueness
    canonical = [Chem.MolToSmiles(m) for m in mols]
    unique = set(canonical)
    uniqueness = len(unique) / len(mols) if mols else 0
    print(f"Uniqueness: {uniqueness:.4f}")

    # QED
    qed_vals = []
    for m in mols:
        try:
            qed_vals.append(QED.qed(m))
        except Exception:
            pass
    mean_qed = float(np.mean(qed_vals)) if qed_vals else 0
    print(f"Mean QED: {mean_qed:.3f}")

    # Lipinski
    lipinski_pass = sum(
        1 for m in mols
        if Descriptors.MolWt(m) <= 500 and Descriptors.MolLogP(m) <= 5
        and Descriptors.NumHDonors(m) <= 5
        and Descriptors.NumHAcceptors(m) <= 10
    )
    pct_lipinski = lipinski_pass / len(mols) if mols else 0
    print(f"Pct Lipinski: {pct_lipinski:.3f}")

    # Scaffold entropy
    scaffold_counts = Counter()
    for m in mols:
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
            scaffold_counts[s] += 1
        except Exception:
            pass
    total = sum(scaffold_counts.values())
    if total > 1:
        probs = [c / total for c in scaffold_counts.values()]
        raw = -sum(p * math.log(p) for p in probs)
        scaffold_entropy = raw / math.log(len(scaffold_counts))
    else:
        scaffold_entropy = 0
    print(f"Scaffold entropy: {scaffold_entropy:.3f}")
    print(f"Unique scaffolds: {len(scaffold_counts)}")

    # Internal diversity
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048) for m in mols[:500]]
    cap = min(len(fps), 200)
    sims = [DataStructs.TanimotoSimilarity(fps[i], fps[j]) for i in range(cap) for j in range(i + 1, cap)]
    mean_tanimoto = float(np.mean(sims)) if sims else 0
    int_div = 1.0 - mean_tanimoto
    print(f"Int div: {int_div:.3f}")

    # Score
    score = validity * scaffold_entropy * int_div
    print(f"Score: {score:.4f}")

    # Extra
    mw_vals = [Descriptors.MolWt(m) for m in mols]
    logp_vals = [Descriptors.MolLogP(m) for m in mols]
    print(f"Mean MW: {np.mean(mw_vals):.0f}")
    print(f"Mean LogP: {np.mean(logp_vals):.2f}")

    # Output as TSV line
    print(f"\nsyk_dataset\t{score:.4f}\t{validity:.3f}\t{mean_qed:.3f}\t{pct_lipinski:.3f}\t{scaffold_entropy:.3f}\t{int_div:.3f}\t{uniqueness:.4f}\t1.0000")


if __name__ == "__main__":
    main()
