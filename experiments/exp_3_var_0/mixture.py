"""
mixture.py — create a pretraining data mixture from multiple sources.

The agent modifies only MIXTURE_CONFIG at the top of this file.
Run:
    python mixture.py              # create mixed_pretrain.csv
    python mixture.py --dry-run    # show composition without writing file

Selection strategies:
  random              — random sampling (seed=42)
  scaffold_first      — S1: maximize unique Murcko scaffolds
  property_coverage   — S2: uniform coverage (LogP, MW, TPSA)
  farthest_from_ref   — S3: farthest from ChEMBL (requires tanimoto_sum column)
  most_non_druglike   — S4: most Lipinski/QED violations
  high_predicted      — S5: high predicted pIC50 (requires predicted_pIC50 column)
  by_chemical_class   — S6: uniform sampling from each chemical_super_class
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold


# Resolve project root (2 levels up from experiments/exp_3_var_0/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ===========================================================================
# MIXTURE CONFIGURATION — agent modifies only this block
# ===========================================================================

TOTAL_SIZE = 500_000
SEED = 42

# Format: "path_to_csv": (count, strategy)
# The sum of all counts must equal TOTAL_SIZE
MIXTURE_CONFIG = {
    "data/coconut_smiles.csv": (500000, "by_chemical_class"),
}

# Path to reference ChEMBL sample for S3 (farthest_from_ref)
# If None, a random 20k sample from ChEMBL is used
CHEMBL_REF_PATH = None

# ===========================================================================
# Utilities
# ===========================================================================

CACHE_DIR = Path(".mixture_cache")
CONFIG_HASH_FILE = CACHE_DIR / "config_hash.txt"


def _resolve(path: str) -> str:
    """Resolve a relative path against PROJECT_ROOT."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(PROJECT_ROOT / p)


def _config_hash() -> str:
    """Compute hash of current configuration for caching."""
    raw = json.dumps(
        {"TOTAL": TOTAL_SIZE, "SEED": SEED,
         "MIXTURE": MIXTURE_CONFIG, "REF": CHEMBL_REF_PATH},
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _cache_is_valid() -> bool:
    """Check whether configuration hash matches cache."""
    if not CONFIG_HASH_FILE.exists():
        return False
    return CONFIG_HASH_FILE.read_text().strip() == _config_hash()


def _save_cache():
    """Save configuration hash."""
    CACHE_DIR.mkdir(exist_ok=True)
    CONFIG_HASH_FILE.write_text(_config_hash())


def _smiles_column(df: pd.DataFrame) -> str:
    """Find SMILES column by name."""
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    raise ValueError(f"No smiles column in {df.columns.tolist()}")


def load_smiles(path: str) -> pd.Series:
    """
    Load SMILES from CSV, canonicalize, and filter valid molecules.

    Supports:
    - CSV with 'smiles' column
    - One SMILES per line (no header)
    """
    resolved = _resolve(path)
    df = pd.read_csv(resolved)

    if "smiles" not in [c.lower() for c in df.columns]:
        col = df.columns[0]
        raw = df[col].dropna().astype(str).str.strip()
    else:
        col = _smiles_column(df)
        raw = df[col].dropna().astype(str).str.strip()

    raw = raw.str.split().str[0]

    print(f"  Loading {len(raw):,} rows from {path}...")
    canonical = []
    valid = 0

    for smi in raw:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical.append(Chem.MolToSmiles(mol))
            valid += 1
        if len(canonical) % 100_000 == 0:
            print(f"    {len(canonical):,} valid / {valid:,} processed...")

    print(f"  {valid:,} valid molecules from {path}")
    return pd.Series(canonical, name="smiles")


def load_full_df(path: str) -> pd.DataFrame:
    """Load full DataFrame (for strategies requiring additional columns)."""
    return pd.read_csv(_resolve(path))


# ===========================================================================
# Selection strategies
# ===========================================================================

def strategy_random(smiles: pd.Series, n: int, seed: int = SEED) -> list[str]:
    """S0: Random sampling."""
    rng = random.Random(seed)
    return rng.sample(sorted(smiles.unique()), min(n, len(smiles.unique())))


def strategy_scaffold_first(smiles: pd.Series, n: int, seed: int = SEED) -> list[str]:
    """S1: Maximize unique Murcko scaffolds."""
    rng = random.Random(seed)
    unique = list(smiles.unique())
    rng.shuffle(unique)

    scaffold_groups = defaultdict(list)

    for smi in unique:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol, includeChirality=False)
        except Exception:
            scaffold = "__NO_SCAFFOLD__"
        scaffold_groups[scaffold].append(smi)

    one_per = [rng.choice(mols) for mols in scaffold_groups.values()]
    rng.shuffle(one_per)

    if len(one_per) >= n:
        return one_per[:n]

    used = set(one_per)
    remaining = [s for s in unique if s not in used]
    rng.shuffle(remaining)
    return one_per + remaining[:n - len(one_per)]


def strategy_most_non_druglike(smiles: pd.Series, n: int, seed: int = SEED) -> list[str]:
    """S4: Most violations of drug-like rules (Lipinski + QED)."""
    rng = random.Random(seed)
    unique = list(smiles.unique())

    def count_violations(smi: str) -> int:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return 0
        v = 0
        if Descriptors.MolWt(mol) > 500:
            v += 1
        if Descriptors.MolLogP(mol) > 5:
            v += 1
        if Descriptors.NumHDonors(mol) > 5:
            v += 1
        if Descriptors.NumHAcceptors(mol) > 10:
            v += 1
        try:
            if QED.qed(mol) < 0.3:
                v += 1
        except Exception:
            pass
        return v

    scored = [(count_violations(smi), smi) for smi in unique]
    scored.sort(key=lambda x: -x[0])

    result = []
    grouped = defaultdict(list)
    for v, s in scored:
        grouped[v].append(s)

    for v in sorted(grouped.keys(), reverse=True):
        mols = grouped[v]
        rng.shuffle(mols)
        result.extend(mols)
        if len(result) >= n:
            break

    return result[:n]


# ===========================================================================
# Strategy dispatcher
# ===========================================================================

STRATEGIES = {
    "random": strategy_random,
    "scaffold_first": strategy_scaffold_first,
    "most_non_druglike": strategy_most_non_druglike,
}


def select(smiles: pd.Series, n: int, strategy: str) -> list[str]:
    """Select n molecules using specified strategy."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    print(f"  Strategy: {strategy}, target: {n:,} molecules")
    return STRATEGIES[strategy](smiles, n)


# ===========================================================================
# Main pipeline
# ===========================================================================

def make_mixture(dry_run: bool = False) -> Path:
    """Create pretraining data mixture."""
    total_config = sum(cfg[0] for cfg in MIXTURE_CONFIG.values())
    assert total_config == TOTAL_SIZE, (
        f"Sum of counts ({total_config}) != TOTAL_SIZE ({TOTAL_SIZE})"
    )

    output_path = Path("mixed_pretrain.csv")
    pretrain_set_path = Path("pretrain_smiles.txt")

    if _cache_is_valid() and output_path.exists() and not dry_run:
        print(f"Config hash {_config_hash()} matches cache. Skipping.")
        return output_path

    all_smiles = []

    if dry_run:
        print(f"[DRY RUN] Would write {output_path}")
        return output_path

    for data_path, (count, strategy) in MIXTURE_CONFIG.items():
        print(f"\nProcessing {data_path} ({count:,}, strategy={strategy})")
        smiles = load_smiles(data_path)
        selected = select(smiles, count, strategy)
        all_smiles.extend(selected)

    unique_smiles = list(dict.fromkeys(all_smiles))

    output_path.write_text("smiles\n" + "\n".join(unique_smiles) + "\n")
    pretrain_set_path.write_text("\n".join(unique_smiles) + "\n")

    _save_cache()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create pretraining data mixture")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Mixture config: TOTAL_SIZE={TOTAL_SIZE:,}, SEED={SEED}")
    make_mixture(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
