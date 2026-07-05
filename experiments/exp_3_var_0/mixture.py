"""
mixture.py — создание смеси pretrain данных из нескольких источников.

Агент меняет только MIXTURE_CONFIG вверху файла.
Запуск:
    python mixture.py                          # создать mixed_pretrain.csv
    python mixture.py --dry-run                # показать состав, не писать файл

Стратегии отбора:
  random              — случайная выборка (seed=42)
  scaffold_first      — S1: максимум уникальных Murcko-скаффолдов
  property_coverage   — S2: равномерное покрытие (LogP, MW, TPSA)
  farthest_from_ref   — S3: самые далёкие от ChEMBL (нужен столбец tanimoto_sum)
  most_non_druglike   — S4: больше всего нарушений Lipinski/QED
  high_predicted      — S5: высокий предсказанный pIC50 (нужен столбец predicted_pIC50)
  by_chemical_class   — S6: равномерно из каждого chemical_super_class
"""

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

# Resolve project root (2 levels up from experiments/exp_3_var_0/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ===========================================================================
# КОНФИГУРАЦИЯ СМЕСИ — агент меняет только этот блок
# ===========================================================================

TOTAL_SIZE = 500_000
SEED = 42

# Формат: "path_to_csv": (count, strategy)
# Сумма всех count должна быть == TOTAL_SIZE
#
# Примеры:
#   baseline (чистый ChEMBL):
#       MIXTURE_CONFIG = {
#           "data/chembl.csv": (500000, "random"),
#       }
#
#   80% ChEMBL + 20% COCONUT scaffold-first:
#       MIXTURE_CONFIG = {
#           "data/chembl.csv": (400000, "random"),
#           "data/coconut_smiles.csv": (100000, "scaffold_first"),
#       }
#
#   60% ChEMBL + 20% COCONUT + 20% HMDB:
#       MIXTURE_CONFIG = {
#           "data/chembl.csv": (300000, "random"),
#           "data/coconut_smiles.csv": (100000, "scaffold_first"),
#           "data/hmdb_smiles.csv": (100000, "property_coverage"),
#       }

MIXTURE_CONFIG = {
    "data/chembl.csv": (300000, "random"),
    "data/zinc_vs_chembl_full_QED.csv": (200000, "farthest_from_ref"),
}

# Путь к референсному сэмпу ChEMBL для S3 (farthest_from_ref)
# Если не задан, берётся random.sample(chembl, 20000, seed=42)
CHEMBL_REF_PATH = None  # например, "data/chembl_ref_20k.csv"

# ===========================================================================
# Утилиты
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
    """Считать хэш текущей конфигурации для кэширования."""
    raw = json.dumps({"TOTAL": TOTAL_SIZE, "SEED": SEED,
                      "MIXTURE": MIXTURE_CONFIG, "REF": CHEMBL_REF_PATH},
                     sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _cache_is_valid() -> bool:
    """Проверить, совпадает ли хэш конфига с кэшем."""
    if not CONFIG_HASH_FILE.exists():
        return False
    return CONFIG_HASH_FILE.read_text().strip() == _config_hash()


def _save_cache():
    """Сохранить хэш конфига."""
    CACHE_DIR.mkdir(exist_ok=True)
    CONFIG_HASH_FILE.write_text(_config_hash())


def _smiles_column(df: pd.DataFrame) -> str:
    """Найти столбец со SMILES по имени."""
    for c in df.columns:
        if "smiles" in c.lower():
            return c
    raise ValueError(f"No smiles column in {df.columns.tolist()}")


def load_smiles(path: str) -> pd.Series:
    """Загрузить SMILES из CSV, канонизировать, отфильтровать валидные.

    Поддерживает два формата:
    - CSV со столбцом 'smiles' (заголовок)
    - Одна SMILES на строку (без заголовка)
    """
    resolved = _resolve(path)
    df = pd.read_csv(resolved)

    # Handle headerless files (one SMILES per line)
    if "smiles" not in [c.lower() for c in df.columns]:
        # Single column, no header — treat as SMILES
        col = df.columns[0]
        raw = df[col].dropna().astype(str).str.strip()
    else:
        col = _smiles_column(df)
        raw = df[col].dropna().astype(str).str.strip()

    # Take first SMILES if multiple (separated by space)
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
    """Загрузить полный DataFrame (для стратегий, needing additional columns)."""
    return pd.read_csv(_resolve(path), header=0)


# ===========================================================================
# Стратегии отбора
# ===========================================================================

def strategy_random(smiles: pd.Series, n: int, seed: int = SEED) -> list[str]:
    """S0: Случайная выборка."""
    rng = random.Random(seed)
    return rng.sample(sorted(smiles.unique()), min(n, len(smiles.unique())))


def strategy_scaffold_first(smiles: pd.Series, n: int, seed: int = SEED) -> list[str]:
    """S1: Максимум уникальных Murcko-скаффолдов.

    Группируем по скаффолду, из каждой группы берём 1 молекулу (рандом).
    Если нужно больше — берём оставшиеся рандомно.
    """
    rng = random.Random(seed)
    unique = list(smiles.unique())
    rng.shuffle(unique)

    scaffold_groups = defaultdict(list)
    processed = 0
    for smi in unique:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol, includeChirality=False)
        except Exception:
            scaffold = "__NO_SCAFFOLD__"
        scaffold_groups[scaffold].append(smi)
        processed += 1
        if processed % 50_000 == 0:
            print(f"    Scaffolds: {len(scaffold_groups):,} from {processed:,} molecules...")

    # One per scaffold
    one_per = [rng.choice(mols) for mols in scaffold_groups.values()]
    rng.shuffle(one_per)

    if len(one_per) >= n:
        return one_per[:n]

    # Fill remaining with random from unused
    used = set(one_per)
    remaining = [s for s in unique if s not in used]
    rng.shuffle(remaining)
    return one_per + remaining[:n - len(one_per)]


def strategy_property_coverage(smiles: pd.Series, n: int, seed: int = SEED,
                                n_clusters: int = 20) -> list[str]:
    """S2: Равномерное покрытие пространства (LogP, MW, TPSA).

    Считаем дескрипторы, делаем k-means-подобную кластеризацию,
    берём равномерно из каждого кластера.
    """
    rng = random.Random(seed)
    unique = list(smiles.unique())

    props = []
    valid_smiles = []
    for smi in unique:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            logp = Descriptors.MolLogP(mol)
            mw = Descriptors.MolWt(mol)
            tpsa = Descriptors.TPSA(mol)
            if math.isnan(logp) or math.isnan(mw):
                continue
            props.append([logp, mw, tpsa])
            valid_smiles.append(smi)
        except Exception:
            continue

    props = np.array(props)
    # Normalize
    mins = props.min(axis=0)
    maxs = props.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    normed = (props - mins) / ranges

    # Simple grid-based clustering
    # Assign each molecule to a 3D grid cell
    cell_size = int(np.ceil(n_clusters ** (1 / 3)))
    cells = normed * (cell_size - 0.001)
    cell_ids = tuple(cells[:, 0].astype(int)), tuple(cells[:, 1].astype(int)), tuple(cells[:, 2].astype(int))

    cell_groups = defaultdict(list)
    for i, (c0, c1, c2) in enumerate(zip(*cell_ids)):
        cell_groups[(c0, c1, c2)].append(valid_smiles[i])

    # Sample uniformly from cells
    cell_list = list(cell_groups.keys())
    rng.shuffle(cell_list)

    result = []
    for cell in cell_list:
        mols = cell_groups[cell]
        rng.shuffle(mols)
        result.extend(mols)
        if len(result) >= n:
            break

    # If not enough, fill randomly
    if len(result) < n:
        remaining = [s for s in valid_smiles if s not in set(result)]
        rng.shuffle(remaining)
        result.extend(remaining)

    return result[:n]


def strategy_farthest_from_ref(smiles: pd.Series, n: int,
                                ref_path: str = None, seed: int = SEED) -> list[str]:
    """S3: Самые далёкие от ChEMBL референса.

    Ожидает, что в DataFrame есть столбец с Tanimoto-расстоянием до ChEMBL.
    Для файлов *_smiles.csv ищем соответствующий файл с танimoto.
    Для zinc_vs_chembl_full_QED.csv используем столбец 'tanimoto_sum'.
    """
    rng = random.Random(seed)
    unique = list(smiles.unique())

    # Try to find a tanimoto column
    # Check if there's a corresponding file with tanimoto info
    tanimoto_scores = _load_tanimoto_scores(smiles, ref_path)

    if tanimoto_scores is not None:
        # Sort by distance (higher = farther from ChEMBL)
        scored = [(tanimoto_scores.get(s, 0.0), s) for s in unique]
        scored.sort(key=lambda x: -x[0])
        result = [s for _, s in scored[:n]]
    else:
        # Fallback: random
        print("    WARNING: No tanimoto scores found, falling back to random")
        return strategy_random(pd.Series(unique), n, seed)

    return result[:n]


def _load_tanimoto_scores(smiles: pd.Series, ref_path: str = None) -> dict[str, float] | None:
    """Try to load tanimoto scores from various sources."""
    # Check zinc_vs_chembl_full_QED.csv
    zinc_path = _resolve("data/zinc_vs_chembl_full_QED.csv")
    if os.path.exists(zinc_path):
        try:
            df = pd.read_csv(zinc_path, usecols=["smiles", "tanimoto_sum"])
            return dict(zip(df["smiles"], df["tanimoto_sum"]))
        except Exception:
            pass

    # Check if there's a precomputed file
    if ref_path and os.path.exists(_resolve(ref_path)):
        try:
            df = pd.read_csv(_resolve(ref_path))
            col = _smiles_column(df)
            # Look for a tanimoto-like column
            for c in df.columns:
                if "tanimoto" in c.lower() or "distance" in c.lower():
                    return dict(zip(df[col], df[c]))
        except Exception:
            pass

    return None


def strategy_most_non_druglike(smiles: pd.Series, n: int, seed: int = SEED) -> list[str]:
    """S4: Больше всего нарушений drug-like правил (Lipinski + QED).

    Считаем количество нарушений для каждой молекулы, берём те,
    что нарушают больше всего правил.
    """
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

    scored = []
    for smi in unique:
        v = count_violations(smi)
        scored.append((v, smi))
        if len(scored) % 50_000 == 0:
            print(f"    Scored {len(scored):,} molecules for non-druglikeness...")

    # Sort by violations (desc), then random within same violation count
    scored.sort(key=lambda x: -x[0])

    # Group by violation count
    by_violations = defaultdict(list)
    v, s = scored[0][0], scored[0][1] if scored else (0, "")
    for v, s in scored:
        by_violations[v].append(s)

    result = []
    for v in sorted(by_violations.keys(), reverse=True):
        mols = by_violations[v]
        rng.shuffle(mols)
        result.extend(mols)
        if len(result) >= n:
            break

    return result[:n]


def strategy_high_predicted(smiles: pd.Series, n: int, seed: int = SEED) -> list[str]:
    """S5: Высокий предсказанный pIC50.

    Ожидает столбец 'predicted_pIC50' в исходном CSV или
    файл с предвычисленными предсказаниями.
    """
    rng = random.Random(seed)
    unique = list(smiles.unique())

    # Try to find predicted pIC50 scores
    pred_scores = _load_predicted_scores(smiles)

    if pred_scores is not None:
        scored = [(pred_scores.get(s, 0.0), s) for s in unique]
        scored.sort(key=lambda x: -x[0])
        result = [s for _, s in scored[:n]]
    else:
        print("    WARNING: No predicted pIC50 scores found, falling back to random")
        return strategy_random(pd.Series(unique), n, seed)

    return result[:n]


def _load_predicted_scores(smiles: pd.Series) -> dict[str, float] | None:
    """Try to load predicted pIC50 scores from precomputed files."""
    # Check for a file like *_predicted.csv
    candidates = [
        "data/zinc_predicted.csv",
        "data/coconut_predicted.csv",
        "data/hmdb_predicted.csv",
    ]
    for path in candidates:
        rp = _resolve(path)
        if os.path.exists(rp):
            try:
                df = pd.read_csv(rp)
                col = _smiles_column(df)
                pred_col = next((c for c in df.columns if "pic50" in c.lower() or "predict" in c.lower()), None)
                if pred_col:
                    return dict(zip(df[col], df[pred_col]))
            except Exception:
                pass
    return None


def strategy_by_chemical_class(smiles: pd.Series, n: int,
                                full_df: pd.DataFrame = None,
                                seed: int = SEED) -> list[str]:
    """S6: Равномерно из каждого chemical_super_class.

    Работает для COCONUT (есть chemical_super_class) и HMDB.
    Для датасетов без классов — fallback на random.
    """
    rng = random.Random(seed)
    unique = list(smiles.unique())

    if full_df is None:
        print("    WARNING: No full DataFrame for chemical_class, falling back to random")
        return strategy_random(pd.Series(unique), n, seed)

    # Find the class column
    class_col = None
    for c in ["chemical_super_class", "chemical_class", "chemical_sub_class",
               "np_classifier_superclass", "direct_parent_classification"]:
        if c in full_df.columns:
            class_col = c
            break

    if class_col is None:
        print(f"    WARNING: No class column found in {full_df.columns.tolist()}, falling back to random")
        return strategy_random(pd.Series(unique), n, seed)

    # Map smiles to class
    smiles_col = _smiles_column(full_df)
    df_valid = full_df.dropna(subset=[smiles_col, class_col])

    # Canonicalize and map
    class_groups = defaultdict(list)
    for _, row in df_valid.iterrows():
        smi = row[smiles_col].strip().split()[0]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        cls = str(row[class_col])
        class_groups[cls].append(canonical)

    # Sample uniformly from classes
    n_per_class = max(1, n // len(class_groups)) if class_groups else n
    result = []
    for cls, mols in class_groups.items():
        rng.shuffle(mols)
        result.extend(mols[:n_per_class])

    # Fill remaining if needed
    if len(result) < n:
        remaining = [s for s in unique if s not in set(result)]
        rng.shuffle(remaining)
        result.extend(remaining)

    return result[:n]


# ===========================================================================
# Диспетчер стратегий
# ===========================================================================

STRATEGIES = {
    "random": strategy_random,
    "scaffold_first": strategy_scaffold_first,
    "property_coverage": strategy_property_coverage,
    "farthest_from_ref": strategy_farthest_from_ref,
    "most_non_druglike": strategy_most_non_druglike,
    "high_predicted": strategy_high_predicted,
    "by_chemical_class": strategy_by_chemical_class,
}


def select(smiles: pd.Series, n: int, strategy: str,
           full_df: pd.DataFrame = None, ref_path: str = None) -> list[str]:
    """Выбрать n молекул из датасета по заданной стратегии."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(STRATEGIES.keys())}")

    print(f"  Strategy: {strategy}, target: {n:,} molecules")
    fn = STRATEGIES[strategy]

    if strategy == "by_chemical_class":
        return fn(smiles, n, full_df=full_df)
    elif strategy == "farthest_from_ref":
        return fn(smiles, n, ref_path=ref_path)
    else:
        return fn(smiles, n)


# ===========================================================================
# Основной пайплайн
# ===========================================================================

def make_mixture(dry_run: bool = False) -> Path:
    """Создать смесь pretrain данных."""
    total_config = sum(cfg[0] for cfg in MIXTURE_CONFIG.values())
    assert total_config == TOTAL_SIZE, (
        f"Sum of counts ({total_config}) != TOTAL_SIZE ({TOTAL_SIZE})"
    )

    output_path = Path("mixed_pretrain.csv")
    pretrain_set_path = Path("pretrain_smiles.txt")
    # Output goes to experiment dir (same dir as this script)
    # Input paths are resolved against PROJECT_ROOT

    # Check cache — если конфиг не изменился, пропускаем
    if _cache_is_valid() and output_path.exists() and not dry_run:
        print(f"Config hash {_config_hash()} matches cache. Skipping mixture generation.")
        return output_path

    all_smiles = []

    if dry_run:
        print(f"[DRY RUN] Would write {output_path} and {pretrain_set_path}")
        for data_path, (count, strategy) in MIXTURE_CONFIG.items():
            print(f"  {data_path}: {count:,} molecules, strategy={strategy}")
        return output_path

    for data_path, (count, strategy) in MIXTURE_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"Processing {data_path} ({count:,} molecules, strategy={strategy})")

        t0 = time.time()

        # Load full DataFrame for strategies that need extra columns
        full_df = None
        if strategy in ("by_chemical_class",):
            full_df = load_full_df(data_path)
            smiles = load_smiles(data_path)
        else:
            smiles = load_smiles(data_path)

        selected = select(smiles, count, strategy, full_df=full_df,
                          ref_path=CHEMBL_REF_PATH)
        all_smiles.extend(selected)

        elapsed = time.time() - t0
        print(f"  Selected {len(selected):,} molecules in {elapsed:.0f}s")

    # Deduplicate
    seen = set()
    unique_smiles = []
    for s in all_smiles:
        if s not in seen:
            seen.add(s)
            unique_smiles.append(s)

    print(f"\n{'='*60}")
    print(f"Total: {len(all_smiles):,} → {len(unique_smiles):,} unique")

    # Write mixed CSV
    output_path.write_text("smiles\n" + "\n".join(unique_smiles) + "\n")
    print(f"Written {output_path} ({len(unique_smiles):,} molecules)")

    # Write pretrain set for novelty calculation
    pretrain_set_path.write_text("\n".join(unique_smiles) + "\n")
    print(f"Written {pretrain_set_path}")

    # Save config hash for caching
    _save_cache()
    print(f"Config hash {_config_hash()} saved to cache.")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create pretraining data mixture")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show composition without writing files")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Mixture config: TOTAL_SIZE={TOTAL_SIZE:,}, SEED={SEED}")
    print(f"Sources: {len(MIXTURE_CONFIG)}")
    for path, (count, strategy) in MIXTURE_CONFIG.items():
        print(f"  {path}: {count:,} molecules, strategy={strategy}")
    print()

    make_mixture(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
