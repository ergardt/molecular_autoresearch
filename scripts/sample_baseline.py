"""
sample_baseline.py — один проход по ChEMBL, 10 независимых сэмплов по 500к.

Избегает 10 полных проходов по файлу: канонизация и дедупликация
делаются один раз, затем 10 разных random_seed дают 10 сэмплов.

Использование:
    uv run scripts/sample_baseline.py
"""

import csv
import os

import numpy as np
from rdkit import Chem

INPUT = "data/chembl.csv"
OUTPUT_DIR = "data/baseline_samples"
N = 500_000
SEEDS = list(range(10))
MAX_SMILES_LEN = 128


def canonicalize(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: read, canonicalize, deduplicate ---
    print(f"[1/3] Чтение и канонизация: {INPUT}")

    seen = set()
    unique_smiles = []
    total = 0
    invalid = 0
    duplicate = 0
    too_long = 0

    with open(INPUT, newline="") as fin:
        reader = csv.DictReader(fin)
        col = next(c for c in reader.fieldnames if "smiles" in c.lower())

        for row in reader:
            total += 1
            raw = row[col].strip().split()[0]

            canon = canonicalize(raw)
            if canon is None:
                invalid += 1
                continue

            if len(canon) > MAX_SMILES_LEN:
                too_long += 1
                continue

            if canon in seen:
                duplicate += 1
                continue

            seen.add(canon)
            unique_smiles.append(canon)

            if total % 500_000 == 0:
                print(f"      {total:,} прочитано -> {len(unique_smiles):,} уникальных")

    print(f"      Итого: {total:,} сырых -> {len(unique_smiles):,} уникальных")
    print(f"      Невалидных: {invalid:,}, дубликатов: {duplicate:,}, слишком длинных: {too_long:,}")

    if len(unique_smiles) < N:
        print(f"ОШИБКА: только {len(unique_smiles):,} уникальных, нужно {N:,}.")
        raise SystemExit(1)

    # --- Step 2: sample with different seeds ---
    print(f"\n[2/3] Сэмплирование {len(SEEDS)} сэмплов по {N:,} молекул...")

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(unique_smiles), size=N, replace=False)
        sampled = [unique_smiles[i] for i in indices]

        out_path = os.path.join(OUTPUT_DIR, f"chembl_500k_seed{seed}.csv")
        print(f"      seed={seed} -> {out_path}")

        with open(out_path, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["smiles"])
            writer.writerows(sampled)

    # --- Step 3: verify ---
    print(f"\n[3/3] Верификация...")
    for seed in SEEDS:
        out_path = os.path.join(OUTPUT_DIR, f"chembl_500k_seed{seed}.csv")
        with open(out_path, newline="") as f:
            line_count = sum(1 for _ in f) - 1  # minus header
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        assert line_count == N, f"seed={seed}: ожидалось {N}, получено {line_count}"
        print(f"      seed={seed}: {line_count:,} молекул, {size_mb:.1f} MB — OK")

    print(f"\nГотово. {len(SEEDS)} сэмплов сохранены в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
