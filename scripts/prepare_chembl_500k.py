"""
prepare_chembl_500k.py — детерминированная подготовка датасета ChEMBL на 500к молекул.

Полная воспроизводимость:
  - фиксированный seed для сэмплинга
  - RDKit-канонизация (убирает неэквивалентные SMILES-разметки одной молекулы)
  - дедупликация по каноническому SMILES
  - детерминированный shuffle (numpy.random.Generator with PCG64)
  - SHA-256 чексумма результата для верификации

Использование:
    uv run scripts/prepare_chembl_500k.py \
        --input data/chembl.csv \
        --output data/chembl_500k.csv \
        --n 500000 \
        --seed 42
"""

import argparse
import csv
import hashlib
import os
import sys

import numpy as np
from rdkit import Chem


def sha256_file(path: str) -> str:
    """Compute SHA-256 of a file (1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonicalize(smi: str):
    """Return canonical SMILES or None if invalid."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def prepare(input_path: str, output_path: str, n_target: int, seed: int,
            max_smiles_len: int = 128):
    rng = np.random.default_rng(seed)

    # --- Step 1: read, canonicalize, deduplicate ---
    print(f"[1/3] Чтение и канонизация: {input_path}")

    seen = set()
    unique_smiles = []
    total = 0
    invalid = 0
    duplicate = 0
    too_long = 0

    with open(input_path, newline="") as fin:
        reader = csv.DictReader(fin)
        col = next(c for c in reader.fieldnames if "smiles" in c.lower())

        for row in reader:
            total += 1
            raw = row[col].strip().split()[0]  # first token only (no salts/mixtures)

            canon = canonicalize(raw)
            if canon is None:
                invalid += 1
                continue

            if len(canon) > max_smiles_len:
                too_long += 1
                continue

            if canon in seen:
                duplicate += 1
                continue

            seen.add(canon)
            unique_smiles.append(canon)

            if total % 500_000 == 0:
                print(f"      {total:,} прочитано → {len(unique_smiles):,} уникальных")

    print(f"      Итого: {total:,} сырых → {len(unique_smiles):,} уникальных")
    print(f"      Невалидных: {invalid:,}, дубликатов: {duplicate:,}, слишком длинных: {too_long:,}")

    if len(unique_smiles) < n_target:
        print(f"\nОШИБКА: только {len(unique_smiles):,} уникальных молекул, нужно {n_target:,}.")
        print("Увеличьте источник или уменьшите --n.")
        sys.exit(1)

    # --- Step 2: deterministic shuffle and sample ---
    print(f"\n[2/3] Детерминированный shuffle (seed={seed}) и выборка {n_target:,}...")
    indices = rng.choice(len(unique_smiles), size=n_target, replace=False)
    sampled = [unique_smiles[i] for i in indices]

    # --- Step 3: write output ---
    print(f"\n[3/3] Запись: {output_path}")
    with open(output_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["smiles"])
        writer.writerows(sampled)

    # --- Verify ---
    with open(output_path, newline="") as fcheck:
        lines = fcheck.readlines()
    assert len(lines) == n_target + 1, f"Ожидалось {n_target + 1} строк, получено {len(lines)}"

    checksum = sha256_file(output_path)
    print(f"\nРезультат:")
    print(f"  Файл:     {output_path}")
    print(f"  Молекул:  {n_target:,}")
    print(f"  Seed:     {seed}")
    print(f"  Max len:  {max_smiles_len}")
    print(f"  SHA-256:  {checksum}")
    print(f"  Размер:   {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка воспроизводимого датасета ChEMBL на N молекул"
    )
    parser.add_argument("--input",  required=True, help="Сырой CSV со столбцом SMILES")
    parser.add_argument("--output", required=True, help="Выходной CSV с каноническими SMILES")
    parser.add_argument("--n",      type=int, default=500_000, help="Количество молекул (default: 500000)")
    parser.add_argument("--seed",   type=int, default=42, help="Seed для воспроизводимости (default: 42)")
    parser.add_argument("--max-len", type=int, default=128, help="Макс. длина канонического SMILES (default: 128)")
    args = parser.parse_args()

    print(f"Параметры:")
    print(f"  Input:   {args.input}")
    print(f"  Output:  {args.output}")
    print(f"  N:       {args.n:,}")
    print(f"  Seed:    {args.seed}")
    print(f"  Max len: {args.max_len}")
    print()

    prepare(args.input, args.output, args.n, args.seed, args.max_len)


if __name__ == "__main__":
    main()
