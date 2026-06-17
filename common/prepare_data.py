"""
prepare_data.py — фильтрация и канонизация SMILES для обучения.

Запускается один раз перед обучением. Принимает сырой CSV,
фильтрует молекулы по заданным критериям и сохраняет чистый CSV.

Использование:
    python prepare_data.py --input data/chembl_500k.csv --output data/chembl_clean.csv
    python prepare_data.py --input data/zinc_nondrug.csv --output data/nondrug_clean.csv --no-druglike
"""

import argparse
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def passes_druglike(mol, mw_max=500, logp_max=5, hbd_max=5, hba_max=10, qed_min=0.3) -> bool:
    """Стандартный drug-like фильтр: расширенное правило Липинского + QED."""
    if Descriptors.MolWt(mol) > mw_max:
        return False
    if Descriptors.MolLogP(mol) > logp_max:
        return False
    if Descriptors.NumHDonors(mol) > hbd_max:
        return False
    if Descriptors.NumHAcceptors(mol) > hba_max:
        return False
    if QED.qed(mol) < qed_min:
        return False
    return True


def prepare(input_path: str, output_path: str, druglike: bool, max_smiles_len: int):
    total = valid = filtered = duplicate = 0
    seen = set()

    with open(input_path) as fin, open(output_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        col = next(c for c in reader.fieldnames if "smiles" in c.lower())
        writer = csv.writer(fout)
        writer.writerow(["smiles"])

        for row in reader:
            total += 1
            smi = row[col].strip().split()[0]

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                filtered += 1
                continue
            valid += 1

            if druglike and not passes_druglike(mol):
                filtered += 1
                continue

            canonical = Chem.MolToSmiles(mol)

            if len(canonical) > max_smiles_len:
                filtered += 1
                continue

            if canonical in seen:
                duplicate += 1
                continue

            seen.add(canonical)
            writer.writerow([canonical])

            if total % 100_000 == 0:
                print(f"  {total:,} processed → {len(seen):,} kept...")

    print(f"\nДатасет готов: {output_path}")
    print(f"  Всего:      {total:,}")
    print(f"  Валидных:   {valid:,}")
    print(f"  Отфильтр.:  {filtered:,}")
    print(f"  Дубликаты:  {duplicate:,}")
    print(f"  Итого:      {len(seen):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",          required=True,        help="Входной CSV со столбцом smiles")
    parser.add_argument("--output",         required=True,        help="Выходной CSV")
    parser.add_argument("--no-druglike",    action="store_true",  help="Отключить drug-like фильтр")
    parser.add_argument("--max-len",        type=int, default=128, help="Макс. длина SMILES (символов)")
    args = parser.parse_args()

    druglike = not args.no_druglike
    print(f"Фильтрация: {'drug-like (MW≤500, logP≤5, HBD≤5, HBA≤10, QED≥0.3)' if druglike else 'только валидность'}")
    print(f"Макс. длина SMILES: {args.max_len}")
    print()

    prepare(args.input, args.output, druglike, args.max_len)


if __name__ == "__main__":
    main()