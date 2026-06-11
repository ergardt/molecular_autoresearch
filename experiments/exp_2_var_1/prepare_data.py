"""
prepare_data.py — фильтрация и канонизация SMILES для обучения.

Запускается один раз перед обучением. Принимает сырой CSV,
фильтрует молекулы по заданным критериям и сохраняет чистый CSV.

Использование:
    # Drug-like фильтр (по умолчанию)
    python prepare_data.py --input data/chembl_500k.csv --output data/chembl_clean.csv
    # Только валидность, без drug-like
    python prepare_data.py --input data/zinc.csv --output data/zinc_clean.csv --no-druglike
    # Non-druglike: все нарушившие ≥1 правило
    python prepare_data.py --input data/zinc.csv --output data/zinc_nondrug.csv --non-druglike
    # Soft violators: нарушили 1-2 правила
    python prepare_data.py --input data/zinc.csv --output data/zinc_soft.csv --non-druglike --violations-min 1 --violations-max 2
    # Hard violators: нарушили 3+ правил
    python prepare_data.py --input data/zinc.csv --output data/zinc_hard.csv --non-druglike --violations-min 3
"""

import argparse
import csv
csv.field_size_limit(1000000)
from rdkit import Chem
from rdkit.Chem import Descriptors, QED


def count_violations(mol, mw_max=500, logp_max=5, hbd_max=5, hba_max=10, qed_min=0.3) -> int:
    """Count how many drug-like rules a molecule violates (0–5)."""
    violations = 0
    if Descriptors.MolWt(mol) > mw_max:
        violations += 1
    if Descriptors.MolLogP(mol) > logp_max:
        violations += 1
    if Descriptors.NumHDonors(mol) > hbd_max:
        violations += 1
    if Descriptors.NumHAcceptors(mol) > hba_max:
        violations += 1
    if QED.qed(mol) < qed_min:
        violations += 1
    return violations


def passes_druglike(mol, mw_max=500, logp_max=5, hbd_max=5, hba_max=10, qed_min=0.3) -> bool:
    """Стандартный drug-like фильтр: расширенное правило Липинского + QED."""
    return count_violations(mol, mw_max, logp_max, hbd_max, hba_max, qed_min) == 0


def prepare(input_path: str, output_path: str, druglike: bool, max_smiles_len: int,
            non_druglike: bool = False, violations_min: int = 0, violations_max: int = 5):
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

            if non_druglike:
                v = count_violations(mol)
                if v < violations_min or v > violations_max:
                    filtered += 1
                    continue
            elif druglike and not passes_druglike(mol):
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
    parser.add_argument("--input",           required=True,        help="Входной CSV со столбцом smiles")
    parser.add_argument("--output",          required=True,        help="Выходной CSV")
    parser.add_argument("--no-druglike",     action="store_true",  help="Отключить drug-like фильтр")
    parser.add_argument("--non-druglike",    action="store_true",  help="Оставить только non-druglike (инверсия фильтра)")
    parser.add_argument("--violations-min",  type=int, default=1,  help="Мин. кол-во нарушений (для --non-druglike, default=1)")
    parser.add_argument("--violations-max",  type=int, default=5,  help="Макс. кол-во нарушений (для --non-druglike, default=5)")
    parser.add_argument("--max-len",         type=int, default=128, help="Макс. длина SMILES (символов)")
    args = parser.parse_args()

    if args.non_druglike:
        mode = f"non-druglike: {args.violations_min}–{args.violations_max} нарушений"
        druglike = False
    elif args.no_druglike:
        mode = "только валидность"
        druglike = False
    else:
        mode = "drug-like (MW≤500, logP≤5, HBD≤5, HBA≤10, QED≥0.3)"
        druglike = True

    print(f"Фильтрация: {mode}")
    print(f"Макс. длина SMILES: {args.max_len}")
    print()

    prepare(args.input, args.output, druglike, args.max_len,
            non_druglike=args.non_druglike,
            violations_min=args.violations_min,
            violations_max=args.violations_max)


if __name__ == "__main__":
    main()