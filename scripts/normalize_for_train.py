"""
normalize_for_train.py — привести все CSV-файлы на кластере к формату,
который читает train.py: один столбец 'smiles' (ZINC — плюс метрики).

Форматы на входе:
  - chembl_500k_seed0.csv: comma-per-character (каждый символ SMILES в отдельной ячейке)
  - coconut.csv: многоколоночный, SMILES в 'canonical_smiles'
  - hmdb.csv: многоколоночный, SMILES в 'SMILES'
  - zinc_vs_chembl_full_QED.csv: многоколоночный, SMILES в 'mol1_smiles' + метрики

Формат на выходе:
  - chembl_500k_seed0.csv: 'smiles' (одна строка SMILES в одной ячейке)
  - coconut_smiles.csv: 'smiles'
  - hmdb_smiles.csv: 'smiles'
  - zinc_vs_chembl_full_QED.csv: 'smiles,tanimoto_sum,scaffold_sum,QED'

Использование:
    python normalize_for_train.py --dir /path/to/data
"""

import argparse
import csv
import sys
import os

# COCONUT has very long fields (InChI, names) — exceed default 131072
# Python 3.12+: csv.fieldsize_limit(); Python 3.11: csv.field_size_limit()
try:
    csv.fieldsize_limit(max(csv.fieldsize_limit(), 10**7))
except (AttributeError, TypeError):
    csv.field_size_limit(10**7)


def detect_comma_per_char(path: str) -> bool:
    """Check if file uses comma-per-character format."""
    with open(path) as f:
        header = f.readline().strip()
        if header == "smiles":
            first_line = f.readline().strip()
            # comma-per-char: each char separated by comma, no quotes around full SMILES
            # First field should be a single character
            first_field = first_line.split(",")[0]
            return len(first_field) == 1


def convert_comma_per_char(input_path: str, output_path: str):
    """Convert comma-per-character CSV to single-column smiles CSV."""
    written = 0
    with open(input_path) as fin, open(output_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["smiles"])

        # Skip header
        fin.readline()

        for line in fin:
            line = line.strip()
            if not line:
                continue
            # Join all fields (each is one character)
            smiles = "".join(line.split(","))
            writer.writerow([smiles])
            written += 1
            if written % 100_000 == 0:
                print(f"  {written:,} written...")

    print(f"  Done: {written:,} molecules -> {output_path}")


def extract_smiles(input_path: str, output_path: str, smiles_col: str):
    """Extract SMILES column from multi-column CSV."""
    written = 0
    with open(input_path, newline="") as fin, open(output_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout)
        writer.writerow(["smiles"])

        for row in reader:
            smi = row.get(smiles_col, "").strip()
            if smi:
                writer.writerow([smi])
                written += 1
                if written % 100_000 == 0:
                    print(f"  {written:,} written...")

    print(f"  Done: {written:,} molecules -> {output_path}")


def normalize_zinc(input_path: str, output_path: str):
    """Normalize ZINC: rename mol1_smiles -> smiles, keep metrics."""
    written = 0
    with open(input_path, newline="") as fin, open(output_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        # Handle possible BOM or empty first field
        fieldnames = [f for f in reader.fieldnames if f]
        smiles_field = None
        for f in fieldnames:
            if "mol1_smiles" in f.lower() or "smiles" in f.lower():
                smiles_field = f
                break

        if not smiles_field:
            print(f"  ERROR: no SMILES column found in {input_path}")
            print(f"  Available columns: {fieldnames}")
            sys.exit(1)

        writer = csv.writer(fout)
        writer.writerow(["smiles", "tanimoto_sum", "scaffold_sum", "QED"])

        for row in reader:
            smi = row.get(smiles_field, "").strip()
            tanimoto = str(row.get("tanimoto_sum") or "")
            scaffold = str(row.get("scaffold_sum") or "")
            qed = str(row.get("QED") or "")

            if smi:
                writer.writerow([smi, tanimoto, scaffold, qed])
                written += 1
                if written % 500_000 == 0:
                    print(f"  {written:,} written...")

    print(f"  Done: {written:,} molecules -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Data directory path")
    args = parser.parse_args()

    d = args.dir

    # 1. ChEMBL seed0 — comma-per-character -> single column
    chembl_in = os.path.join(d, "chembl_500k_seed0.csv")
    if os.path.exists(chembl_in):
        chembl_tmp = os.path.join(d, "chembl_500k_seed0.csv.tmp")
        if detect_comma_per_char(chembl_in):
            print("[1/4] Converting chembl_500k_seed0.csv (comma-per-char -> single column)...")
            convert_comma_per_char(chembl_in, chembl_tmp)
            os.replace(chembl_tmp, chembl_in)
            print(f"  Replaced {chembl_in}")
        else:
            print("[1/4] chembl_500k_seed0.csv already in correct format, skipping.")
    else:
        print(f"[1/4] {chembl_in} not found, skipping.")

    # 2. COCONUT — extract canonical_smiles
    coconut_in = os.path.join(d, "coconut.csv")
    coconut_out = os.path.join(d, "coconut_smiles.csv")
    if os.path.exists(coconut_in):
        print(f"\n[2/4] Extracting SMILES from coconut.csv -> coconut_smiles.csv...")
        coconut_tmp = coconut_out + ".tmp"
        extract_smiles(coconut_in, coconut_tmp, "canonical_smiles")
        os.replace(coconut_tmp, coconut_out)
        print(f"  Saved {coconut_out}")
    else:
        print(f"[2/4] {coconut_in} not found, skipping.")

    # 3. HMDB — extract SMILES
    hmdb_in = os.path.join(d, "hmdb.csv")
    hmdb_out = os.path.join(d, "hmdb_smiles.csv")
    if os.path.exists(hmdb_in):
        print(f"\n[3/4] Extracting SMILES from hmdb.csv -> hmdb_smiles.csv...")
        hmdb_tmp = hmdb_out + ".tmp"
        extract_smiles(hmdb_in, hmdb_tmp, "SMILES")
        os.replace(hmdb_tmp, hmdb_out)
        print(f"  Saved {hmdb_out}")
    else:
        print(f"[3/4] {hmdb_in} not found, skipping.")

    # 4. ZINC — normalize column names
    zinc_in = os.path.join(d, "zinc_vs_chembl_full_QED.csv")
    if os.path.exists(zinc_in):
        zinc_tmp = zinc_in + ".tmp"
        print(f"\n[4/4] Normalizing zinc_vs_chembl_full_QED.csv...")
        normalize_zinc(zinc_in, zinc_tmp)
        os.replace(zinc_tmp, zinc_in)
        print(f"  Replaced {zinc_in}")
    else:
        print(f"[4/4] {zinc_in} not found, skipping.")

    print("\nAll done.")


if __name__ == "__main__":
    main()
