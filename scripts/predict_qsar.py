"""
predict_qsar.py — generate molecules from fine-tuned checkpoints and predict pIC50.

Usage on remote:
    uv run python predict_qsar.py --checkpoint_dir checkpoints/ --output predictions.tsv --model data/stacking_regressor.joblib
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

import joblib


def morgan_fp_bits(mol, radius=2, nbits=2048):
    """Return a 2048-bit Morgan fingerprint as a list of ints."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return [int(x) for x in fp]


def predict_pIC50(smiles_list, model):
    """Predict pIC50 for a list of SMILES using the QSAR model."""
    valid = []
    preds = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            bits = morgan_fp_bits(mol)
            valid.append(smi)
            preds.append(bits)

    if not preds:
        return pd.DataFrame(columns=["smiles", "pIC50"])

    X = pd.DataFrame(preds, columns=[f"fingerprint_{i}" for i in range(2048)])
    y_pred = model.predict(X)
    return pd.DataFrame({"smiles": valid, "pIC50": y_pred})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Root checkpoints directory")
    parser.add_argument("--output", default="predictions.tsv", help="Output TSV file")
    parser.add_argument("--model", required=True, help="Path to stacking_regressor.joblib")
    parser.add_argument("--gen", type=int, default=5000, help="Molecules to generate per model")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--models", nargs="+", default=None, help="Specific model names (default: all syk_*)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load QSAR model
    print(f"Loading QSAR model: {args.model}")
    model = joblib.load(args.model)

    # Import GPT model dynamically (matches train.py)
    sys.path.insert(0, str(Path(__file__).parent))
    from fine_tune import GPT, SmilesTokenizer

    # Find checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    if args.models:
        ckpt_names = args.models
    else:
        ckpt_names = sorted([d.name for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("syk_")])

    print(f"Processing {len(ckpt_names)} checkpoints: {ckpt_names}")

    all_results = []

    for name in ckpt_names:
        ckpt_path = ckpt_dir / name
        print(f"\n{'='*60}")
        print(f"Loading: {name}")

        try:
            gpt, tok = GPT.load(str(ckpt_path), device)
        except Exception as e:
            print(f"ERROR loading {name}: {e}")
            continue

        # Generate
        random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

        seqs = gpt.generate(args.gen, tok.BOS, tok.EOS, args.temp)
        decoded = [tok.decode(s) for s in seqs]
        print(f"Generated {len(decoded)} molecules")

        # Predict
        df_pred = predict_pIC50(decoded, model)
        df_pred["model"] = name

        if len(df_pred) > 0:
            print(f"Valid molecules: {len(df_pred)}")
            print(f"Mean pIC50: {df_pred['pIC50'].mean():.3f}")
            print(f"Max pIC50: {df_pred['pIC50'].max():.3f}")
            print(f"Min pIC50: {df_pred['pIC50'].min():.3f}")
            print(f"Median pIC50: {df_pred['pIC50'].median():.3f}")
            all_results.append(df_pred)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, sep="\t", index=False)
        print(f"\n{'='*60}")
        print(f"Saved {len(combined)} predictions to {args.output}")

        # Summary per model
        summary = combined.groupby("model")["pIC50"].agg(["mean", "median", "max", "min", "count"])
        summary = summary.sort_values("mean", ascending=False)
        print("\nPer-model summary:")
        print(summary.to_string())
    else:
        print("No valid predictions generated!")


if __name__ == "__main__":
    main()
