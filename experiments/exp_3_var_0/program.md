# Experiment 3 var 0: Non-drug-like pretraining diversity → SYK adaptability

## Hypothesis

Adding non-drug-like molecular structures to pretraining data improves
predicted pIC50 after fine-tuning on SYK inhibitors, because structural
diversity increases the model's adaptability to the target space.

## Setup

To set up a new run:

1. **Confirm branch**: You are on branch `run_exp_3_var_0`. Do NOT create a new branch.
2. **Read the in-scope files**:
   - `common/train.py` — two-phase training (pretrain + fine-tune). Do NOT modify.
   - `common/prepare_data.py` — data filtering. Do NOT modify.
   - `common/breakit_tokenizer.py` — SMILES tokenizer. Do NOT modify.
   - `experiments/exp_3_var_0/mixture.py` — THE ONLY FILE YOU MODIFY. Data mixture configuration.
3. **Verify data exists on cluster**: All runs execute on the cluster. Verify:
   ```bash
   ssh aichem "ls -lh /mnt/tank/scratch/aergardt/molecular_autoresearch/data/"
   ```
   Expected files: `chembl.csv`, `coconut_smiles.csv`, `hmdb_smiles.csv`,
   `zinc_vs_chembl_full_QED.csv`, `syk_inhibitors.csv`, `stacking_regressor.joblib`.
3. **Prepare SYK active subset**: Filter `syk_inhibitors.csv` to keep only pIC50 > 7
   (~1900 molecules). Save as `data/syk_active.csv` on the cluster.
4. **Initialize results.tsv**: Create `experiments/exp_3_var_0/results.tsv` with header row.
5. **Confirm and go**.

## What You Can Do

- **Modify `mixture.py`** — change `MIXTURE_CONFIG` to try different data mixtures and selection strategies.
- That's it. The model architecture, hyperparameters, fine-tuning procedure, and evaluation are all fixed.

## What You Cannot Do

- Modify any file in `common/`
- Install new packages
- Change model architecture, optimizer, or hyperparameters
- Change the fine-tuning procedure (LR=1e-4, batch=64, epochs=10)
- Change the evaluation metric

## The Goal

**Maximize `score = scaffold_entropy × mean_topK_pIC50`**

Where:
- `scaffold_entropy` — normalized diversity of Murcko scaffolds in generated molecules
- `mean_topK_pIC50` — mean predicted pIC50 among top-50 molecules by predicted activity

This metric balances **diversity** (many scaffold types) and **target relevance** (high predicted activity). A model that generates 5000 copies of one molecule scores 0 (no entropy). A model that generates diverse garbage scores 0 (no pIC50).

## Experiment Loop

Each experiment has TWO phases. All commands run on the cluster via SSH.

### Phase 1: Create data mixture
```bash
ssh aichem "cd /mnt/tank/scratch/aergardt/molecular_autoresearch/experiments/exp_3_var_0 && \
    python mixture.py"                     # creates mixed_pretrain.csv
ssh aichem "cd /mnt/tank/scratch/aergardt/molecular_autoresearch/experiments/exp_3_var_0 && \
    python mixture.py --dry-run"           # preview without computing
```

The mixture is cached by config hash. If `MIXTURE_CONFIG` hasn't changed,
the script skips and uses the cached `mixed_pretrain.csv`.

### Phase 2: Pretrain + Fine-tune + Evaluate
```bash
# Pretrain (no evaluation during training)
ssh aichem "cd /mnt/tank/scratch/aergardt/molecular_autoresearch && \
    uv run common/train.py \
    --data experiments/exp_3_var_0/mixed_pretrain.csv \
    --epochs 5 \
    --save-final experiments/exp_3_var_0/checkpoints/exp_N"

# Fine-tune on active SYK inhibitors (pIC50 > 7) + evaluate
ssh aichem "cd /mnt/tank/scratch/aergardt/molecular_autoresearch && \
    uv run common/train.py \
    --finetune \
    --load experiments/exp_3_var_0/checkpoints/exp_N \
    --data data/syk_active.csv \
    --epochs 10 \
    --pretrain-set experiments/exp_3_var_0/pretrain_smiles.txt \
    --save experiments/exp_3_var_0/checkpoints/exp_N_ft"
```

The final `score` is the best score achieved during fine-tuning (saved in `metrics.json`).

### Mixture Configuration

Edit `MIXTURE_CONFIG` in `mixture.py`:

```python
TOTAL_SIZE = 500_000  # fixed

MIXTURE_CONFIG = {
    "data/chembl.csv": (400000, "random"),
    "data/coconut_smiles.csv": (100000, "scaffold_first"),
}
```

Sum of all counts must equal `TOTAL_SIZE`.

### Available Strategies

| Strategy | Description | Best for |
|----------|-------------|----------|
| `random` | Random sample (seed=42) | Baseline, ChEMBL |
| `scaffold_first` | Max unique Murcko scaffolds | COCONUT, HMDB, ZINC |
| `property_coverage` | Uniform (LogP, MW, TPSA) coverage | All |
| `farthest_from_ref` | Farthest from ChEMBL by Tanimoto | ZINC (needs tanimoto_sum) |
| `most_non_druglike` | Most Lipinski/QED violations | ChEMBL non-druglike, ZINC |
| `high_predicted` | High predicted pIC50 | All (needs precomputed predictions) |
| `by_chemical_class` | Uniform from each chemical_super_class | COCONUT |

### Available Datasets

| Path | Size | Description |
|------|------|-------------|
| `data/chembl.csv` | 2.8M | Drug-like molecules |
| `data/coconut_smiles.csv` | 738K | Natural products |
| `data/hmdb_smiles.csv` | 114K | Metabolites |
| `data/zinc_vs_chembl_full_QED.csv` | 7.5M | ZINC with tanimoto_sum |

## Logging Results

After each experiment, log to `experiments/exp_3_var_0/results.tsv`:

```
commit	score	mean_topK_pIC50	scaffold_entropy	topK_scaffolds	validity	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `score` from metrics.json (scaffold_entropy × mean_topK_pIC50) — use 0.000000 for crashes
3. `mean_topK_pIC50` from metrics.json
4. `scaffold_entropy` from metrics.json
5. `topK_scaffolds` from metrics.json
6. `validity` from metrics.json
7. peak memory in GB (round to .1f) — use 0.0 for crashes
8. status: `keep`, `discard`, or `crash`
9. short description of the mixture

Example:
```
commit	score	mean_topK_pIC50	scaffold_entropy	topK_scaffolds	validity	memory_gb	status	description
a1b2c3d	0.000000	0.000	0.000	0	0.000	0.0	crash	baseline (OOM)
b2c3d4e	2.345678	7.821	0.299	42	0.951	28.3	keep	500K ChEMBL random baseline
c3d4e5f	2.567890	7.901	0.325	38	0.934	28.5	keep	400K ChEMBL + 100K COCONUT scaffold_first
d4e5f6g	2.123456	7.750	0.274	35	0.962	28.1	discard	300K ChEMBL + 200K HMDB property_coverage
```

## The Experiment Loop

LOOP FOREVER:

1. Look at `results.tsv` — what's the current best score?
2. Look at `mixture.py` — what's the current mixture?
3. Change `MIXTURE_CONFIG` in `mixture.py` with a new hypothesis about what mixture will improve score.
4. git commit the mixture.py change
5. Run the experiment (mixture + pretrain + fine-tune)
6. Extract the final score from the fine-tune checkpoint's `metrics.json`
7. Record in results.tsv
8. If score improved → keep the commit
9. If score equal or worse → git reset back

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Run until manually interrupted. If you run out of ideas, try more radical mixtures — 100% non-druglike, extreme strategies, combinations of 3+ sources.

## Key Principles

- **The hypothesis is about diversity → adaptability**. Your job is to find the mixture that maximizes this.
- **Start with baseline**: First experiment = 500K random ChEMBL. This establishes the reference point.
- **One variable at a time**: Change one source or strategy per experiment to attribute improvements.
- **Simplicity**: A 5% improvement that adds 3 dataset sources is worth it. A 0.1% improvement that requires 5 sources is not.
- **The score combines diversity AND activity**. Don't optimize one at the expense of the other.
