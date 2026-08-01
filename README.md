# Autonomous Discovery of Training Distributions for Controllable Molecular Generation

This repository contains the codebase for autonomous research experiments on training data distribution design for molecular language models.

The central research question:

Can we automatically discover optimal pretraining data mixtures that improve downstream controllability (predicted pIC50) after fine‑tuning?

The system autonomously modifies pretraining data composition, trains a SMILES GPT model, fine‑tunes it on a target task (SYK inhibitors), evaluates diversity and activity, and iteratively improves the data mixture.

---

# Core Hypothesis

Adding non‑drug‑like molecular structures to pretraining increases structural diversity, which improves model adaptability to a downstream target (SYK inhibitors), measured by:

```
score = scaffold_entropy × mean_topK_pIC50
```

Where:

- scaffold_entropy — normalized Murcko scaffold diversity of generated molecules  
- mean_topK_pIC50 — mean predicted pIC50 of top‑50 generated molecules  

The objective balances diversity and activity.

---

# Repository Structure

```
molecular_autoresearch_exp_3_var_0/
│
├── README.md
├── program_template.md
│
├── common/
│   ├── train.py
│   ├── prepare_data.py
│   ├── breakit_tokenizer.py
│   └── pyproject.toml
│
└── experiments/
    └── exp_3_var_0/
        ├── program.md
        ├── mixture.py
        └── results.tsv
```

---

# common/

Core implementation. These files are never modified during experiments.

## train.py

Two‑phase training pipeline:
- Pretraining  
- Fine‑tuning  
- Evaluation  
- Generation  
- Metric computation  

Implements:
- Character‑level SMILES GPT  
- RoPE attention  
- Scaffold entropy metric  
- Internal diversity  
- Predicted pIC50 via stacking regressor  

## prepare_data.py

SMILES filtering and canonicalization. Supports:
- Drug‑like filtering (Lipinski + QED)
- Non‑druglike selection
- Violation‑based filtering

## breakit_tokenizer.py

Fixed SMILES tokenizer and vocabulary.

---

# experiments/

## mixture.py

This is the only file modified during autonomous search.

Defines:

```python
TOTAL_SIZE = 500_000

MIXTURE_CONFIG = {
    "data/chembl.csv": (400000, "random"),
    "data/coconut_smiles.csv": (100000, "scaffold_first"),
}
```

Controls:
- Which datasets are used  
- How many molecules from each  
- Selection strategy  

The mixture is cached via configuration hash.

## results.tsv

Experiment log with columns:

```
commit	score	mean_topK_pIC50	scaffold_entropy	topK_scaffolds	validity	memory_gb	status	description
```

---

# Environment Setup

This project uses `uv`.

Install uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies:

```bash
uv sync
```

Requirements:
- Python 3.10+
- PyTorch
- RDKit
- NumPy
- pandas

---

# Required Datasets

On cluster path:

```
/mnt/tank/scratch/<USER>/molecular_autoresearch/data/
```

Expected files:

- chembl.csv  
- coconut_smiles.csv  
- hmdb_smiles.csv  
- zinc_vs_chembl_full_QED.csv  
- syk_inhibitors.csv  
- stacking_regressor.joblib  

---

# Running Experiment 3 var 0

All experiments run on the cluster.

## Step 1 — Confirm Branch

```bash
git checkout run_exp_3_var_0
```

## Step 2 — Prepare SYK Active Subset

Filter inhibitors with:

```
pIC50 > 7
```

Save as:

```
data/syk_active.csv
```

## Step 3 — Initialize Results Log

Create:

```
experiments/exp_3_var_0/results.tsv
```

With header:

```
commit	score	mean_topK_pIC50	scaffold_entropy	topK_scaffolds	validity	memory_gb	status	description
```

---

# Experiment Loop

## Phase 1 — Create Mixture

```bash
ssh aichem "
cd /mnt/tank/scratch/<USER>/molecular_autoresearch/experiments/exp_3_var_0 &&
python mixture.py
"
```

Preview only:

```bash
python mixture.py --dry-run
```

## Phase 2 — Pretrain

```bash
ssh aichem "
cd /mnt/tank/scratch/<USER>/molecular_autoresearch &&
uv run common/train.py \
  --data experiments/exp_3_var_0/mixed_pretrain.csv \
  --epochs 5 \
  --save-final experiments/exp_3_var_0/checkpoints/exp_N
"
```

## Phase 3 — Fine‑Tune + Evaluate

```bash
ssh aichem "
cd /mnt/tank/scratch/<USER>/molecular_autoresearch &&
uv run common/train.py \
  --finetune \
  --load experiments/exp_3_var_0/checkpoints/exp_N \
  --data data/syk_active.csv \
  --epochs 10 \
  --pretrain-set experiments/exp_3_var_0/pretrain_smiles.txt \
  --save experiments/exp_3_var_0/checkpoints/exp_N_ft
"
```

Final metrics are saved in:

```
metrics.json
```

---

# Evaluation Objective

```
score = scaffold_entropy × mean_topK_pIC50
```

Higher is better.

---

# Restarting From Scratch

Delete:

```
experiments/exp_3_var_0/checkpoints/
.mixture_cache/
```

Reset `results.tsv`.

Set baseline mixture:

```python
MIXTURE_CONFIG = {
    "data/chembl.csv": (500000, "random"),
}
```

Start experiment loop again.
