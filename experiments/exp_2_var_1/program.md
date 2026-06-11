# molgen-autoresearch: non-druglike additive experiment

Autonomous experimentation on SMILES-based molecular generation.

**Hypothesis**: adding a small fraction of non-druglike molecules (molecules failing ≥1 rule of `passes_druglike()`) to the ChEMBL training dataset improves generative model diversity without breaking drug-likeness control metrics.

The model is frozen at N_LAYER=4, N_EMBD=256. Training data = `chembl_clean.csv` (100%) + variable non-druglike additive subset.

---

## Setup

To set up a new experiment, work with the user to:

1. **Branch**: `run_exp_2_var_0` — current working branch.
2. **Verify data exists** on the remote node:
   ```bash
   ssh aichem "ls -lh /mnt/tank/scratch/aergardt/autoresearch/data/"
   ```
   Required files:
   - `chembl_clean.csv` — 329K drug-like ChEMBL molecules (frozen baseline)
   - `zinc_vs_chembl_full_QED.csv` — ZINC molecules with tanimoto_sum, scaffold_sum, QED, MW, logP
   - `coconut_smiles.csv` — COCONUT natural/isolated compounds
   - `hmdb_smiles.csv` — HMDB human metabolites

3. **Initialize results.tsv** locally with a header row.
4. **Confirm and go**.

---

## Experimentation

**Workflow** — hybrid setup:
- **Prepare additive subsets locally**: filter sources by non-druglike criterion, save to CSV
- **Transfer to remote**: `scp <dataset>.csv aichem:/mnt/tank/scratch/aergardt/autoresearch/data/`
- **Run on remote**: execute on GPU cluster node (`aichem`) — **train.py is read-only**
- **Retrieve results**: fetch logs back to analyze

**What you CAN do:**
- Apply the non-druglike filter to any source (ZINC, COCONUT, HMDB)
- Change the training dataset via `--data` argument
- Vary the additive fraction (2%, 5%, 10%, 25%, 50% relative to ChEMBL)
- Select by strategy: random, soft-violators, hard-violators, ZINC-specific (structurally close/far)
- Save checkpoints with `--save <dir>`
- Generate molecules from checkpoints with `--load <dir>`

**What you CANNOT do:**
- Modify `train.py` — model architecture, hyperparameters, and score formula are frozen (N_LAYER=4, N_EMBD=256, EPOCHS=20, LR=3e-4, score = validity × scaffold_entropy × int_div)
- Install new packages beyond `pyproject.toml`

**Decision metric**: `score = validity × scaffold_entropy × int_div` — computed post-hoc from the final epoch metrics. Both diversity components (scaffold_entropy and int_div) are weighted equally.

**Control metrics** (must NOT fall below baseline):
- `validity` ≥ baseline
- `mean_qed` ≥ baseline
- `pct_lipinski` ≥ baseline

**Simplicity criterion**: all else being equal, simpler is better. An interpretable selection criterion beats an opaque weighted combination if results are similar.

---

## Non-druglike filter

A molecule is **non-druglike** if it fails ≥1 rule from `passes_druglike()` in `prepare_data.py`:
- MW > 500
- logP > 5
- HBD > 5
- HBA > 10
- QED < 0.3

To prepare a non-druglike subset from a source, invert the filter:
```python
# Keep molecules that FAIL at least one rule
non_druglike = [mol for mol in source if not passes_druglike(mol)]
```

**Counting violations**: for each non-druglike molecule, count how many of the 5 rules it violates (1–5). This count is used by the soft/hard strategies.

---

## Additive sources

### 1. ZINC — structurally diverse, scored against ChEMBL

`zinc_vs_chembl_full_QED.csv`

| Column | Type | Meaning |
|--------|------|---------|
| smiles | str | canonical SMILES |
| tanimoto_sum | float | sum of Tanimoto similarities (Morgan FP, radius=2) to all ChEMBL molecules |
| scaffold_sum | int | count of Bemis-Murcko scaffold occurrences in ChEMBL |
| qed | float | QED score |
| mw | float | molecular weight |
| logp | float | logP |

### 2. COCONUT — natural and isolated compounds

`coconut_smiles.csv`

### 3. HMDB — human metabolites

`hmdb_smiles.csv`

### 4. CompTox (phase 2, if needed)

EPA CompTox — industrial/ecological chemicals. Download only if phase A shows a positive signal from the three existing sources.

---

## Selection strategies

### Universal (all sources)

| # | Name | Description |
|---|------|-------------|
| 1 | **Random** | Uniform sample from all non-druglike molecules |
| 2 | **Soft violators** | Molecules violating 1–2 rules out of 5 |
| 3 | **Hard violators** | Molecules violating 3–5 rules out of 5 |

### ZINC-specific (uses tanimoto_sum and scaffold_sum)

| # | Name | Description |
|---|------|-------------|
| 5 | **Structurally close** | High `tanimoto_sum` — non-druglike but similar to ChEMBL by fingerprint |
| 6 | **Structurally far** | Low `tanimoto_sum` AND `scaffold_sum` = 0 — non-druglike with novel scaffolds |

---

## Experiment phases

### Phase A — screen sources

**Goal**: determine which sources show a positive signal from non-druglike additives.

**Design**: 3 sources × 2 fractions × 1 strategy = **6 runs**
- Sources: ZINC, COCONUT, HMDB
- Fractions: 2%, 5% (relative to ChEMBL size of ~329K → ~6.6K, ~16.5K molecules)
- Strategy: random (strategy 1)

**Transition criterion to Phase B** (per source):
- `score = validity × scaffold_entropy × int_div` exceeds baseline score
- **AND**: `validity ≥ baseline`, `mean_qed ≥ baseline`, `pct_lipinski ≥ baseline`

Sources meeting the criterion advance to Phase B. Sources failing all fractions are dropped.

### Phase B — find best strategy per source

**Goal**: for each advancing source, find the selection strategy that gives the best diversity gain.

**Design**: N sources × 5 strategies × 1 fraction (5%) = **N × 5 runs**
- Strategies: 1 (random), 2 (soft), 3 (hard) for all sources; + 5, 6 for ZINC only

Select the best strategy per source by `score = validity × scaffold_entropy × int_div` (among runs meeting control metrics).

### Phase C — sweep fractions with best strategy

**Goal**: find the optimal fraction for each source using its best strategy.

**Design**: N sources × 5 fractions × 1 strategy = **N × 5 runs**
- Fractions: 2%, 5%, 10%, 25%, 50%
- Strategy: best from Phase B per source

### Phase D — CompTox (optional)

If phases A–C confirm the hypothesis, download and prepare CompTox, then repeat Phase A with CompTox as a 4th source.

---

## Running an experiment

1. **Prepare non-druglike subset** locally:
   - Read source CSV
   - Parse each SMILES with RDKit, apply inverted `passes_druglike()`
   - For soft/hard strategies: count violated rules per molecule, filter by count
   - For ZINC-specific: sort/filter by `tanimoto_sum` and `scaffold_sum`
   - Sample N molecules (N = fraction × 329K)
   - Save as `<source>_<strategy>_<fraction>pct.csv`

2. **Merge with ChEMBL**:
   ```bash
   ssh aichem "cd /mnt/tank/scratch/aergardt/autoresearch/data && \
     head -1 chembl_clean.csv > mixed_<name>.csv && \
     tail -n +2 chembl_clean.csv >> mixed_<name>.csv && \
     tail -n +2 <additive>.csv >> mixed_<name>.csv"
   ```

3. **Transfer additive file**:
   ```bash
   scp <source>_<strategy>_<fraction>pct.csv aichem:/mnt/tank/scratch/aergardt/autoresearch/data/
   ```

4. **Run on remote** (always use `--save`):
   ```bash
   ssh aichem "cd /mnt/tank/scratch/aergardt/autoresearch && CUDA_VISIBLE_DEVICES=<N> PYTHONUNBUFFERED=1 nohup uv run train.py --data data/mixed_<name>.csv --save checkpoints/<name> > run_<name>.log 2>&1 &"
   ```
   Check free GPUs: `ssh aichem "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader"`

5. **Monitor**:
   ```bash
   ssh aichem "tail -20 /mnt/tank/scratch/aergardt/autoresearch/run_<name>.log | grep -v DEPRECATION"
   ```

6. **Extract final epoch metrics**:
   ```bash
   ssh aichem "grep 'Epoch 20/20' /mnt/tank/scratch/aergardt/autoresearch/run_<name>.log && \
     grep -E 'score:|validity:|scaffold_entropy:|int_div:|mean_qed:|pct_lipinski:' \
     /mnt/tank/scratch/aergardt/autoresearch/run_<name>.log | tail -5"
   ```

---

## Output format

Each epoch prints:
```
============================================================
Epoch 20/20  |  loss: 0.5535  |  time: 36s  |  elapsed: 1.0min
  score:            0.979  (validity × scaffold_entropy × int_div)
  ---
  validity:         0.888
  uniqueness:       1.000
  novelty:          0.990
  int_div:          0.883
  scaffold_entropy: 0.979
  snn:              0.300
  mean_qed:         0.147
  pct_lipinski:     0.005
  w1_logp:          0.147
  w1_qed:           0.005
  w1_mw:            7.6
```

**Decision metric**: `score = validity × scaffold_entropy × int_div` (computed post-hoc)
**Control metrics**: `validity`, `mean_qed`, `pct_lipinski`
**Diagnostic metrics**: `snn`, `novelty`, `uniqueness`, `w1_*`

---

## Logging results

Log to `results.tsv` (tab-separated) after each run. Keep it local — do not commit.

Columns:
```
commit  score  dataset  phase  strategy  fraction  vocab_size  validity  mean_qed  pct_lipinski  scaffold_entropy  int_div  status  description
```

- `phase`: `baseline`, `A`, `B`, `C`, or `D`
- `strategy`: `random`, `soft`, `hard`, `zinc_close`, `zinc_far`
- `fraction`: `0` (baseline), `0.02`, `0.05`, `0.10`, `0.25`, `0.50`
- `status`: `keep` (meets criterion), `discard` (fails), `crash`

Example:
```
commit  score  dataset  phase  strategy  fraction  validity  mean_qed  pct_lipinski  scaffold_entropy  int_div  status  description
a55b4ca  0.8693  chembl_clean  baseline  -  0  0.888  0.520  0.850  0.979  0.883  keep  baseline: drug-like ChEMBL 329K
b1c2d3e  0.8812  chembl_zinc_random_02  A  random  0.02  0.890  0.525  0.855  0.985  0.890  keep  +2% ZINC non-druglike improved scaffold_entropy
c3d4e5f  0.8550  chembl_coconut_random_05  A  random  0.05  0.870  0.490  0.820  0.980  0.885  discard  +5% COCONUT broke mean_qed and pct_lipinski
```

---

## The experiment loop

1. **Run baseline** first: `chembl_clean.csv` alone → record all metrics
2. **Phase A**: for each source (ZINC, COCONUT, HMDB) and fraction (2%, 5%):
   - Prepare non-druglike random subset
   - Merge, transfer, run, record
   - Apply transition criterion
3. **Phase B**: for each advancing source × strategies (1–3 universal, +5–6 for ZINC) at 5%:
   - Prepare, merge, transfer, run, record
   - Select best strategy per source
4. **Phase C**: for each source × best strategy × all fractions (2–50%):
   - Prepare, merge, transfer, run, record
   - Identify optimal fraction
5. **Phase D** (optional): download CompTox, repeat Phase A

**NEVER STOP**: do not pause between runs. The loop runs until manually interrupted.

**Goal**: Find the additive source, fraction, and selection strategy that maximizes `score = validity × scaffold_entropy × int_div` beyond the ChEMBL baseline while maintaining validity, mean_qed, and pct_lipinski.
