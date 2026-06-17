# molgen-autoresearch: non-ChEMBL additive experiment

Autonomous experimentation on SMILES-based molecular generation.

**Hypothesis**: adding a small fraction of non-ChEMBL molecules (molecules from sources other than ChEMBL) to the ChEMBL training dataset improves generative model diversity without breaking drug-likeness control metrics.

The model is frozen at N_LAYER=4, N_EMBD=256. Training data = `chembl_500k_seed0.csv` (100%) + variable additive subset from non-ChEMBL sources.

---

## Setup

To set up a new experiment, work with the user to:

1. **Branch**: `run_exp_2_var_2` — current working branch.
2. **Verify data exists** on the remote node:
   ```bash
   ssh aichem "ls -lh /mnt/tank/scratch/aergardt/autoresearch/data/"
   ```
   Required files:
   - `chembl_500k_seed0.csv` — drug-like ChEMBL molecules (frozen baseline)
   - `zinc_vs_chembl_full_QED.csv` — ZINC molecules with smiles, tanimoto_sum, scaffold_sum, QED
   - `coconut_smiles.csv` — COCONUT natural/isolated compounds
   - `hmdb_smiles.csv` — HMDB human metabolites

3. **Initialize results.tsv** locally with a header row.
4. **Confirm and go**.

---

## Experimentation

**Workflow** — hybrid setup:
- **Prepare additive subsets locally**: sample molecules from source by strategy, save to CSV
- **Transfer to remote**: `scp <dataset>.csv aichem:/mnt/tank/scratch/aergardt/autoresearch/data/`
- **Merge with ChEMBL on remote**: concatenate additive CSV with chembl_500k_seed0.csv
- **Run on remote**: execute on GPU cluster node (`aichem`) — **train.py is read-only**
- **Retrieve results**: fetch logs back to analyze

**What you CAN do:**
- Change the training dataset via `--data` argument
- Vary the additive fraction (2%, 5%, 10%, 25%, 50% relative to ChEMBL)
- Select molecules by strategy (see Selection strategies below)
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

## Additive sources

### 1. ZINC — structurally diverse, scored against ChEMBL

`zinc_vs_chembl_full_QED.csv`

| Column | Type | Meaning |
|--------|------|---------|
| smiles | str | canonical SMILES |
| tanimoto_sum | float | sum of Tanimoto similarities (Morgan FP, radius=2) to all ChEMBL molecules |
| scaffold_sum | int | count of Bemis-Murcko scaffold occurrences in ChEMBL |
| qed | float | QED score |

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
| 1 | **Random** | Uniform random sample from all molecules in source |
| 2 | **High QED** | Sort by QED descending, take top N — drug-like molecules from non-ChEMBL |
| 3 | **Low QED** | Sort by QED ascending, take top N — non-druglike molecules from non-ChEMBL |
| 4 | **High MW** | Sort by molecular weight descending, take top N — heavy molecules |
| 5 | **Low MW** | Sort by molecular weight ascending, take top N — light molecules |

### ZINC-specific (uses tanimoto_sum and scaffold_sum)

| # | Name | Description |
|---|------|-------------|
| 6 | **Structurally close** | High `tanimoto_sum` — non-ChEMBL but similar to ChEMBL by fingerprint |
| 7 | **Structurally far** | Low `tanimoto_sum` AND `scaffold_sum` = 0 — non-ChEMBL with novel scaffolds |

---

## Experiment phases

### Phase A — screen sources

**Goal**: determine which sources show a positive signal from non-ChEMBL additives.

**Design**: 3 sources × 2 fractions × 1 strategy = **6 runs**
- Sources: ZINC, COCONUT, HMDB
- Fractions: 2%, 5%
- Strategy: random (strategy 1)

**Transition criterion to Phase B** (per source):
- `score = validity × scaffold_entropy × int_div` exceeds baseline score
- **AND**: `validity ≥ baseline`, `mean_qed ≥ baseline`, `pct_lipinski ≥ baseline`

Sources meeting the criterion advance to Phase B. Sources failing all fractions are dropped.

### Phase B — find best strategy per source

**Goal**: for each advancing source, find the selection strategy that gives the best diversity gain.

**Design**: N sources × M strategies × 1 fraction (5%)
- ZINC: M = 7 (strategies 1–5 universal + 6, 7 ZINC-specific)
- COCONUT, HMDB: M = 5 (strategies 1–5 universal)

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

1. **Prepare additive subset** locally:
   - Read source CSV
   - Apply selection strategy (random sample, sort by QED/MW, or sort by tanimoto_sum/scaffold_sum for ZINC)
   - Sample N molecules (N = fraction × 500,000, size of chembl_500k_seed0.csv)
   - Save as `<source>_<strategy>_<fraction>pct.csv`

2. **Transfer additive file**:
   ```bash
   scp <source>_<strategy>_<fraction>pct.csv aichem:/mnt/tank/scratch/aergardt/autoresearch/data/
   ```

3. **Merge with ChEMBL**:
   ```bash
   ssh aichem "cd /mnt/tank/scratch/aergardt/autoresearch/data && \
     head -1 chembl_500k_seed0.csv > mixed_<name>.csv && \
     tail -n +2 chembl_500k_seed0.csv >> mixed_<name>.csv && \
     tail -n +2 <additive>.csv >> mixed_<name>.csv"
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
     /mnt/tank/scratch/aergardt/autoresearch/run_<name>.log | tail -6"
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
- `strategy`: `random`, `high_qed`, `low_qed`, `high_mw`, `low_mw`, `zinc_close`, `zinc_far`
- `fraction`: `0` (baseline), `0.02`, `0.05`, `0.10`, `0.25`, `0.50`
- `status`: `keep` (meets criterion), `discard` (fails), `crash`

Example:
```
commit  score  dataset  phase  strategy  fraction  validity  mean_qed  pct_lipinski  scaffold_entropy  int_div  status  description
a55b4ca  0.8693  chembl_500k_seed0  baseline  -  0  0.888  0.520  0.850  0.979  0.883  keep  baseline: drug-like ChEMBL 500K
b1c2d3e  0.8812  chembl_zinc_random_02  A  random  0.02  0.890  0.525  0.855  0.985  0.890  keep  +2% ZINC improved scaffold_entropy
c3d4e5f  0.8550  chembl_coconut_random_05  A  random  0.05  0.870  0.490  0.820  0.980  0.885  discard  +5% COCONUT broke mean_qed and pct_lipinski
```

---

## The experiment loop

1. **Run baseline** first: `chembl_500k_seed0.csv` alone → record all metrics
2. **Phase A**: for each source (ZINC, COCONUT, HMDB) and fraction (2%, 5%):
   - Prepare random subset
   - Transfer, merge, run, record
   - Apply transition criterion
3. **Phase B**: for each advancing source × all applicable strategies at 5%:
   - Prepare, transfer, merge, run, record
   - Select best strategy per source
4. **Phase C**: for each source × best strategy × all fractions (2–50%):
   - Prepare, transfer, merge, run, record
   - Identify optimal fraction
5. **Phase D** (optional): download CompTox, repeat Phase A

**NEVER STOP**: do not pause between runs. The loop runs until manually interrupted.

**Goal**: Find the additive source, fraction, and selection strategy that maximizes `score = validity × scaffold_entropy × int_div` beyond the ChEMBL baseline while maintaining validity, mean_qed, and pct_lipinski.
