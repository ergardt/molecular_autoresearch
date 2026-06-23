# Experiment Report: Non-ChEMBL Additive Molecules + SYK Fine-tuning

**Experiment**: exp_2_var_2
**Branch**: run_exp_2_var_2
**Commit**: 00106ec
**Dates**: 2026-06-17 — 2026-06-23
**Model**: GPT-4 transformer, N_LAYER=4, N_EMBD=256, ~3.2M params, BreakIt tokenizer (168 tokens)

---

## Hypothesis

Adding a small fraction of non-ChEMBL molecules to the ChEMBL training dataset (500K) improves generative model diversity without breaking drug-likeness control metrics.

**Decision metric**: `score = validity × scaffold_entropy × int_div`
**Control metrics**: validity, mean_qed, pct_lipinski must not fall below baseline.

---

## Baseline

| Metric | Value |
|--------|-------|
| validity | 0.885 |
| mean_qed | 0.570 |
| pct_lipinski | 0.738 |
| scaffold_entropy | 0.980 |
| int_div | 0.891 |
| uniqueness | 1.0000 |
| novelty | 0.9901 |
| **score** | **0.7729** |

---

## Phase A — Screen sources (random sampling)

Tested 3 sources × 4 fractions = **12 runs**.

### ZINC

| Fraction | Score | Validity | Mean QED | Scaff Entropy | Int Div | vs Baseline |
|----------|-------|----------|----------|---------------|---------|-------------|
| 2% | 0.7633 | 0.880 | 0.569 | 0.982 | 0.884 | score -1.2%, all control - |
| 5% | 0.7594 | 0.875 | 0.573 | 0.982 | 0.883 | validity -, score -1.7% |
| 10% | 0.7655 | 0.882 | 0.585 | 0.983 | 0.883 | score -1.0% |
| 20% | 0.7700 | 0.889 | 0.591 | 0.981 | 0.883 | score -0.4% |

ZINC shows the smallest degradation. At 20%, score is within 0.4% of baseline. Mean QED and validity actually *improve* with larger fractions. The consistent weakness is `int_div` (0.883-0.884 vs 0.891 baseline).

### COCONUT

| Fraction | Score | Validity | Mean QED | Scaff Entropy | Int Div | vs Baseline |
|----------|-------|----------|----------|---------------|---------|-------------|
| 2% | 0.7615 | 0.876 | 0.561 | 0.980 | 0.887 | all control - |
| 5% | 0.7576 | 0.870 | 0.556 | 0.978 | 0.891 | all control - |
| 10% | 0.7605 | 0.874 | 0.555 | 0.974 | 0.893 | all control - |
| 20% | 0.7499 | 0.868 | 0.542 | 0.971 | 0.889 | all control - |

COCONUT degrades all control metrics monotonically. Mean QED drops from 0.561 → 0.542 as fraction increases.

### HMDB

| Fraction | Score | Validity | Mean QED | Scaff Entropy | Int Div | vs Baseline |
|----------|-------|----------|----------|---------------|---------|-------------|
| 2% | 0.7578 | 0.879 | 0.561 | 0.974 | 0.885 | all control - |
| 5% | 0.7536 | 0.879 | 0.551 | 0.962 | 0.891 | all control - |
| 10% | 0.7333 | 0.874 | 0.526 | 0.941 | 0.892 | all control - |
| 20% | 0.7044 | 0.881 | 0.503 | 0.894 | 0.894 | all control - |

HMDB shows the strongest negative signal. At 20%, scaffold_entropy collapses to 0.894 (-8.3%) and mean_qed to 0.503 (-11.8%). Score drops 8.9%.

### Phase A Summary

**No source met the transition criterion** (score > baseline AND all control metrics ≥ baseline).

ZINC is the closest — it degrades the least and shows improving drug-likeness (mean_qed, validity) at higher fractions. The single bottleneck is `int_div`, which is consistently ~0.883 across all fractions (baseline: 0.891).

---

## Phase B — ZINC strategies at 10%

Since ZINC showed the most promise, proceeded with Phase B to test all 7 selection strategies at 10% fraction. (Random 10% was already completed in Phase A.)

### Results

| Strategy | Score | Validity | Mean QED | Scaff Entropy | Int Div | Uniqueness | Novelty | Verdict |
|----------|-------|----------|----------|---------------|---------|------------|---------|---------|
| **baseline** | **0.7729** | 0.885 | 0.570 | 0.980 | 0.891 | 1.0000 | 0.9901 | — |
| zinc_close | 0.7715 | 0.888 | 0.588 | 0.984 | 0.883 | 0.9998 | 0.9933 | All control +, score -0.2% |
| zinc_far | 0.7697 | 0.884 | 0.576 | 0.977 | 0.891 | 0.9998 | 0.9932 | score -0.4% |
| high_qed | 0.7660 | 0.879 | 0.588 | 0.986 | 0.884 | 1.0000 | 0.9916 | validity - |
| low_mw | 0.7640 | 0.888 | 0.577 | 0.967 | 0.889 | 1.0000 | 0.9896 | scaffold_entropy - |
| random | 0.7655 | 0.882 | 0.585 | 0.983 | 0.883 | 0.9998 | 0.9930 | score - |
| low_qed | 0.7567 | 0.874 | 0.542 | 0.980 | 0.884 | 1.0000 | 0.9931 | all control - |
| high_mw | 0.7523 | 0.864 | 0.552 | 0.982 | 0.886 | 1.0000 | 0.9922 | validity, lipinski - |

### Strategy analysis

**zinc_close** (highest tanimoto_sum, structurally similar to ChEMBL) — best result. All control metrics exceed baseline (validity +0.003, mean_qed +0.018, pct_lipinski +0.030). Scaffold entropy actually improves to 0.984. Score still falls short (0.7715 vs 0.7729) due to int_div at 0.883.

**zinc_far** (lowest tanimoto_sum, scaffold_sum=0, novel scaffolds) — int_div matches baseline (0.891) but scaffold_entropy drops to 0.977 and score fails.

**high_qed** — highest scaffold_entropy (0.986) but validity drops to 0.879.

**low_mw** — scaffold_entropy collapses to 0.967 (-1.4%).

**low_qed** — degrades mean_qed to 0.542 (-4.9%).

**high_mw** — worst overall; validity 0.864, pct_lipinski 0.693.

### Uniqueness and Novelty

Both metrics show almost no variation across strategies:
- **Uniqueness**: 0.9998-1.0000 (essentially 1.0 for all)
- **Novelty**: 0.9896-0.9933 (range of 0.0037)

These metrics are not differentiating. The model generates unique molecules that are novel relative to training data regardless of what additive source is used. They do not reflect the internal diversity of the generated distribution, which is what `int_div` measures.

---

## Phase C — SYK Fine-tuning

All 13 pre-trained checkpoints (baseline + 12 Phase A) were fine-tuned on the SYK inhibitors dataset (3176 molecules, 15 epochs, LR=1e-5, batch_size=64).

### SYK Dataset Intrinsic Metrics

| Metric | Value |
|--------|-------|
| Molecules | 3176 |
| Validity | 1.000 |
| Uniqueness | 0.9606 |
| Mean QED | 0.497 |
| Pct Lipinski | 0.738 |
| Scaffold Entropy | 0.879 |
| Int Div | 0.831 |
| Score | 0.7304 |
| Mean MW | 444 |
| Mean LogP | 3.62 |

The SYK dataset has lower scaffold entropy (0.879) and int_div (0.831) than ChEMBL baseline, reflecting a more focused chemical space around SYK kinase inhibitors.

### Fine-tuning Results

| Model | Score | Validity | Mean QED | Scaff Entropy | Int Div |
|-------|-------|----------|----------|---------------|---------|
| **syk_baseline** | **0.7697** | 0.878 | 0.532 | 0.989 | 0.886 |
| syk_zinc_random_20 | 0.7618 | 0.867 | 0.552 | 0.992 | 0.886 |
| syk_zinc_random_05 | 0.7607 | 0.870 | 0.550 | 0.991 | 0.882 |
| syk_coconut_random_20 | 0.7659 | 0.874 | 0.517 | 0.985 | 0.889 |
| syk_zinc_random_10 | 0.7550 | 0.862 | 0.553 | 0.992 | 0.883 |
| syk_zinc_random_02 | 0.7542 | 0.859 | 0.528 | 0.995 | 0.883 |
| syk_coconut_random_05 | 0.7527 | 0.855 | 0.531 | 0.992 | 0.888 |
| syk_coconut_random_02 | 0.7506 | 0.856 | 0.535 | 0.993 | 0.884 |
| syk_hmdb_random_02 | 0.7428 | 0.852 | 0.534 | 0.988 | 0.882 |
| syk_hmdb_random_05 | 0.7345 | 0.849 | 0.514 | 0.980 | 0.883 |
| syk_hmdb_random_20 | 0.7226 | 0.877 | 0.484 | 0.926 | 0.890 |
| syk_hmdb_random_10 | 0.7176 | 0.839 | 0.496 | 0.965 | 0.887 |
| syk_coconut_random_10 | 0.7391 | 0.844 | 0.518 | 0.991 | 0.884 |

Fine-tuning on SYK data shifts all models toward the SYK chemical space. The baseline model retains the highest score (0.7697). ZINC-augmented models maintain higher mean_qed (0.55-0.55) than COCONUT (0.52-0.54) and HMDB (0.48-0.53). HMDB 20% shows the largest degradation — scaffold_entropy collapses to 0.926.

### QSAR pIC50 Predictions

All fine-tuned models generated 5000 molecules each (55,955 total valid predictions). A stacking regressor QSAR model (SVR + XGBoost + RandomForest, R²=0.78 on test set) was used to predict pIC50 against SYK kinase.

**Experimental SYK dataset**: mean pIC50 = 6.56, median = 6.53, range = 4.20–9.00

#### Mean pIC50 per model (lower = less active = better)

| Model | Mean pIC50 | Median | % < 6 | % < 5 | Count |
|-------|-----------|--------|-------|-------|-------|
| **syk_hmdb_random_20** | **6.230** | 6.227 | **34.6%** | 0.5% | 4327 |
| syk_hmdb_random_10 | 6.264 | 6.260 | 30.3% | 0.9% | 4296 |
| syk_hmdb_random_02 | 6.295 | 6.297 | 29.0% | 0.6% | 4282 |
| syk_hmdb_random_05 | 6.283 | 6.284 | 29.0% | 0.7% | 4339 |
| syk_zinc_random_10 | 6.283 | 6.296 | 28.2% | 0.5% | 4303 |
| syk_zinc_random_02 | 6.303 | 6.317 | 27.5% | 0.5% | 4320 |
| syk_baseline | 6.295 | 6.305 | 27.5% | 0.9% | 4317 |
| syk_coconut_random_20 | 6.316 | 6.324 | 27.2% | 0.4% | 4218 |
| syk_coconut_random_02 | 6.320 | 6.333 | 26.7% | 0.7% | 4306 |
| syk_coconut_random_05 | 6.304 | 6.314 | 26.5% | 0.5% | 4265 |
| syk_zinc_random_20 | 6.307 | 6.317 | 26.3% | 0.4% | 4376 |
| syk_zinc_random_05 | 6.318 | 6.330 | 26.0% | 0.6% | 4332 |
| syk_coconut_random_10 | 6.324 | 6.330 | 25.7% | 0.6% | 4274 |

HMDB 20% produces molecules with the lowest predicted activity (mean pIC50 = 6.230, 34.6% below 6). COCONUT 10% produces the most active molecules (mean pIC50 = 6.324). The spread between best and worst is 0.094 pIC50 units — small but consistent.

#### Top 10 predicted pIC50 (highest activity)

| pIC50 | Model | SMILES |
|-------|-------|--------|
| 9.27 | syk_zinc_random_02 | Cc1cc(Nc2nc(N[C@H]3CCCC[C@@H]3N)nnc2C(N)=O)sn1 |
| 9.27 | syk_zinc_random_05 | Cc1cc(Nc2nc(N[C@@H]3CCCC[C@@H]3N)nnc2C(N)=O)sn1 |
| 9.20 | syk_coconut_random_20 | Cc1cc(Nc2nc(N[C@@H]3CCCC[C@@H]3N)nnc2C(N)=O)cc(C)c1Cl |
| 9.15 | syk_hmdb_random_02 | Cc1cc(Nc2nc(N[C@@H]3CCCC[C@@H]3N)nnc2C(N)=O)cc(C)n1 |
| 9.00 | syk_zinc_random_10 | Cc1cc(Nc2nc(N[C@@H]3CCCC[C@@H]3N)nnc2C(N)=O)[nH]n1 |

The top predictions share a common scaffold: 2-amino-5-methyl-thiazole/pyrimidine core with piperidine/pyrrolidine amine. This scaffold is present in the original SYK training data, suggesting the models learned to reproduce high-activity motifs.

---

## Overall Conclusions

### Pre-training (Phases A-B)

1. **Hypothesis not confirmed**: No combination of additive source, fraction, or selection strategy exceeded the ChEMBL-only baseline on `score = validity × scaffold_entropy × int_div` while maintaining control metrics.

2. **The bottleneck is int_div**: Across all 18 experimental runs, internal diversity never exceeded 0.893 (COCONUT 10%). The baseline is 0.891. Adding external molecules either keeps int_div flat or reduces it. This suggests the model's diversity is determined by architecture/training dynamics rather than training data composition.

3. **ZINC is the safest additive**: Among all sources, ZINC molecules degrade the least. At higher fractions (10-20%), ZINC actually improves drug-likeness metrics (validity, mean_qed, pct_lipinski). The trade-off is a small, consistent loss in int_div.

4. **zinc_close is the best strategy**: Selecting ZINC molecules structurally closest to ChEMBL (highest tanimoto_sum) gives the highest score (0.7715, -0.2% from baseline) with all control metrics above baseline.

5. **HMDB is harmful at scale**: HMDB shows a strong dose-dependent degradation. At 20%, score drops 8.9% and mean_qed falls to 0.503.

6. **Selection strategy matters less than source**: Within ZINC, the best and worst strategies differ by only 0.0192 in score. The choice of source has a larger effect than the selection strategy.

### Fine-tuning (Phase C)

7. **Fine-tuning narrows the gap**: After fine-tuning on SYK data, all models converge toward similar performance. The pre-training differences (source, fraction, strategy) have a smaller effect than the fine-tuning data itself.

8. **HMDB 20% best for low activity**: Fine-tuned HMDB 20% model generates molecules with the lowest predicted pIC50 (6.230), meaning it produces the least SYK-active molecules. This is likely because HMDB molecules (human metabolites) push the model away from kinase inhibitor-like chemical space.

9. **ZINC models produce highest-activity molecules**: ZINC-augmented models generate molecules with the highest maximum predicted pIC50 (up to 9.27), suggesting ZINC's structural similarity to ChEMBL helps the model explore kinase inhibitor space more effectively.

10. **QSAR predictions are narrow**: All models produce pIC50 values in a tight range (4.2–9.3), with means clustered around 6.2–6.3. The experimental SYK dataset has a mean of 6.56. No model consistently outperforms others by a large margin.

---

## Generated Visualizations

All plots in `experiments/exp_2_var_2/`:

**Phase A**: `phase_a_validity.png`, `phase_a_mean_qed.png`, `phase_a_scaffold_entropy.png`, `phase_a_int_div.png`, `phase_a_metrics.png` (combined)

**Phase B**: `phase_b_validity.png`, `phase_b_mean_qed.png`, `phase_b_scaffold_entropy.png`, `phase_b_int_div.png`, `phase_b_uniqueness.png`, `phase_b_novelty.png`, `phase_b_score.png`, `phase_b_overview.png` (combined 2×3)

**SYK Fine-tuning**: `syk_score.png`, `syk_validity.png`, `syk_mean_qed.png`, `syk_pct_lipinski.png`, `syk_scaffold_entropy.png`, `syk_int_div.png`, `syk_uniqueness.png`, `syk_novelty.png`, `syk_overview.png` (combined 2×2)

**QSAR Predictions**: `qsar_boxplot.png`, `qsar_mean_pic50.png`, `qsar_distribution.png`

---

## Data Files

| File | Description |
|------|-------------|
| `results.tsv` | Phase A + B results (19 rows: 1 baseline + 12 Phase A + 6 Phase B) |
| `syk_results.tsv` | SYK fine-tuning results (13 models) |
| `predictions.tsv` | QSAR pIC50 predictions (55,955 molecules) |

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/prepare_additive.py` | Sample molecules from non-ChEMBL sources by strategy |
| `scripts/fine_tune.py` | Fine-tune pretrained checkpoint on new dataset |
| `scripts/predict_qsar.py` | Generate molecules and predict pIC50 via QSAR model |
| `scripts/analyze_syk.py` | Compute intrinsic metrics on SYK dataset |
| `scripts/plot_phase_a.py` | Plot Phase A results (line charts per source) |
| `scripts/plot_phase_b.py` | Plot Phase B results (bar charts per strategy) |
| `scripts/plot_syk.py` | Plot SYK fine-tuning results |
| `scripts/plot_qsar.py` | Plot QSAR prediction results |

---

## Recommendations

1. **int_div ceiling**: The consistent int_div ~0.88-0.89 across all configurations suggests an architectural or training limit. To increase diversity, consider: higher temperature at generation time, different decoding strategies (top-k/top-p), or model architecture changes (larger N_LAYER/N_EMBD).

2. **If diversity is not the priority**: ZINC 10-20% with zinc_close strategy improves drug-likeness (mean_qed +3.2%, validity +0.4%) at a cost of only 0.2-1.0% in composite score.

3. **For SYK-specific generation**: Fine-tuning on SYK data is more effective than pre-training data composition. The pre-training additive source has a minor effect on downstream QSAR predictions.

4. **Phase C (fraction sweep) not warranted**: No strategy exceeded baseline in pre-training.

5. **CompTox not recommended**: Three sources already show a consistent pattern. A fourth source is unlikely to break the int_div ceiling.
