# Phase 0 summary — abundance encodings under cross-study evaluation

_Generated 2026-09-28 18:38 by `src/report.py` from `results/phase0_rf.csv`._

Train on AGP-IBD, test on HMP2 and Halfvarson with no fine-tuning; in-domain references on AGP-IBD. Cell = **median AUROC over seeds [median per-seed bootstrap 95% CI]**. Bootstrap CIs resample samples independently and ignore repeated samples per patient (HMP2/HV are longitudinal), so they are too narrow.

Pope et al. 2025 No-DR reference: HMP2 0.657 · Halfvarson 0.530 · AGP-IBD 0.717 (patient-blocked).

## Primary: spec RF (500 trees, sqrt, leaf 3, class_weight='balanced')

| encoding | HMP2 | Halfvarson | AGP-IBD-holdout | AGP-IBD-5fold |
|---|---|---|---|---|
| `ra` | 0.736 [0.646, 0.815] | 0.818 [0.751, 0.876] | 0.492 [0.443, 0.540] | 0.791 [0.768, 0.815] |
| `pa` | 0.728 [0.645, 0.806] | 0.814 [0.761, 0.863] | 0.446 [0.390, 0.497] | 0.786 [0.761, 0.810] |
| `rank_within` | 0.684 [0.596, 0.765] | 0.776 [0.717, 0.831] | 0.443 [0.395, 0.491] | 0.773 [0.747, 0.797] |
| `clr` | 0.522 [0.431, 0.613] | 0.614 [0.546, 0.682] | 0.438 [0.392, 0.487] | 0.723 [0.695, 0.749] |
| `rank_raw` | 0.706 [0.612, 0.794] | 0.789 [0.729, 0.841] | 0.446 [0.395, 0.495] | 0.769 [0.744, 0.794] |

**In-domain minus cross-study gap** (median AUROC; positive = worse under shift)

| encoding | holdout − HMP2 | holdout − HV | 5-fold − HMP2 | 5-fold − HV |
|---|---|---|---|---|
| `ra` | -0.243 | -0.326 | +0.056 | -0.027 |
| `pa` | -0.282 | -0.367 | +0.058 | -0.027 |
| `rank_within` | -0.242 | -0.333 | +0.089 | -0.003 |
| `clr` | -0.085 | -0.176 | +0.201 | +0.109 |
| `rank_raw` | -0.260 | -0.343 | +0.064 | -0.020 |

**Paired difference vs `ra`** (AUROC of seed-averaged predictions, encoding − ra, paired bootstrap 95% CI; CI excluding 0 marked *)

| encoding | HMP2 | Halfvarson | AGP-IBD-holdout | AGP-IBD-5fold |
|---|---|---|---|---|
| `pa` | -0.004 [-0.051, +0.042] | +0.020 [-0.011, +0.052] | -0.039 [-0.061, -0.017]* | -0.007 [-0.017, +0.003] |
| `rank_within` | -0.049 [-0.103, +0.001] | -0.029 [-0.072, +0.012] | -0.036 [-0.058, -0.013]* | -0.020 [-0.033, -0.008]* |
| `clr` | -0.218 [-0.314, -0.114]* | -0.206 [-0.300, -0.110]* | -0.048 [-0.085, -0.009]* | -0.069 [-0.089, -0.051]* |
| `rank_raw` | -0.031 [-0.081, +0.020] | -0.019 [-0.065, +0.028] | -0.042 [-0.068, -0.018]* | -0.023 [-0.035, -0.011]* |

## Sensitivity: unweighted RF (identical but class_weight=None, as DeepMicro)

| encoding | HMP2 | Halfvarson | AGP-IBD-holdout |
|---|---|---|---|
| `ra` | 0.659 [0.559, 0.751] | 0.339 [0.269, 0.407] | 0.566 [0.518, 0.609] |
| `pa` | 0.645 [0.552, 0.731] | 0.390 [0.316, 0.460] | 0.591 [0.546, 0.630] |
| `rank_within` | 0.567 [0.468, 0.664] | 0.340 [0.274, 0.410] | 0.609 [0.565, 0.651] |
| `clr` | 0.714 [0.615, 0.811] | 0.435 [0.359, 0.510] | 0.570 [0.525, 0.620] |
| `rank_raw` | 0.564 [0.467, 0.663] | 0.362 [0.288, 0.434] | 0.631 [0.588, 0.672] |

**In-domain minus cross-study gap** (median AUROC; positive = worse under shift)

| encoding | holdout − HMP2 | holdout − HV | 5-fold − HMP2 | 5-fold − HV |
|---|---|---|---|---|
| `ra` | -0.093 | +0.227 | — | — |
| `pa` | -0.054 | +0.201 | — | — |
| `rank_within` | +0.042 | +0.270 | — | — |
| `clr` | -0.144 | +0.135 | — | — |
| `rank_raw` | +0.067 | +0.269 | — | — |

**Paired difference vs `ra`** (AUROC of seed-averaged predictions, encoding − ra, paired bootstrap 95% CI; CI excluding 0 marked *)

| encoding | HMP2 | Halfvarson | AGP-IBD-holdout |
|---|---|---|---|
| `pa` | -0.009 [-0.099, +0.076] | +0.042 [-0.012, +0.096] | +0.014 [-0.026, +0.052] |
| `rank_within` | -0.090 [-0.199, +0.013] | +0.005 [-0.050, +0.061] | +0.039 [-0.008, +0.082] |
| `clr` | +0.066 [+0.004, +0.124]* | +0.092 [+0.065, +0.122]* | -0.004 [-0.024, +0.015] |
| `rank_raw` | -0.101 [-0.212, +0.006] | +0.022 [-0.033, +0.077] | +0.058 [+0.015, +0.099]* |

## Rank ties (mean tied non-zero entries per sample, spec RF runs)

| encoding | HMP2 | Halfvarson | AGP-IBD-holdout | AGP-IBD-5fold |
|---|---|---|---|---|
| `rank_within` | 0.0 | 0.0 | 0.2 | 0.3 |
| `rank_raw` | 72.2 | 178.2 | 208.0 | 206.4 |

## Caveats

- **In-domain 5-fold CV is leakage-inflated** (S2: 32% of IBD test samples have an IBD nearest training neighbour vs 5% base rate). Patient IDs are unavailable.
- **The shipped holdout is leakage-free but batch-shifted**: an RF separates Pope's IBD_train from IBD_test with AUROC 0.935 (5-fold, `ra`); IBD_test has min depth 5,115 vs 17 and higher richness (median 251 vs 216), i.e. it was processed differently. There is therefore **no clean in-domain estimate** in this data: 5-fold CV is an upper bound, the shipped holdout behaves like another shifted cohort, and the 'gap' columns must be read with that in mind.
- **Halfvarson is fragile**: with `ra`, AUROC moves from 0.818 (balanced RF) to 0.340 (unweighted RF), seed 0. Compare encodings within an RF arm, never across.
- ASV overlap on Fig 2 is Pope et al.'s BLAST figure. Our ID-match overlap is 0.961 (HMP2) / 0.983 (HV), near 1 by construction because the cohorts were pre-mapped to the AGP vocabulary.
- One vocabulary (26,726 IDs), one classifier per arm, no per-encoding tuning; all encoding statistics fitted on training data only.
