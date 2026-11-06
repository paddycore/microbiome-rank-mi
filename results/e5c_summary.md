# E5c summary

_Generated 2026-11-07 01:49 by `src/e5c_report.py` from `results/e5c.csv`._

Everything except SC-2 is correlational: a score that tracks richness is *explained by* it, which does not show the model *uses* it. CIs ignore repeated samples per patient (no subject IDs ship with the data), so they are too narrow. Cohorts are never pooled.


## SC-0: richness and read depth alone (HC1)

AUROC of each variable for IBD, bootstrap 95% CI. Below 0.5 = higher in controls. Direction is named only when the CI excludes 0.5.

| Variable | AGP-IBD | HMP2 | Halfvarson |
|---|---|---|---|
| richness (all taxa) | 0.541 [0.512, 0.568], higher in IBD | 0.407 [0.310, 0.502], no clear direction | 0.364 [0.292, 0.437], higher in controls |
| taxa the model sees (max 512) | 0.541 [0.512, 0.568], higher in IBD | 0.407 [0.310, 0.502], no clear direction | 0.372 [0.304, 0.443], higher in controls |
| richness (distinct vocab IDs) | 0.541 [0.512, 0.568], higher in IBD | 0.407 [0.310, 0.502], no clear direction | 0.302 [0.236, 0.373], higher in controls |
| log10 read depth | 0.565 [0.538, 0.591], higher in IBD | 0.641 [0.560, 0.715], higher in IBD | 0.590 [0.507, 0.670], higher in IBD |

## SC-1: what the score adds beyond richness and depth (HC2)

Logistic models cross-fitted over 5 folds; Δ = paired bootstrap 95% CI on the out-of-fold predictions. Members: median [min–max] over the models averaged into each source (10 heads, 5 RF seeds, 5 E5b seeds). Covariates `n_taxa` = richness as the model sees it (capped at 512).


**HMP2, covariates: richness + log10 depth**

| Source | M0: richness + depth | M2: score alone | M1: both | Δ M1 − M0 (score over simple signals) | Δ M1 − M2 (simple signals over score) | AUROC of residual score | Δ M1 − M0, members | residual, members |
|---|---|---|---|---|---|---|---|---|
| Pope's ensemble | 0.718 [0.645, 0.788] | 0.678 [0.577, 0.768] | 0.743 [0.660, 0.811] | 0.025 [-0.021, 0.072] | 0.066 [0.007, 0.126] | 0.617 [0.515, 0.708] | 0.024 [0.012, 0.054] | 0.606 [0.582, 0.698] |
| RF, relative abundance | 0.718 [0.645, 0.788] | 0.738 [0.650, 0.821] | 0.773 [0.694, 0.847] | 0.055 [0.000, 0.114] | 0.035 [-0.015, 0.085] | 0.620 [0.530, 0.713] | 0.053 [0.039, 0.060] | 0.618 [0.603, 0.624] |
| E5b heads, 1 epoch | 0.718 [0.645, 0.788] | 0.652 [0.548, 0.745] | 0.738 [0.659, 0.805] | 0.020 [-0.024, 0.062] | 0.086 [0.016, 0.158] | 0.609 [0.508, 0.703] | 0.021 [0.020, 0.021] | 0.610 [0.604, 0.612] |
| E5b heads, 50 epochs | 0.718 [0.645, 0.788] | 0.497 [0.394, 0.588] | 0.701 [0.625, 0.775] | -0.017 [-0.032, -0.002] | 0.204 [0.076, 0.335] | 0.508 [0.416, 0.606] | -0.016 [-0.020, -0.015] | 0.509 [0.501, 0.515] |

_AUROC of the score inside richness tertiles (23-109, 109-160, 161-330)_

| Source | tertile 1 | tertile 2 | tertile 3 |
|---|---|---|---|
| Pope's ensemble | 0.661 [0.373, 0.899] (n 66.0, IBD 57.0) | 0.565 [0.371, 0.760] (n 66.0, IBD 52.0) | 0.707 [0.556, 0.853] (n 65.0, IBD 46.0) |
| RF, relative abundance | 0.669 [0.406, 0.895] (n 66.0, IBD 57.0) | 0.685 [0.522, 0.827] (n 66.0, IBD 52.0) | 0.732 [0.599, 0.850] (n 65.0, IBD 46.0) |
| E5b heads, 1 epoch | 0.669 [0.400, 0.900] (n 66.0, IBD 57.0) | 0.516 [0.326, 0.707] (n 66.0, IBD 52.0) | 0.650 [0.491, 0.810] (n 65.0, IBD 46.0) |
| E5b heads, 50 epochs | 0.532 [0.322, 0.758] (n 66.0, IBD 57.0) | 0.401 [0.270, 0.545] (n 66.0, IBD 52.0) | 0.370 [0.233, 0.521] (n 65.0, IBD 46.0) |

**HMP2, covariates: n_taxa + log10 depth**

| Source | M0: richness + depth | M2: score alone | M1: both | Δ M1 − M0 (score over simple signals) | Δ M1 − M2 (simple signals over score) | AUROC of residual score | Δ M1 − M0, members | residual, members |
|---|---|---|---|---|---|---|---|---|
| Pope's ensemble | 0.718 [0.645, 0.788] | 0.678 [0.577, 0.768] | 0.743 [0.660, 0.811] | 0.025 [-0.021, 0.072] | 0.066 [0.007, 0.126] | 0.617 [0.515, 0.708] | 0.024 [0.012, 0.054] | 0.606 [0.582, 0.698] |
| RF, relative abundance | 0.718 [0.645, 0.788] | 0.738 [0.650, 0.821] | 0.773 [0.694, 0.847] | 0.055 [0.000, 0.114] | 0.035 [-0.015, 0.085] | 0.620 [0.530, 0.713] | 0.053 [0.039, 0.060] | 0.618 [0.603, 0.624] |
| E5b heads, 1 epoch | 0.718 [0.645, 0.788] | 0.652 [0.548, 0.745] | 0.738 [0.659, 0.805] | 0.020 [-0.024, 0.062] | 0.086 [0.016, 0.158] | 0.609 [0.508, 0.703] | 0.021 [0.020, 0.021] | 0.610 [0.604, 0.612] |
| E5b heads, 50 epochs | 0.718 [0.645, 0.788] | 0.497 [0.394, 0.588] | 0.701 [0.625, 0.775] | -0.017 [-0.032, -0.002] | 0.204 [0.076, 0.335] | 0.508 [0.416, 0.606] | -0.016 [-0.020, -0.015] | 0.509 [0.501, 0.515] |

_AUROC of the score inside n_taxa tertiles (23-109, 109-160, 161-330)_

| Source | tertile 1 | tertile 2 | tertile 3 |
|---|---|---|---|
| Pope's ensemble | 0.661 [0.373, 0.899] (n 66.0, IBD 57.0) | 0.565 [0.371, 0.760] (n 66.0, IBD 52.0) | 0.707 [0.556, 0.853] (n 65.0, IBD 46.0) |
| RF, relative abundance | 0.669 [0.406, 0.895] (n 66.0, IBD 57.0) | 0.685 [0.522, 0.827] (n 66.0, IBD 52.0) | 0.732 [0.599, 0.850] (n 65.0, IBD 46.0) |
| E5b heads, 1 epoch | 0.669 [0.400, 0.900] (n 66.0, IBD 57.0) | 0.516 [0.326, 0.707] (n 66.0, IBD 52.0) | 0.650 [0.491, 0.810] (n 65.0, IBD 46.0) |
| E5b heads, 50 epochs | 0.532 [0.322, 0.758] (n 66.0, IBD 57.0) | 0.401 [0.270, 0.545] (n 66.0, IBD 52.0) | 0.370 [0.233, 0.521] (n 65.0, IBD 46.0) |

**Halfvarson, covariates: richness + log10 depth**

| Source | M0: richness + depth | M2: score alone | M1: both | Δ M1 − M0 (score over simple signals) | Δ M1 − M2 (simple signals over score) | AUROC of residual score | Δ M1 − M0, members | residual, members |
|---|---|---|---|---|---|---|---|---|
| Pope's ensemble | 0.725 [0.659, 0.793] | 0.827 [0.767, 0.883] | 0.848 [0.798, 0.896] | 0.123 [0.059, 0.187] | 0.021 [-0.006, 0.053] | 0.754 [0.682, 0.819] | 0.116 [0.059, 0.137] | 0.740 [0.667, 0.804] |
| RF, relative abundance | 0.725 [0.659, 0.793] | 0.815 [0.747, 0.873] | 0.828 [0.759, 0.882] | 0.103 [0.049, 0.157] | 0.013 [-0.015, 0.041] | 0.771 [0.704, 0.832] | 0.097 [0.068, 0.116] | 0.759 [0.711, 0.795] |
| E5b heads, 1 epoch | 0.725 [0.659, 0.793] | 0.803 [0.736, 0.862] | 0.828 [0.772, 0.881] | 0.103 [0.041, 0.164] | 0.025 [-0.006, 0.063] | 0.737 [0.663, 0.801] | 0.103 [0.097, 0.108] | 0.737 [0.727, 0.741] |
| E5b heads, 50 epochs | 0.725 [0.659, 0.793] | 0.771 [0.699, 0.838] | 0.814 [0.756, 0.871] | 0.089 [0.036, 0.143] | 0.043 [-0.004, 0.092] | 0.747 [0.671, 0.818] | 0.089 [0.074, 0.096] | 0.749 [0.728, 0.756] |

_AUROC of the score inside richness tertiles (29-285, 285-478, 479-1479)_

| Source | tertile 1 | tertile 2 | tertile 3 |
|---|---|---|---|
| Pope's ensemble | 0.822 [0.705, 0.912] (n 188.0, IBD 180.0) | 0.874 [0.792, 0.944] (n 188.0, IBD 165.0) | 0.788 [0.675, 0.884] (n 188.0, IBD 165.0) |
| RF, relative abundance | 0.887 [0.788, 0.976] (n 188.0, IBD 180.0) | 0.804 [0.693, 0.896] (n 188.0, IBD 165.0) | 0.843 [0.727, 0.930] (n 188.0, IBD 165.0) |
| E5b heads, 1 epoch | 0.774 [0.661, 0.876] (n 188.0, IBD 180.0) | 0.853 [0.758, 0.933] (n 188.0, IBD 165.0) | 0.765 [0.645, 0.872] (n 188.0, IBD 165.0) |
| E5b heads, 50 epochs | 0.628 [0.442, 0.790] (n 188.0, IBD 180.0) | 0.847 [0.753, 0.930] (n 188.0, IBD 165.0) | 0.766 [0.657, 0.859] (n 188.0, IBD 165.0) |

**Halfvarson, covariates: n_taxa + log10 depth**

| Source | M0: richness + depth | M2: score alone | M1: both | Δ M1 − M0 (score over simple signals) | Δ M1 − M2 (simple signals over score) | AUROC of residual score | Δ M1 − M0, members | residual, members |
|---|---|---|---|---|---|---|---|---|
| Pope's ensemble | 0.712 [0.649, 0.775] | 0.827 [0.767, 0.883] | 0.845 [0.794, 0.893] | 0.133 [0.070, 0.194] | 0.018 [-0.008, 0.048] | 0.765 [0.696, 0.828] | 0.120 [0.061, 0.150] | 0.744 [0.666, 0.809] |
| RF, relative abundance | 0.712 [0.649, 0.775] | 0.815 [0.747, 0.873] | 0.839 [0.772, 0.892] | 0.127 [0.067, 0.182] | 0.024 [-0.004, 0.055] | 0.808 [0.743, 0.863] | 0.119 [0.083, 0.141] | 0.800 [0.744, 0.829] |
| E5b heads, 1 epoch | 0.712 [0.649, 0.775] | 0.803 [0.736, 0.862] | 0.822 [0.764, 0.877] | 0.110 [0.045, 0.170] | 0.019 [-0.009, 0.053] | 0.744 [0.670, 0.811] | 0.110 [0.103, 0.116] | 0.747 [0.734, 0.750] |
| E5b heads, 50 epochs | 0.712 [0.649, 0.775] | 0.771 [0.699, 0.838] | 0.807 [0.747, 0.863] | 0.095 [0.043, 0.151] | 0.036 [-0.008, 0.084] | 0.745 [0.666, 0.817] | 0.097 [0.078, 0.105] | 0.746 [0.729, 0.755] |

_AUROC of the score inside n_taxa tertiles (29-285, 285-478, 479-512)_

| Source | tertile 1 | tertile 2 | tertile 3 |
|---|---|---|---|
| Pope's ensemble | 0.822 [0.705, 0.912] (n 188.0, IBD 180.0) | 0.874 [0.792, 0.944] (n 188.0, IBD 165.0) | 0.788 [0.661, 0.883] (n 188.0, IBD 165.0) |
| RF, relative abundance | 0.887 [0.788, 0.976] (n 188.0, IBD 180.0) | 0.804 [0.693, 0.896] (n 188.0, IBD 165.0) | 0.843 [0.723, 0.934] (n 188.0, IBD 165.0) |
| E5b heads, 1 epoch | 0.774 [0.661, 0.876] (n 188.0, IBD 180.0) | 0.853 [0.758, 0.933] (n 188.0, IBD 165.0) | 0.765 [0.630, 0.868] (n 188.0, IBD 165.0) |
| E5b heads, 50 epochs | 0.628 [0.442, 0.790] (n 188.0, IBD 180.0) | 0.847 [0.753, 0.930] (n 188.0, IBD 165.0) | 0.766 [0.646, 0.858] (n 188.0, IBD 165.0) |

## SC-2: which taxa vs how many (HC4)

Pope's ensemble with K taxa kept per sample (E4d). Random-K: mean [min–max] over draws. Gap: paired bootstrap 95% CI (E4's preds).


**HMP2** (AUROC change per doubling of tokens: top: -0.028, rand: +0.023)

| K | Mean tokens | Top-K | Random-K | Random − top |
|---|---|---|---|---|
| 16 | 16 | 0.762 [0.762, 0.762] | 0.605 [0.539–0.653] | -0.157 [-0.270, -0.054] |
| 32 | 32 | 0.774 [0.774, 0.774] | 0.591 [0.550–0.622] | -0.183 [-0.281, -0.094] |
| 64 | 63 | 0.780 [0.780, 0.780] | 0.576 [0.570–0.594] | -0.204 [-0.306, -0.106] |
| 128 | 111 | 0.713 [0.713, 0.713] | 0.647 [0.631–0.658] | -0.066 [-0.133, -0.005] |
| 256 | 140 | 0.683 [0.683, 0.683] | 0.685 [0.685–0.685] | 0.002 [-0.003, 0.007] |
| 512 | 141 | 0.686 [0.686, 0.686] | – | – |

**Halfvarson** (AUROC change per doubling of tokens: top: +0.057, rand: +0.063)

| K | Mean tokens | Top-K | Random-K | Random − top |
|---|---|---|---|---|
| 16 | 16 | 0.613 [0.613, 0.613] | 0.565 [0.529–0.583] | -0.048 [-0.126, 0.026] |
| 32 | 32 | 0.621 [0.621, 0.621] | 0.619 [0.580–0.652] | -0.002 [-0.072, 0.068] |
| 64 | 64 | 0.690 [0.690, 0.690] | 0.696 [0.669–0.738] | 0.006 [-0.052, 0.066] |
| 128 | 123 | 0.785 [0.785, 0.785] | 0.729 [0.720–0.742] | -0.056 [-0.105, -0.007] |
| 256 | 227 | 0.825 [0.825, 0.825] | 0.813 [0.795–0.827] | -0.012 [-0.042, 0.016] |
| 512 | 350 | 0.830 [0.830, 0.830] | – | – |

## SC-3: do the RF and the ensemble rank samples alike (HC3)

Spearman ρ with bootstrap 95% CI. Yardstick: ρ between two members of the same source, median [min–max] over pairs. A ρ(ensemble, RF) inside the yardstick range means the two cannot be told apart by how they rank.

| Cohort | ensemble vs RF ra | ensemble vs RF pa | within ensemble | within RF ra |
|---|---|---|---|---|
| HMP2 | 0.712 [0.640, 0.766] | 0.842 [0.796, 0.875] | 0.904 [0.595–0.979] | 0.949 [0.935–0.965] |
| Halfvarson | 0.722 [0.671, 0.765] | 0.770 [0.726, 0.808] | 0.890 [0.598–0.968] | 0.967 [0.962–0.971] |

**ρ of each score with richness and depth**

| Source | HMP2: richness | HMP2: log10 depth | Halfvarson: richness | Halfvarson: log10 depth |
|---|---|---|---|---|
| Pope's ensemble | -0.477 [-0.584, -0.355] | -0.034 [-0.188, 0.117] | -0.575 [-0.634, -0.504] | -0.066 [-0.149, 0.020] |
| RF, relative abundance | -0.490 [-0.607, -0.357] | 0.010 [-0.139, 0.154] | -0.762 [-0.798, -0.720] | -0.101 [-0.185, -0.016] |
| RF, presence/absence | -0.425 [-0.548, -0.281] | 0.032 [-0.119, 0.184] | -0.714 [-0.755, -0.663] | -0.159 [-0.237, -0.077] |
| E5b heads, 1 epoch | -0.523 [-0.619, -0.413] | -0.136 [-0.285, 0.014] | -0.553 [-0.619, -0.480] | -0.073 [-0.156, 0.013] |
| E5b heads, 50 epochs | -0.431 [-0.540, -0.312] | -0.463 [-0.577, -0.338] | -0.273 [-0.353, -0.186] | 0.015 [-0.070, 0.100] |
