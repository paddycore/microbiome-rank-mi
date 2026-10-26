# E4 summary

_Generated 2026-10-22 11:03 by `src/e4_report.py` from `results/e4.csv`, `results/preds/e4_*.npz`, `results/e4c.csv`._

Model: Pope et al.'s 10-model fine-tuned ensemble (`ensemble.zip`), trained on AGP-IBD; test cohorts are cross-study. Ensemble score = mean P(IBD) over the 10 models (as in Pope's `begin.py`). ΔAUROC = condition minus Pope's order, paired bootstrap on shared resamples (1000); for repeated conditions the condition's AUROC is the mean over repeats. CIs ignore repeated samples per patient.

## S4-0 reproduction gate

| Cohort | Pope et al. | Ours, Pope's order [95% CI] | Ours, Pope's 10% test dropout (mean of 10) |
|---|---|---|---|
| HMP2 | 0.682 | 0.686 [0.582, 0.774] | 0.678 |
| Halfvarson | 0.805 | 0.830 [0.769, 0.886] | 0.824 |

## E4a: token order at test time

| Condition | Test set | Repeats | AUROC (mean) | Range over repeats | ΔAUROC vs Pope's order [95% CI] | Reading |
|---|---|---|---|---|---|---|
| Pope's order | HMP2 | 1 | 0.686 |  |  |  |
| Pope's order | Halfvarson | 1 | 0.830 |  |  |  |
| shuffle within tied counts (noise floor) | HMP2 | 20 | 0.686 | 0.686–0.687 | +0.001 [-0.001, +0.002] | CI includes 0 |
| shuffle within tied counts (noise floor) | Halfvarson | 20 | 0.830 | 0.830–0.830 | +0.000 [-0.001, +0.001] | CI includes 0 |
| shuffle all taxa | HMP2 | 20 | 0.705 | 0.697–0.714 | +0.020 [+0.009, +0.033] | above baseline (CI excludes 0) |
| shuffle all taxa | Halfvarson | 20 | 0.828 | 0.825–0.831 | -0.002 [-0.005, +0.001] | CI includes 0 |
| reverse (least abundant first) | HMP2 | 1 | 0.724 |  | +0.039 [+0.019, +0.062] | above baseline (CI excludes 0) |
| reverse (least abundant first) | Halfvarson | 1 | 0.828 |  | -0.002 [-0.008, +0.004] | CI includes 0 |
| Pope's test-time dropout 10% | HMP2 | 10 | 0.678 | 0.664–0.692 | -0.008 [-0.017, +0.000] | CI includes 0 |
| Pope's test-time dropout 10% | Halfvarson | 10 | 0.824 | 0.811–0.830 | -0.005 [-0.012, +0.000] | CI includes 0 |

## E4d: token budget (top-K most abundant taxa)

| K | Test set | Mean taxa kept | AUROC [95% CI] | ΔAUROC vs all taxa [95% CI] | Reading |
|---|---|---|---|---|---|
| 16 | HMP2 | 16 | 0.762 [0.678, 0.848] | +0.076 [-0.009, +0.178] | CI includes 0 |
| 16 | Halfvarson | 16 | 0.613 [0.531, 0.691] | -0.217 [-0.296, -0.132] | below baseline (CI excludes 0) |
| 32 | HMP2 | 32 | 0.774 [0.690, 0.848] | +0.088 [+0.018, +0.168] | above baseline (CI excludes 0) |
| 32 | Halfvarson | 32 | 0.621 [0.539, 0.696] | -0.208 [-0.283, -0.134] | below baseline (CI excludes 0) |
| 64 | HMP2 | 63 | 0.780 [0.701, 0.853] | +0.094 [+0.041, +0.157] | above baseline (CI excludes 0) |
| 64 | Halfvarson | 64 | 0.690 [0.619, 0.759] | -0.140 [-0.203, -0.080] | below baseline (CI excludes 0) |
| 128 | HMP2 | 111 | 0.713 [0.614, 0.800] | +0.027 [+0.003, +0.055] | above baseline (CI excludes 0) |
| 128 | Halfvarson | 123 | 0.785 [0.729, 0.839] | -0.045 [-0.091, -0.001] | below baseline (CI excludes 0) |
| 256 | HMP2 | 140 | 0.683 [0.579, 0.773] | -0.002 [-0.006, -0.000] | below baseline (CI excludes 0) |
| 256 | Halfvarson | 227 | 0.825 [0.774, 0.877] | -0.005 [-0.029, +0.023] | CI includes 0 |
| 512 | HMP2 | 141 | 0.686 [0.582, 0.774] | +0.000 [+0.000, +0.000] | CI includes 0 |
| 512 | Halfvarson | 350 | 0.830 [0.769, 0.886] | +0.000 [+0.000, +0.000] | CI includes 0 |

### Control: K taxa chosen at random (5 draws per K, kept in Pope's order)

| K | Test set | AUROC (mean) | Range over draws | ΔAUROC vs all taxa [95% CI] | ΔAUROC vs top-K at same K [95% CI] |
|---|---|---|---|---|---|
| 16 | HMP2 | 0.605 | 0.539–0.653 | -0.081 [-0.170, +0.009] | -0.157 [-0.270, -0.054] |
| 16 | Halfvarson | 0.565 | 0.529–0.583 | -0.265 [-0.324, -0.203] | -0.048 [-0.126, +0.026] |
| 32 | HMP2 | 0.591 | 0.550–0.622 | -0.095 [-0.181, -0.014] | -0.183 [-0.281, -0.094] |
| 32 | Halfvarson | 0.619 | 0.580–0.652 | -0.211 [-0.261, -0.155] | -0.002 [-0.072, +0.068] |
| 64 | HMP2 | 0.576 | 0.570–0.594 | -0.110 [-0.198, -0.024] | -0.204 [-0.306, -0.106] |
| 64 | Halfvarson | 0.696 | 0.669–0.738 | -0.134 [-0.181, -0.080] | +0.006 [-0.052, +0.066] |
| 128 | HMP2 | 0.647 | 0.631–0.658 | -0.039 [-0.092, +0.011] | -0.066 [-0.133, -0.005] |
| 128 | Halfvarson | 0.729 | 0.720–0.742 | -0.101 [-0.145, -0.055] | -0.056 [-0.105, -0.007] |
| 256 | HMP2 | 0.685 | 0.685–0.685 | -0.001 [-0.005, +0.002] | +0.002 [-0.002, +0.007] |
| 256 | Halfvarson | 0.813 | 0.795–0.827 | -0.016 [-0.044, +0.014] | -0.012 [-0.042, +0.016] |

## E4c: order-free baselines (absolute AUROC, not paired)

| Model | HMP2 | Halfvarson |
|---|---|---|
| Transformer, Pope's order | 0.686 [0.582, 0.774] | 0.830 [0.769, 0.886] |
| Transformer, shuffled (20 perms) | 0.705 [0.606, 0.797] | 0.828 [0.769, 0.882] |
| Bag of embeddings, unweighted, LR balanced | 0.581 [0.484, 0.670] | 0.398 [0.335, 0.464] |
| Bag of embeddings, unweighted, LR unweighted | 0.551 [0.459, 0.641] | 0.425 [0.359, 0.495] |
| Bag of embeddings, weighted, LR balanced | 0.551 [0.457, 0.651] | 0.651 [0.578, 0.727] |
| Bag of embeddings, weighted, LR unweighted | 0.513 [0.432, 0.604] | 0.693 [0.627, 0.761] |
| RF relative abundance (Phase 0) | 0.736 [0.646, 0.815] | 0.818 [0.751, 0.876] |
| RF presence/absence (Phase 0) | 0.728 [0.645, 0.806] | 0.814 [0.761, 0.863] |

Phase 0 RF rows: spec RF (`class_weight='balanced'`), median over 5 seeds, full 1,897-column input (not the top-512 input the transformer sees).

## E4b: head trained on frozen-encoder features, train order x test order

Encoder `5head5layer_epoch120_disc` frozen; logistic regression head on AGP-IBD; shuffled orders: mean over seeds [0, 1, 2, 3, 4] (range in brackets). Last column: the same features and head under 5-fold CV inside AGP-IBD (not patient-grouped, so a leaky upper bound).

| Feature | class_weight | Test set | sorted → sorted | sorted → shuffled | shuffled → sorted | shuffled → shuffled | AGP 5-fold (leaky) |
|---|---|---|---|---|---|---|---|
| cls | balanced | HMP2 | 0.495 | 0.483 [0.466–0.497] | 0.490 [0.481–0.495] | 0.495 [0.458–0.519] | 0.758 |
| cls | balanced | Halfvarson | 0.644 | 0.636 [0.630–0.642] | 0.618 [0.607–0.629] | 0.606 [0.600–0.613] | 0.758 |
| cls | None | HMP2 | 0.427 | 0.430 [0.420–0.438] | 0.414 [0.406–0.423] | 0.425 [0.398–0.439] | 0.752 |
| cls | None | Halfvarson | 0.666 | 0.661 [0.657–0.666] | 0.658 [0.654–0.663] | 0.649 [0.641–0.655] | 0.752 |
| mean | balanced | HMP2 | 0.539 | 0.549 [0.543–0.557] | 0.490 [0.479–0.504] | 0.530 [0.521–0.542] | 0.768 |
| mean | balanced | Halfvarson | 0.416 | 0.413 [0.410–0.414] | 0.363 [0.338–0.395] | 0.363 [0.339–0.397] | 0.768 |
| mean | None | HMP2 | 0.470 | 0.496 [0.489–0.504] | 0.422 [0.418–0.431] | 0.461 [0.457–0.471] | 0.769 |
| mean | None | Halfvarson | 0.470 | 0.467 [0.464–0.470] | 0.449 [0.433–0.470] | 0.447 [0.428–0.473] | 0.769 |
