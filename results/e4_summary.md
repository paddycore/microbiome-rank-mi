# E4 summary

_Generated 2026-10-21 02:20 by `src/e4_report.py` from `results/e4.csv`, `results/preds/e4_*.npz`, `results/e4c.csv`._

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
