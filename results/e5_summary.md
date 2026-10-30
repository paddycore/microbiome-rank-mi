# E5 summary

_Generated 2026-10-30 16:09 by `src/e5_report.py` from `results/e5.csv`._

Probes: logistic regression, class-balanced, 5-fold stratified CV, balanced accuracy. Selectivity = score − score on permuted labels. Ensemble values: mean over the 10 models [min–max]. Layer 0 = encoder input; its [CLS] vector is identical for every sample, so [CLS] probes there sit at chance.

Baseline, cohort probe on [log10 depth, n_taxa] only: balanced accuracy 0.836, control 0.329, selectivity **0.507**.


## Pooling: mean, C = 0.1


**Cohort selectivity**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.599 [0.592–0.602] | 0.607 [0.597–0.613] | 0.606 [0.595–0.613] | 0.606 [0.599–0.614] | 0.617 [0.608–0.626] | 0.617 [0.609–0.627] |
| pretrained, before fine-tuning | 0.604 | 0.604 | 0.611 | 0.614 | 0.617 | 0.617 |
| untrained (control) | 0.596 | 0.610 | 0.631 | 0.626 | 0.633 | 0.628 |

**Cohort selectivity, depth + taxa count partialled out**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.481 [0.474–0.486] | 0.496 [0.491–0.501] | 0.501 [0.498–0.510] | 0.500 [0.492–0.509] | 0.497 [0.480–0.506] | 0.504 [0.494–0.514] |
| pretrained, before fine-tuning | 0.486 | 0.493 | 0.504 | 0.503 | 0.499 | 0.513 |
| untrained (control) | 0.490 | 0.496 | 0.510 | 0.515 | 0.515 | 0.523 |

**IBD selectivity inside AGP-IBD (leaky CV)**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.171 [0.169–0.175] | 0.172 [0.166–0.177] | 0.167 [0.157–0.175] | 0.173 [0.167–0.177] | 0.165 [0.154–0.170] | 0.162 [0.152–0.169] |
| pretrained, before fine-tuning | 0.168 | 0.172 | 0.166 | 0.179 | 0.170 | 0.175 |
| untrained (control) | 0.163 | 0.169 | 0.161 | 0.169 | 0.171 | 0.168 |

**Fresh linear head, AUROC on HMP2**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.590 [0.581–0.597] | 0.603 [0.588–0.623] | 0.543 [0.526–0.567] | 0.537 [0.504–0.572] | 0.530 [0.491–0.591] | 0.517 [0.482–0.554] |
| pretrained, before fine-tuning | 0.566 | 0.555 | 0.501 | 0.494 | 0.505 | 0.518 |
| untrained (control) | 0.579 | 0.577 | 0.493 | 0.487 | 0.492 | 0.500 |

**Fresh linear head, AUROC on Halfvarson**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.439 [0.423–0.447] | 0.495 [0.471–0.509] | 0.532 [0.492–0.551] | 0.537 [0.498–0.562] | 0.560 [0.470–0.638] | 0.568 [0.484–0.655] |
| pretrained, before fine-tuning | 0.442 | 0.495 | 0.538 | 0.509 | 0.525 | 0.510 |
| untrained (control) | 0.420 | 0.466 | 0.486 | 0.482 | 0.507 | 0.505 |

**log10 read depth, CV R²**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.790 [0.790–0.790] | 0.829 [0.826–0.830] | 0.850 [0.848–0.853] | 0.855 [0.852–0.857] | 0.859 [0.856–0.861] | 0.855 [0.853–0.859] |
| pretrained, before fine-tuning | 0.790 | 0.829 | 0.852 | 0.856 | 0.860 | 0.864 |
| untrained (control) | 0.788 | 0.823 | 0.842 | 0.849 | 0.854 | 0.855 |

**taxa count, CV R²**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.797 [0.797–0.797] | 0.822 [0.821–0.823] | 0.847 [0.844–0.852] | 0.856 [0.851–0.860] | 0.890 [0.882–0.897] | 0.898 [0.888–0.914] |
| pretrained, before fine-tuning | 0.797 | 0.826 | 0.857 | 0.867 | 0.904 | 0.925 |
| untrained (control) | 0.791 | 0.818 | 0.847 | 0.857 | 0.895 | 0.908 |

## Pooling: cls, C = 0.1


**Cohort selectivity**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.000 [0.000–0.000] | 0.583 [0.575–0.589] | 0.605 [0.599–0.612] | 0.602 [0.589–0.618] | 0.597 [0.581–0.621] | 0.616 [0.602–0.626] |
| pretrained, before fine-tuning | 0.000 | 0.584 | 0.588 | 0.597 | 0.614 | 0.621 |
| untrained (control) | 0.000 | 0.579 | 0.592 | 0.607 | 0.615 | 0.616 |

**Cohort selectivity, depth + taxa count partialled out**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.000 [0.000–0.000] | 0.463 [0.458–0.469] | 0.479 [0.466–0.492] | 0.482 [0.473–0.495] | 0.478 [0.467–0.487] | 0.485 [0.465–0.498] |
| pretrained, before fine-tuning | 0.000 | 0.466 | 0.465 | 0.469 | 0.490 | 0.507 |
| untrained (control) | 0.000 | 0.470 | 0.462 | 0.482 | 0.491 | 0.500 |

**IBD selectivity inside AGP-IBD (leaky CV)**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.000 [0.000–0.000] | 0.168 [0.160–0.183] | 0.173 [0.160–0.191] | 0.173 [0.167–0.179] | 0.185 [0.172–0.199] | 0.181 [0.165–0.195] |
| pretrained, before fine-tuning | 0.000 | 0.146 | 0.176 | 0.198 | 0.178 | 0.183 |
| untrained (control) | 0.000 | 0.161 | 0.175 | 0.184 | 0.179 | 0.177 |

**Fresh linear head, AUROC on HMP2**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.500 [0.500–0.500] | 0.592 [0.570–0.607] | 0.515 [0.495–0.533] | 0.502 [0.482–0.533] | 0.547 [0.509–0.576] | 0.549 [0.482–0.576] |
| pretrained, before fine-tuning | 0.500 | 0.581 | 0.578 | 0.534 | 0.550 | 0.501 |
| untrained (control) | 0.500 | 0.584 | 0.565 | 0.519 | 0.529 | 0.488 |

**Fresh linear head, AUROC on Halfvarson**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | 0.500 [0.500–0.500] | 0.546 [0.519–0.579] | 0.570 [0.549–0.625] | 0.607 [0.571–0.662] | 0.572 [0.546–0.610] | 0.548 [0.465–0.627] |
| pretrained, before fine-tuning | 0.500 | 0.583 | 0.601 | 0.580 | 0.553 | 0.699 |
| untrained (control) | 0.500 | 0.555 | 0.622 | 0.608 | 0.554 | 0.659 |

**log10 read depth, CV R²**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | -0.001 [-0.001–-0.001] | 0.792 [0.788–0.795] | 0.828 [0.823–0.832] | 0.836 [0.830–0.842] | 0.839 [0.833–0.847] | 0.841 [0.837–0.845] |
| pretrained, before fine-tuning | -0.001 | 0.793 | 0.844 | 0.844 | 0.858 | 0.860 |
| untrained (control) | -0.001 | 0.789 | 0.840 | 0.840 | 0.856 | 0.860 |

**taxa count, CV R²**

| Model | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| Pope ensemble (fine-tuned, n=10) | -0.003 [-0.003–-0.003] | 0.802 [0.799–0.806] | 0.836 [0.831–0.839] | 0.838 [0.832–0.842] | 0.847 [0.840–0.857] | 0.874 [0.862–0.884] |
| pretrained, before fine-tuning | -0.003 | 0.810 | 0.876 | 0.878 | 0.931 | 0.938 |
| untrained (control) | -0.003 | 0.800 | 0.857 | 0.861 | 0.921 | 0.929 |

## Probe capacity (mean pooling): cohort selectivity by C

| Model | C | input | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|---|
| ens1 | 0.01 | 0.579 | 0.596 | 0.599 | 0.594 | 0.614 | 0.609 |
| ens1 | 0.1 | 0.600 | 0.612 | 0.606 | 0.604 | 0.608 | 0.610 |
| ens1 | 1.0 | 0.622 | 0.626 | 0.618 | 0.616 | 0.620 | 0.618 |
| pre120 | 0.01 | 0.582 | 0.593 | 0.596 | 0.605 | 0.617 | 0.620 |
| pre120 | 0.1 | 0.604 | 0.604 | 0.611 | 0.614 | 0.617 | 0.617 |
| pre120 | 1.0 | 0.623 | 0.632 | 0.631 | 0.634 | 0.648 | 0.647 |
| pre0 | 0.01 | 0.583 | 0.601 | 0.596 | 0.608 | 0.616 | 0.622 |
| pre0 | 0.1 | 0.596 | 0.610 | 0.631 | 0.626 | 0.633 | 0.628 |
| pre0 | 1.0 | 0.619 | 0.627 | 0.632 | 0.641 | 0.646 | 0.642 |
