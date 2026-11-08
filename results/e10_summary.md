# E10 summary: alignment of disease directions vs transfer

16 space types, 159 feature spaces; cos with a null from 200 test-label shuffles.

## Spearman ρ between alignment and transfer, over feature spaces

| Eval set | class-mean difference | LR C = 1e-4 | LR C = 1 | separation vs LR C = 1 |
|---|---|---|---|---|
| AGP-IBD hold-out | 0.12 | 0.12 | 0.12 | 0.37 |
| HMP2 | 0.78 | 0.71 | -0.18 | -0.30 |
| Halfvarson | 0.74 | 0.40 | 0.35 | -0.10 |

## By family: median cos [min–max], share of spaces above the null, median AUROC (mean diff / LR C = 1)

| Family | Eval set | cos | above null | AUROC mean diff | AUROC LR C=1 |
|---|---|---|---|---|---|
| Micro16S | AGP-IBD hold-out | 0.59 [0.46–0.63] | 67% | 0.591 | 0.652 |
| Micro16S | HMP2 | -0.58 [-0.68–-0.44] | 0% | 0.333 | 0.484 |
| Micro16S | Halfvarson | 0.08 [-0.02–0.13] | 0% | 0.529 | 0.565 |
| Pope fine-tuned | AGP-IBD hold-out | 0.95 [0.86–0.99] | 100% | 0.693 | 0.756 |
| Pope fine-tuned | HMP2 | 0.50 [-0.08–0.71] | 15% | 0.700 | 0.552 |
| Pope fine-tuned | Halfvarson | 0.30 [-0.33–0.58] | 8% | 0.746 | 0.486 |
| Pope pretrained / untrained | AGP-IBD hold-out | 0.86 [0.66–0.88] | 100% | 0.627 | 0.756 |
| Pope pretrained / untrained | HMP2 | 0.22 [-0.13–0.29] | 0% | 0.606 | 0.548 |
| Pope pretrained / untrained | Halfvarson | 0.13 [-0.11–0.28] | 0% | 0.663 | 0.471 |

## Micro16S spaces

```
model         test_set    cos  null_lo  null_hi  auroc_meandiff  auroc_lr_strong  auroc_lr_weak
  pre             HMP2 -0.575   -0.533    0.542           0.333            0.347          0.527
  pre       Halfvarson -0.023   -0.552    0.622           0.506            0.550          0.511
  pre AGP-IBD hold-out  0.465   -0.433    0.373           0.622            0.636          0.662
 rand             HMP2 -0.437   -0.442    0.417           0.270            0.270          0.479
 rand       Halfvarson  0.134   -0.407    0.365           0.612            0.621          0.627
 rand AGP-IBD hold-out  0.633   -0.554    0.573           0.591            0.600          0.652
  bag             HMP2 -0.680   -0.745    0.708           0.396            0.387          0.484
  bag       Halfvarson  0.080   -0.719    0.738           0.529            0.633          0.565
  bag AGP-IBD hold-out  0.587   -0.648    0.705           0.578            0.627          0.649
```

## Pope ens1, last layer (for comparison)

```
   space         test_set   cos  null_hi  auroc_meandiff  auroc_lr_weak
 mean L5             HMP2 0.617    0.622           0.733          0.464
 mean L5       Halfvarson 0.360    0.528           0.783          0.613
 mean L5 AGP-IBD hold-out 0.955    0.588           0.701          0.757
  cls L5             HMP2 0.440    0.706           0.642          0.442
  cls L5       Halfvarson 0.422    0.538           0.776          0.421
  cls L5 AGP-IBD hold-out 0.967    0.755           0.680          0.731
abund L5             HMP2 0.408    0.444           0.714          0.573
abund L5       Halfvarson 0.136    0.317           0.632          0.656
abund L5 AGP-IBD hold-out 0.910    0.467           0.697          0.721
```
