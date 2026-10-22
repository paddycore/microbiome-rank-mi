"""E4b step 2: train a head on frozen-encoder features in one token order, test in another (the 2x2).

E4a showed that shuffling Pope's ensemble's input at TEST time does not lower AUROC. That leaves two readings:
order carries no transferable information, or it does but the fine-tuned model ignores it. Here the head is
trained from scratch on features of the frozen encoder (src/e4b_features.py), once with Pope's order and once with
shuffled order, and each head is tested in both orders:

                      test sorted          test shuffled
  train sorted        Pope-like setting    = E4a with our head
  train shuffled      order-free head      order never seen

"""
import csv
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pope_model as pm
from _paths import RESULTS
from e4b_features import load_features
from evaluate import bootstrap_ci

E4 = pm.CFG["e4"]
FEATURES = ("cls", "mean")
CLASS_WEIGHTS = ("balanced", None)
ORDERS = ("sorted", "shuffled")
CSV_PATH = RESULTS / "e4b.csv"
PRED_DIR = RESULTS / "preds"
COLUMNS = ["feature", "class_weight", "train_order", "test_order", "seed", "test_set", "auroc", "auroc_ci_lo",
           "auroc_ci_hi", "auprc", "n_train", "n_test", "timestamp"]


def fit_head(X, y, class_weight):
    """Fit and return make_pipeline(StandardScaler(), LogisticRegression(class_weight=class_weight, max_iter=5000)).

    """
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=class_weight, max_iter=5000)
    )
    return pipe.fit(X,y)


def cell(train_order, test_order, feature, class_weight, seed):
    """One cell of the 2x2: fit on AGP-IBD in `train_order`, predict each test cohort in `test_order`.

    feature      "cls" or "mean": which of load_features' two arrays to use
    seed         which random permutation to load for a `shuffled` order (ignored for `sorted`)
    returns      {cohort: (y, P(IBD))} for every cohort in E4["test_cohorts"] ("hmp2", "hv")

    """
    cls_tr, mean_tr, y_tr = load_features("agp_ibd", train_order, seed)
    X_tr = cls_tr if feature == "cls" else mean_tr

    model = fit_head(X_tr, y_tr, class_weight)

    results = {}
    for cohort in E4["test_cohorts"]:
        cls_te, mean_te, y_te = load_features(cohort, test_order, seed)
        X_te = cls_te if feature == "cls" else mean_te

        p = model.predict_proba(X_te)[:, 1]
        results[cohort] = (y_te, p)

    return results


def main():
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    n_train = len(load_features("agp_ibd", "sorted")[2])
    rows = []
    for feature in FEATURES:
        for cw in CLASS_WEIGHTS:
            for tr in ORDERS:
                for te in ORDERS:
                    seeds = [0] if tr == te == "sorted" else E4["e4b_shuffle_seeds"]
                    for seed in seeds:
                        for cohort, (y, p) in cell(tr, te, feature, cw, seed).items():
                            (lo, hi), _ = bootstrap_ci(y, p, seed=0)
                            np.savez_compressed(PRED_DIR / f"e4b_{feature}_{cw}_{tr}-{te}_s{seed}_{cohort}.npz", y=y, p=p)
                            rows.append({"feature": feature, "class_weight": str(cw), "train_order": tr,
                                         "test_order": te, "seed": seed, "test_set": pm.CFG["cohorts"][cohort]["name"],
                                         "auroc": round(roc_auc_score(y, p), 4), "auroc_ci_lo": round(lo, 4),
                                         "auroc_ci_hi": round(hi, 4), "auprc": round(average_precision_score(y, p), 4),
                                         "n_train": n_train, "n_test": len(y), "timestamp": stamp})
                            r = rows[-1]
                            print(f"{feature:4s} {str(cw):8s} train {tr:8s} test {te:8s} s{seed} {r['test_set']:11s} "
                                  f"AUROC {r['auroc']:.4f} [{lo:.3f}, {hi:.3f}]", flush=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
