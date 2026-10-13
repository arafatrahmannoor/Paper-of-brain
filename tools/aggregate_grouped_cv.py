"""Verify and pool the v2 grouped-CV runs in results/grouped_cv/.

Usage: python tools/aggregate_grouped_cv.py [ablation]   (default: full)

For every fold with a {cv}_{ablation}_fold_{k}_predictions.csv, this script:
- checks the predictions belong to test fold k;
- checks that fold shares no near-duplicate groups with its training data;
- recomputes accuracy and compares it with the run's metrics.json.

It then prints per-fold metrics, mean ± SD, the pooled confusion matrix and
per-class recall, and gate / GNN path-weight summaries.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "grouped_cv"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def main(ablation="full", cv="grouped"):
    groups = pd.read_csv(ROOT / "groups.csv")
    folds = pd.read_csv(ROOT / "fold_assignment_grouped.csv").merge(
        groups[["sample_index", "group"]], on="sample_index")

    rows, preds = [], []
    for k in range(1, 6):
        p_path = ROOT / f"{cv}_{ablation}_fold_{k}_predictions.csv"
        if not p_path.exists():
            continue
        p = pd.read_csv(p_path)
        m = json.loads((ROOT / f"{cv}_{ablation}_fold_{k}_metrics.json").read_text())

        test = folds[folds.test_fold == k]
        assert set(p.sample_index) == set(test.sample_index), f"fold {k}: predictions are not test fold {k}"
        shared = set(test.group) & set(folds[folds.test_fold != k].group)
        assert not shared, f"fold {k}: {len(shared)} groups shared with training"

        acc = float((p.true_index == p.pred_index).mean())
        assert abs(acc - m["accuracy"]) < 1e-9, f"fold {k}: CSV acc {acc} != json {m['accuracy']}"

        row = {
            "fold": k, "n": len(p), "errors": int((p.true_index != p.pred_index).sum()),
            "acc": m["accuracy"], "macro_p": m["macro_precision"], "macro_r": m["macro_recall"],
            "macro_f1": m["macro_f1"], "ece": m["ece"], "brier": m["brier"],
            "best_epoch": m.get("best_epoch"), "epochs_run": m.get("epochs_run"),
        }
        if "w_image" in p and p.w_image.notna().any():
            row["w_image_mean"] = p.w_image.mean()
            row["tau"] = m.get("tau")
        rows.append(row)
        preds.append(p)
        if m.get("softmax_alpha"):
            print(f"fold {k}: alpha={np.round(m['softmax_alpha'], 3)} beta={np.round(m['softmax_beta'], 3)}")

    if not rows:
        sys.exit(f"no {cv}_{ablation} predictions found in {ROOT}")

    df = pd.DataFrame(rows).set_index("fold")
    pd.set_option("display.width", 160)
    print(f"\n{cv}_{ablation}: {len(df)} fold(s), all checks passed\n")
    print(df.round(4).to_string())
    metric_cols = ["acc", "macro_p", "macro_r", "macro_f1", "ece", "brier"]
    print("\nmean ± SD (across folds):")
    for c in metric_cols:
        print(f"  {c:9s} {df[c].mean():.4f} ± {df[c].std(ddof=1) if len(df) > 1 else 0:.4f}")

    allp = pd.concat(preds)
    cm = np.zeros((4, 4), int)
    for t, q in zip(allp.true_index, allp.pred_index):
        cm[t, q] += 1
    print(f"\npooled: n={cm.sum()} errors={cm.sum() - np.trace(cm)} acc={np.trace(cm) / cm.sum():.4f}")
    print("confusion (rows true, cols predicted):", ", ".join(CLASSES))
    for name, r in zip(CLASSES, cm):
        print(f"  {name:11s}" + "".join(f"{v:6d}" for v in r))
    print("recall    :", np.round(cm.diagonal() / cm.sum(1), 4))
    print("precision :", np.round(cm.diagonal() / cm.sum(0), 4))

    # Errors cluster inside near-duplicate groups (offline-augmented copies of one
    # source image), so resample groups, not images, for the confidence interval.
    allp["err"] = allp.true_index != allp.pred_index
    per_group = allp.groupby("group").err.agg(["sum", "size"]).to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(per_group), size=(5000, len(per_group)))
    boot = 1 - per_group[idx, 0].sum(1) / per_group[idx, 1].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    n_bad = int((per_group[:, 0] > 0).sum())
    print(f"\ngroup-cluster bootstrap 95% CI for pooled acc: [{lo:.4f}, {hi:.4f}]")
    print(f"errors fall in {n_bad} of {len(per_group)} groups; "
          f"largest 5 groups hold {int(np.sort(per_group[:, 0])[-5:].sum())} errors")


if __name__ == "__main__":
    main(*sys.argv[1:2])
