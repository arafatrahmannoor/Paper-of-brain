"""Four-model comparison on the grouped folds: confusion matrices and per-fold curves.

Usage: python tools/make_ablation_figures.py

Writes paper/figures/fig_cm_models.pdf (2x2 pooled out-of-fold confusion matrices)
and paper/figures/fig_cv_folds.pdf (per-fold accuracy, macro-F1 and ECE, one line per
model), and prints the summary rows and McNemar tests used in the manuscript.
A model whose runs are missing gets a "results pending" panel.
"""
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "grouped_cv"
FIG = ROOT / "paper" / "figures"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABELS = ["Glioma", "Mening.", "No tumour", "Pituitary"]
MODELS = [
    ("image_only", "Baseline: image only", "#4a7ab7"),
    ("graph_only", "Variant A: graph only", "#7f7f7f"),
    ("concat", "Variant B: concatenation", "#d08c2f"),
    ("full", "Proposed: full model", "#b03a2e"),
]


def load(tag):
    folds, preds = {}, []
    for k in range(1, 6):
        p = RES / f"grouped_{tag}_fold_{k}_predictions.csv"
        if p.exists():
            folds[k] = json.loads((RES / f"grouped_{tag}_fold_{k}_metrics.json").read_text())
            preds.append(pd.read_csv(p))
    return folds, (pd.concat(preds, ignore_index=True) if preds else None)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    data = {tag: load(tag) for tag, _, _ in MODELS}

    # ---- 2x2 confusion matrices --------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.4))
    for ax, (tag, name, _) in zip(axes.ravel(), MODELS):
        folds, df = data[tag]
        if df is None:
            ax.text(0.5, 0.5, "results pending", ha="center", va="center", fontsize=9, color="0.4")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(name, fontsize=9)
            continue
        cm = np.zeros((4, 4), int)
        np.add.at(cm, (df.true_index.to_numpy(), df.pred_index.to_numpy()), 1)
        rowpct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(np.maximum(rowpct, 0.05), cmap="Blues", norm=LogNorm(vmin=0.05, vmax=100))
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm[i, j]}\n{rowpct[i, j]:.1f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if rowpct[i, j] > 30 else "black")
        ax.set_xticks(range(4), LABELS, fontsize=6.5)
        ax.set_yticks(range(4), LABELS, fontsize=6.5)
        acc = np.trace(cm) / cm.sum() * 100
        part = "" if len(folds) == 5 else f", folds {','.join(map(str, folds))}"
        ax.set_title(f"{name}\n(acc. {acc:.2f}%{part})", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("Predicted class", fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("True class", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_cm_models.pdf")
    plt.close(fig)

    # ---- per-fold line charts ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for metric, ax, ylabel, scale in [("accuracy", axes[0], "Accuracy (%)", 100),
                                      ("macro_f1", axes[1], "Macro-F1 (%)", 100),
                                      ("ece", axes[2], "ECE", 1)]:
        for tag, name, color in MODELS:
            folds = data[tag][0]
            if folds:
                ks = sorted(folds)
                ax.plot(ks, [folds[k][metric] * scale for k in ks], "o-", color=color, ms=3.5, lw=1.2,
                        label=name.split(": ")[1])
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Fold", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "fig_cv_folds.pdf")
    plt.close(fig)

    # ---- summary rows and McNemar tests against the full model -------------------------
    full_df = data["full"][1]
    for tag, name, _ in MODELS:
        folds, df = data[tag]
        if not folds:
            print(f"{name}: pending")
            continue
        acc = np.array([folds[k]["accuracy"] for k in sorted(folds)]) * 100
        f1 = np.array([folds[k]["macro_f1"] for k in sorted(folds)]) * 100
        ece = np.array([folds[k]["ece"] for k in sorted(folds)])
        sd = (lambda x: x.std(ddof=1) if len(x) > 1 else 0.0)
        pooled = (df.true_index == df.pred_index).mean() * 100
        line = (f"{name}: folds {sorted(folds)} acc {acc.mean():.2f} ± {sd(acc):.2f} (pooled {pooled:.2f}) "
                f"F1 {f1.mean():.2f} ± {sd(f1):.2f} ECE {ece.mean():.3f} ± {sd(ece):.3f}")
        if tag != "full" and full_df is not None:
            m = df.merge(full_df, on="sample_index", suffixes=("", "_full"))
            ok, ok_full = m.true_index == m.pred_index, m.true_index_full == m.pred_index_full
            b, c = int((ok_full & ~ok).sum()), int((~ok_full & ok).sum())
            line += f" | vs full on {len(m)} paired images: full-only correct {b}, {tag}-only correct {c}, " \
                    f"McNemar exact p = {mcnemar_exact(b, c):.3g}"
        print(line)


if __name__ == "__main__":
    main()
