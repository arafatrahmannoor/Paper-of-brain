"""Paper figures and statistics from the grouped-CV out-of-fold predictions.

Usage: python tools/make_grouped_figures.py [ablation]   (default: full)

Writes paper/figures/fig_confusion.pdf and paper/figures/fig4_reliability.pdf
and prints the extra statistics quoted in the manuscript: group-bootstrap CIs,
error pairs, the low-confidence triage numbers and gate / path weights.
"""
import json
import sys
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
LABELS = ["Glioma", "Meningioma", "No tumour", "Pituitary"]


def ece_of(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for b in range(bins):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def macro_f1(y, yp):
    f = []
    for k in range(4):
        tp = np.sum((y == k) & (yp == k))
        p = tp / max(np.sum(yp == k), 1)
        r = tp / max(np.sum(y == k), 1)
        f.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f))


def main(ablation="full"):
    df = pd.concat([pd.read_csv(RES / f"grouped_{ablation}_fold_{k}_predictions.csv").assign(fold=k)
                    for k in range(1, 6)], ignore_index=True)
    P = df[[f"prob_{c}" for c in CLASSES]].to_numpy()
    y = df.true_index.to_numpy()
    yp = P.argmax(1)
    conf = P.max(1)
    correct = yp == y
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, yp), 1)

    # ---- confusion matrix figure -------------------------------------------------
    rowpct = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.imshow(np.maximum(rowpct, 0.05), cmap="Blues", norm=LogNorm(vmin=0.05, vmax=100))
    for i in range(4):
        for j in range(4):
            dark = rowpct[i, j] > 30
            ax.text(j, i, f"{cm[i, j]}\n{rowpct[i, j]:.1f}%", ha="center", va="center",
                    fontsize=7.5, color="white" if dark else "black")
    ax.set_xticks(range(4), LABELS, fontsize=7.5, rotation=20)
    ax.set_yticks(range(4), LABELS, fontsize=7.5)
    ax.set_xlabel("Predicted class", fontsize=8)
    ax.set_ylabel("True class", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion.pdf")
    plt.close(fig)

    # ---- reliability diagram -------------------------------------------------------
    edges = np.linspace(0, 1, 11)
    mids, accs, confs, counts = [], [], [], []
    for b in range(10):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if m.any():
            mids.append((edges[b] + edges[b + 1]) / 2)
            accs.append(correct[m].mean())
            confs.append(conf[m].mean())
            counts.append(int(m.sum()))
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.plot([0, 1], [0, 1], "--", color="0.5", lw=0.8, label="Perfect calibration")
    ax.bar(mids, accs, width=0.09, color="#4a7ab7", edgecolor="black", lw=0.4, label="Accuracy")
    ax.plot(mids, confs, "o", color="#c0392b", ms=3.5, label="Mean confidence in bin")
    for mx, a, n in zip(mids, accs, counts):
        ax.text(mx, min(a + 0.03, 1.02), f"{n}", ha="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Confidence", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_reliability.pdf")
    plt.close(fig)

    # ---- statistics ---------------------------------------------------------------
    print(f"n={len(df)} errors={int((~correct).sum())} acc={correct.mean():.4f} "
          f"macroF1={macro_f1(y, yp):.4f} ece={ece_of(conf, correct):.4f}")
    groups = df.group.to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(0)
    stats = {"acc": [], "f1": [], "ece": [], "brier": []}
    onehot = np.eye(4)[y]
    brier_i = ((P - onehot) ** 2).sum(1)
    for _ in range(2000):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        stats["acc"].append(correct[idx].mean())
        stats["f1"].append(macro_f1(y[idx], yp[idx]))
        stats["ece"].append(ece_of(conf[idx], correct[idx]))
        stats["brier"].append(brier_i[idx].mean())
    for k, v in stats.items():
        print(f"group-bootstrap 95% CI {k}: [{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]")

    pairs = {}
    for i in range(4):
        for j in range(i + 1, 4):
            pairs[f"{CLASSES[i]}-{CLASSES[j]}"] = int(cm[i, j] + cm[j, i])
    print("error pairs:", pairs)
    tum = y != 2
    print(f"tumour->notumor: {int(((yp == 2) & tum).sum())} of {int(tum.sum())}; "
          f"notumor->tumour: {int(((yp != 2) & ~tum).sum())} of {int((~tum).sum())}")
    low = conf < 0.9
    print(f"conf<0.9: n={int(low.sum())} ({low.mean():.1%}) acc={correct[low].mean():.4f} "
          f"errors={int((~correct & low).sum())}; errors with conf>=0.95: {int((~correct & (conf >= 0.95)).sum())}")
    top = conf > 0.9
    print(f"top bin: n={int(top.sum())} acc={correct[top].mean():.4f} conf={conf[top].mean():.4f} max={conf.max():.4f}")

    if df.w_image.notna().any():
        print(f"w_image: mean {df.w_image.mean():.3f}, median {df.w_image.median():.3f}, "
              f"share>0.5 {(df.w_image > 0.5).mean():.3f}, correct {df.w_image[correct].mean():.3f}, "
              f"wrong {df.w_image[~correct].mean():.3f}")
    for k in range(1, 6):
        m = json.loads((RES / f"grouped_{ablation}_fold_{k}_metrics.json").read_text())
        if "softmax_alpha" in m:
            print(f"fold {k}: alpha={np.round(m['softmax_alpha'], 3).tolist()} "
                  f"beta={np.round(m['softmax_beta'], 3).tolist()} tau={m.get('tau', float('nan')):.3f} "
                  f"best_epoch={m['best_epoch']} epochs_run={m['epochs_run']}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
