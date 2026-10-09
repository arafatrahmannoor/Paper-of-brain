# Usage: python3 tools/final_cv_summary.py OUT_DIR fold_1_predictions.csv ... fold_5_predictions.csv
# Final five-fold summary from the per-image prediction files (pass them in fold order):
#   - per-fold metrics and mean +/- sample SD (ddof=1)
#   - pooled out-of-fold (OOF) metrics: accuracy, macro P/R/F1, per-class P/R/F1/specificity, ECE, Brier, NLL
#   - 95% bootstrap CIs, both image-level and cluster-level (all augmented copies of one slice resampled together)
#   - error rate per filename family
#   - reliability diagram of the pooled OOF predictions (OUT_DIR/reliability_oof.png)
import csv, sys, os, re, math, collections
import numpy as np

C = ["glioma", "meningioma", "notumor", "pituitary"]
out_dir, files = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)

def load(p):
    rows = list(csv.DictReader(open(p)))
    y = np.array([int(r["true_index"]) for r in rows])
    P = np.array([[float(r["prob_" + c]) for c in C] for r in rows])
    names = [os.path.basename(r["path"]) for r in rows]
    return rows, y, P, names

def metrics(y, P):
    yp = P.argmax(1)
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (y, yp), 1)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    spec = np.array([(cm.sum() - cm[k].sum() - cm[:, k].sum() + cm[k, k]) / (cm.sum() - cm[k].sum()) for k in range(4)])
    conf = P.max(1)
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for b in range(10):
        m = (conf > edges[b]) & (conf <= edges[b + 1])
        if b == 0:
            m |= conf == 0
        if m.any():
            ece += m.mean() * abs((yp[m] == y[m]).mean() - conf[m].mean())
    onehot = np.eye(4)[y]
    brier = ((P - onehot) ** 2).sum(1).mean()
    nll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1)).mean()
    return dict(accuracy=(yp == y).mean(), macro_precision=prec.mean(), macro_recall=rec.mean(),
                macro_f1=f1.mean(), ece=ece, brier=brier, nll=nll), cm, prec, rec, f1, spec

def family(nm):
    for pat, lab in [(r"^[GMPN]_\d+", "G/M/P/N_### (+aug)"), (r"^Tr-\w\w_\d+", "Tr-xx_####"), (r"^Te-\w\w_\d+", "Te-xx_####"),
                     (r"^(gg|m\d?|p) ?\(\d+\)", "gg/m/p (n)"), (r"^image ?\(\d+\)", "image(n)"),
                     (r"^(\d+ ?no|N\d+|no ?\d+)", "no/N (Chakrabarty-style)"), (r"^\d+\.jpe?g$", "#.jpg")]:
        if re.match(pat, nm, re.I):
            return lab
    return "other"

def cluster_key(nm, y):
    m = re.match(r"^([GMPN])_(\d+)", nm)
    return f"{y}/{m.group(1)}_{m.group(2)}" if m else f"{y}/{nm}"

keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "ece", "brier", "nll"]
per_fold, Y, PP, NAMES = [], [], [], []
for f in files:
    rows, y, P, names = load(f)
    per_fold.append(metrics(y, P)[0])
    Y.append(y); PP.append(P); NAMES += names
y, P = np.concatenate(Y), np.concatenate(PP)

print("## Per-fold metrics")
print("| Fold | n | " + " | ".join(keys) + " |")
for k, (m, yy) in enumerate(zip(per_fold, Y), 1):
    print(f"| {k} | {len(yy)} | " + " | ".join(f"{m[x]:.5f}" for x in keys) + " |")
arr = np.array([[m[x] for x in keys] for m in per_fold])
print("| Mean ± SD | | " + " | ".join(f"{a:.5f} ± {s:.5f}" for a, s in zip(arr.mean(0), arr.std(0, ddof=1))) + " |")
print("| Min – max | | " + " | ".join(f"{a:.5f} – {b:.5f}" for a, b in zip(arr.min(0), arr.max(0))) + " |")

pooled, cm, prec, rec, f1, spec = metrics(y, P)
print("\n## Pooled OOF (n=%d)" % len(y))
for x in keys:
    print(f"{x:16s} {pooled[x]:.5f}")
print("confusion matrix (rows true, cols pred):\n", cm)
print("| Class | Support | Precision | Recall (sensitivity) | Specificity | F1 |")
for k, c in enumerate(C):
    print(f"| {c} | {cm[k].sum()} | {prec[k]:.4f} | {rec[k]:.4f} | {spec[k]:.4f} | {f1[k]:.4f} |")

rng = np.random.default_rng(42)
B = 2000
clusters = collections.defaultdict(list)
for i, (nm, yy) in enumerate(zip(NAMES, y)):
    clusters[cluster_key(nm, yy)].append(i)
cl = [np.array(v) for v in clusters.values()]
print(f"\nclusters: {len(cl)} (images {len(y)}; augmented-copy clusters merge up to {max(len(c) for c in cl)} images)")
for label, sampler in [("image-level", lambda: rng.integers(0, len(y), len(y))),
                       ("cluster-level", lambda: np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]))]:
    boot = {x: [] for x in ["accuracy", "macro_f1", "ece", "brier"]}
    for _ in range(B):
        idx = sampler()
        m = metrics(y[idx], P[idx])[0]
        for x in boot:
            boot[x].append(m[x])
    print(f"95% bootstrap CI ({label}, B={B}): " + "; ".join(
        f"{x} [{np.percentile(v, 2.5):.4f}, {np.percentile(v, 97.5):.4f}]" for x, v in boot.items()))

yp = P.argmax(1)
fam = collections.defaultdict(lambda: [0, 0])
for nm, a, b in zip(NAMES, y, yp):
    fam[family(nm)][0] += 1
    fam[family(nm)][1] += int(a != b)
print("\n| Filename family | n | Errors | Error rate |")
for f_, (n, e) in sorted(fam.items(), key=lambda t: -t[1][0]):
    print(f"| {f_} | {n} | {e} | {e / n:.2%} |")

conf = P.max(1)
edges = np.linspace(0, 1, 11)
print("\n| Confidence bin | n | Accuracy | Mean confidence |")
bins = []
for b in range(10):
    m = (conf > edges[b]) & (conf <= edges[b + 1])
    if m.any():
        bins.append((edges[b], edges[b + 1], m.sum(), (yp[m] == y[m]).mean(), conf[m].mean()))
        print(f"| ({edges[b]:.1f}, {edges[b+1]:.1f}] | {m.sum()} | {(yp[m]==y[m]).mean():.4f} | {conf[m].mean():.4f} |")
print(f"max confidence anywhere: {conf.max():.4f}; errors with confidence >= 0.95: {int(((yp != y) & (conf >= 0.95)).sum())} of {int((yp != y).sum())}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, (ax, ah) = plt.subplots(2, 1, figsize=(5.2, 6.2), height_ratios=[3, 1.1], sharex=True)
ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color=MUTED, label="Perfect calibration")
centers = [(lo + hi) / 2 for lo, hi, *_ in bins]
ax.bar(centers, [a for *_, a, _ in bins], width=0.094, color=BLUE, edgecolor="white", linewidth=2, label="Observed accuracy")
ax.plot(centers, [c for *_, c in bins], "o", ms=6, color=INK, label="Mean confidence in bin")
for x_, (_, _, n, a, _c) in zip(centers, bins):
    ax.text(x_, 0.03, f"n={n}", ha="center", va="bottom", fontsize=8, color="white")
ax.set_ylabel("Accuracy")
ax.set_xlim(0.4, 1.0)
ax.set_ylim(0, 1.02)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, handlelength=1.6)
fig.suptitle(f"Reliability diagram, pooled out-of-fold (n = {len(y):,})\nECE = {pooled['ece']:.4f}, Brier = {pooled['brier']:.4f}",
             fontsize=9.5, color=INK)
ah.bar(centers, [n for _, _, n, *_ in bins], width=0.094, color=BLUE, edgecolor="white", linewidth=2)
ah.set_yscale("log")
ah.set_ylabel("Images (log)")
ah.set_xlabel("Predicted confidence (max class probability)")
ah.grid(axis="y", color=GRID, lw=0.6)
ah.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
    ah.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "reliability_oof.png"), dpi=200)
print("\nsaved", os.path.join(out_dir, "reliability_oof.png"))
