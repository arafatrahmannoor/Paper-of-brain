# Usage (Kaggle, with the dataset attached as input):
#   python tools/make_paper_figures.py --data-root "/kaggle/input/datasets/arafatrahmann/my-data221/Epic and CSCR hospital Dataset" \
#       --pred fold_1_predictions.csv fold_2_predictions.csv fold_3_predictions.csv fold_4_predictions.csv fold_5_predictions.csv \
#       --out figures
# Optional: --remap "/kaggle/input/...=/local/path" if the prediction CSVs hold paths from another machine.
#
# Writes three 300-dpi PNGs for the manuscript:
#   fig1_classes.png          one representative original (non-augmented) slice per class
#   fig3_graph_pyramid.png    the 4x4, 8x8 and 16x16 grids of the graph pyramid drawn on one slice
#   fig5_confident_errors.png the eight most confident out-of-fold errors with true/predicted label and confidence
import argparse, csv, os, re
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
NAMES = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary tumour"}
SHORT = {"glioma": "Glioma", "meningioma": "Meningioma", "notumor": "No tumour", "pituitary": "Pituitary"}
ACCENT = "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def family(name):
    for pat, lab in [(r"^[GMPN]_\d+", "G/M/P/N_###"), (r"^Tr-\w\w_\d+", "Tr-xx_####"), (r"^Te-\w\w_\d+", "Te-xx_####"),
                     (r"^(gg|m\d?|p) ?\(\d+\)", "gg/m/p (n)"), (r"^image ?\(\d+\)", "image(n)"),
                     (r"^(\d+ ?no|N\d+|no ?\d+)", "N#/# no"), (r"^\d+\.jpe?g$", "#.jpg")]:
        if re.match(pat, name, re.I):
            return lab
    return "other"


def load_gray(path, size=224):
    with Image.open(path) as im:
        return np.asarray(im.convert("L").resize((size, size)))


def pick_examples(data_root):
    """First original (no augmentation suffix) numbered slice per class, sorted by name, from the Train folder."""
    picks = {}
    for c in CLASSES:
        folder = os.path.join(data_root, "Train", c)
        names = sorted(n for n in os.listdir(folder) if re.match(r"^[GMPN]_\d+\.(jpe?g|png)$", n, re.I))
        if not names:  # fall back to any image
            names = sorted(n for n in os.listdir(folder) if n.lower().endswith((".jpg", ".jpeg", ".png")))
        picks[c] = os.path.join(folder, names[0])
    return picks


def fig_classes(picks, out):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for ax, c in zip(axes, CLASSES):
        ax.imshow(load_gray(picks[c]), cmap="gray", vmin=0, vmax=255)
        ax.set_title(NAMES[c], fontsize=9.5)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_pyramid(path, out):
    img = load_gray(path)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, s in zip(axes, [4, 8, 16]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        step = 224 / s
        for k in range(1, s):
            ax.axhline(k * step - 0.5, color=ACCENT, lw=0.8 if s < 16 else 0.5)
            ax.axvline(k * step - 0.5, color=ACCENT, lw=0.8 if s < 16 else 0.5)
        ax.set_title(f"{s} × {s} grid: {s * s} nodes, {224 // s}-px patches", fontsize=8.5)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_errors(pred_files, out, remap, k=8):
    rows = [r for p in pred_files for r in csv.DictReader(open(p, encoding="utf-8"))]
    errs = []
    for r in rows:
        if r["true_index"] != r["pred_index"]:
            conf = max(float(r["prob_" + c]) for c in CLASSES)
            errs.append((conf, r))
    errs.sort(key=lambda t: -t[0])
    errs = errs[:k]
    cols = 4
    nrow = int(np.ceil(len(errs) / cols))
    fig, axes = plt.subplots(nrow, cols, figsize=(7.2, 2.15 * nrow), gridspec_kw={"hspace": 0.35, "wspace": 0.08})
    for ax, (conf, r) in zip(np.ravel(axes), errs):
        path = r["path"]
        for old, new in remap:
            path = path.replace(old, new)
        ax.imshow(load_gray(path), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{SHORT[r['true_class']]} \u2192 {SHORT[r['pred_class']]}\n"
                     f"p = {conf:.2f} \u00b7 {family(os.path.basename(path))}", fontsize=7.5)
        ax.axis("off")
    for ax in np.ravel(axes)[len(errs):]:
        ax.axis("off")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--remap", action="append", default=[], help="OLD=NEW path prefix replacement")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    remap = [tuple(m.split("=", 1)) for m in a.remap]
    picks = pick_examples(a.data_root)
    fig_classes(picks, os.path.join(a.out, "fig1_classes.png"))
    fig_pyramid(picks["glioma"], os.path.join(a.out, "fig3_graph_pyramid.png"))
    errs = fig_errors(a.pred, os.path.join(a.out, "fig5_confident_errors.png"), remap)
    print("Figure 1 slices:", {c: os.path.basename(p) for c, p in picks.items()})
    print("Figure 5 errors:", [(os.path.basename(r["path"]), r["true_class"], r["pred_class"], round(c, 3)) for c, r in errs])
    print("Saved to", a.out)
