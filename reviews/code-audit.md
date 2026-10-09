# Code audit: `code/brain_tumor_cv.py` (the script that produced the 5 folds)

Method:
- I instantiated your exact model classes (with `weights=None` for EfficientNet; the architecture is unchanged) and compared them against the checkpoint reports.
- I ran each suspected issue as a small test. The results below come from actually running the code, not from reading it.

## 1. Verified: code = checkpoints = paper

| Check | Result |
|---|---|
| `state_dict` keys / parameters / buffer elements | **751 / 19,350,155 / 94,043**, identical to every fold's `.pth` report |
| Graph per image | 336 nodes × 24 features, **3,004 directed edges** (as derived in the review) |
| τ parameterisation | `tau = softplus(log_tau) + 1e-6`, with `log_tau` initialised to 0, so τ₀ = 0.693. The learned value is ≈ 0.764 (fold 1) and 0.773 (fold 2). **The paper's "softplus" is correct**; just call the raw parameter ρ rather than "log τ". |
| α / β order | `[GCN, GAT, TransformerConv]`, initialised to zeros. The learned values (≈ ⅓ each) mean they barely moved from initialisation. |
| Class order | `sorted(os.listdir(Train))` gives glioma, meningioma, notumor, pituitary ✅ |
| `num_batches_tracked` | Increases by exactly 1 per training batch (every BatchNorm layer is called once per forward). So `num_batches_tracked ÷ 247` = the best epoch. Your Kaggle log also prints `Best epoch: N` for each fold; just send me those 5 lines. |

## 2. Confirmed issues (ordered by importance)

| # | Issue | Evidence | Effect | Fix in v2 |
|---|---|---|---|---|
| 1 | **Image-level split.** Augmented copies and near-duplicates of the same slice cross train/test | `StratifiedKFold` on images; 35% of images are 7× augmented copies (`final-cv-results.md` §4) | Leakage. No accuracy gap is visible for the filename-detectable copies, but cross-source duplicates cannot be excluded | `CV_MODE="grouped"`: StratifiedGroupKFold over filename stems + pHash clusters, used for both the outer folds and the inner validation split. The code asserts zero shared groups |
| 2 | **Validation is stochastic.** The gate's MC-dropout runs with `training=True` even in `model.eval()` | Two identical eval calls give different logits (max diff 2e-2 at init) | Early stopping on a noisy, saturated metric (val acc ≈ 99% on ~770 images; ties keep the earliest epoch). This is the most likely cause of the weaker Fold 5 | Validation re-seeds the RNG every epoch (`fork_rng` + `manual_seed`), and `SELECT_ON="val_loss"` |
| 3 | **Saturation and hue changes do nothing on greyscale images.** | On an R=G=B image, `adjust_saturation(1.1)` and `adjust_hue(0.05)` change **0** pixels | The "10-way TTA" is really 9 distinct views (identity counted twice). The saturation/hue jitter in training does nothing | `ZoomTTA(1.05)` replaces `SaturationTTA`; saturation/hue removed from `ColorJitter`. *First run the greyscale check in §4 to confirm your files really are greyscale.* |
| 4 | **Different resampling at test time.** TTA views use `PIL.Image.resize` (Pillow's default bicubic filter); training and validation use `transforms.Resize` (bilinear with antialiasing) | Mean difference 9.4 grey levels on a noise image (a worst case; smaller on real MRI) | Even the "identity" TTA view is preprocessed differently from training | All TTA views use the same `transforms.Resize` |
| 5 | The learning-rate split matches by **substring** (`"gat"`, `"trans"`, `"attn"`) | 5,755,520 parameters at 6e-5: the cross-modal transformer, the graph GAT/TransformerConv layers **and their BatchNorm layers** (`bn_gat*`, `bn_trans*`) | Harmless, but the paper must describe it accurately | Unchanged (to stay comparable). Wording for §4.3: "graph-attention, graph-transformer and cross-modal attention modules, including their normalisation layers" |
| 6 | The uncertainty is **not detached**: gradients flow through the MC variance into the auxiliary heads and the features | `_mc_uncertainty` has no `.detach()` | A design choice, but the paper must state it | Unchanged; document it in §3.8 |
| 7 | Test-time MC-dropout masks depend on the global RNG state | No re-seeding per TTA view | Predictions can only be reproduced if the exact same sequence of operations runs | Each TTA view runs under a fixed seed |

## 3. Facts the paper is currently missing (now known from the code)
- Node features use thresholds **bright > 128, dark < 64, mid-range 64–192, very bright > 192** on the 0–255 greyscale. All intensity features are divided by 255 and variance by 255². The position features use the patch's **top-left** grid coordinate (row/s, col/s), and "distance from centre" is computed from that corner.
- GATConv and TransformerConv use **attention dropout 0.1**. GCNConv and GATConv add self-loops (PyG defaults); TransformerConv does not.
- The cross-modal transformer is **post-norm**, 8 heads, dropout 0.05, with modality embeddings initialised N(0, 0.02).
- The label-smoothed loss uses **logits clamped to [−20, 20]**. Batches with a non-finite loss are skipped.
- Gate hidden sizes: 384→24→384 (modality gates), 384→96→384 with LayerNorm (shared gate), 768→384→384 with LayerNorm (cross-modal gate).
- The validation split is 8% of the training part of each fold (≈ 772 images; ≈ 8,879 for training, so 247 batches per epoch).

## 4. Two quick checks to run on Kaggle (CPU is fine, a few minutes)

**Are the images greyscale?** This decides issue 3.
```python
import os, numpy as np
from PIL import Image
root = "/kaggle/input/datasets/arafatrahmann/my-data221/Epic and CSCR hospital Dataset"
n = gray = 0
for d, _, fs in os.walk(root):
    for f in fs:
        a = np.asarray(Image.open(os.path.join(d, f)).convert("RGB")).astype(int)
        n += 1
        gray += int(np.abs(a[..., 0] - a[..., 1]).max() <= 2 and np.abs(a[..., 1] - a[..., 2]).max() <= 2)
print(gray, "of", n, "images are greyscale (R=G=B within 2 levels)")
```

**Where do the images come from?** Add the Kaggle datasets `masoudnickparvar/brain-tumor-mri-dataset` and `sartajbhuvaji/brain-tumor-classification-mri` as notebook inputs, then:
```bash
pip install -q imagehash
python tools/provenance_check.py "/kaggle/input/datasets/arafatrahmann/my-data221/Epic and CSCR hospital Dataset" /kaggle/input/brain-tumor-mri-dataset /kaggle/input/brain-tumor-classification-mri
```
It prints how many of your images (per filename family) are exact or near-identical copies of images in those public datasets. Send me the table.
