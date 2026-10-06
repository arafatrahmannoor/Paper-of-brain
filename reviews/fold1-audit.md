# Fold 1 audit (final code: MC-Dropout T=20, 10-way TTA)

Files audited: `fold_1_predictions.csv`, `fold_1_confusion_matrix.csv`, `fold_1_confusion_matrix_normalized.csv`, `fold_1_metrics.json`, `fold_1_classification_report.json`, `fold_1_results.png`.
Scripts: `tools/audit_fold.py` and `tools/filename_families.py`. Rerun them on folds 2–5.

## 1. Integrity: PASS

| Check | Result |
|---|---|
| Rows / unique sample indices / unique paths | 2413 / 2413 / 2413 (no duplicates within the fold) |
| Probability rows sum to 1 | min = max = 1.000000 |
| `pred_index` = argmax(probabilities) | 0 mismatches |
| Index ↔ class-name, and folder name ↔ `true_class` | 0 mismatches (class order glioma, meningioma, notumor, pituitary is correct) |
| Confusion matrix recomputed from predictions = saved CSV = PNG | identical |
| Fold drawn from merged folders | 1,936 from `Train/` and 477 from `Test/` (≈ 80/20, as expected after merging) |
| Class rows | 755 / 546 / 486 / 626, i.e. one-fifth of 3773 / 2729 / 2432 / 3130 |

| Metric | Recomputed | Saved | Difference |
|---|---|---|---|
| Accuracy | 0.992955 (2396/2413) | 0.992955 | 0 |
| Macro precision | 0.992586 | 0.992586 | 0 |
| Macro recall | 0.993786 | 0.993786 | ~1e-16 |
| Macro F1 | 0.993169 | 0.993169 | ~1e-16 |
| ECE (10 bins, top-label) | 0.018697 | 0.018697 | ~1e-14 |
| Brier (sum over classes) | 0.013225 | 0.013225 | ~1e-13 |
| NLL (not saved) | 0.0460 | – | – |
| Accuracy 95% Wilson CI | [0.9887, 0.9956] | – | – |

The saved numbers are exactly reproducible from the per-image predictions.

**Note:** Fold 1 with the final code gives **99.30%**, not the 99.50% from the earlier code. The old Figure 4 and the 99.50% must not appear anywhere in the paper.

## 2. Finding 1 (critical): the dataset contains offline-augmented copies, and they are split across train and test

Filenames in the numbered family have the form `G_710.jpg`, `G_710_BR_.jpg`, `G_710_DA_.jpg`, `G_710_RO_.jpg`, `G_710_SP_.jpg`, and so on. The suffixes RO, VF, HF, BR, SP and DA are offline augmentations (rotation, vertical flip, horizontal flip, brightness, and probably salt-and-pepper/sharpen and darken) of one original slice.

- 856 of the 2,413 test images (35.5%) belong to this family. They come from 482 distinct source images, so several augmented copies of the same slice sit **inside one test fold**. Example: G_710 appears in Fold 1's test set 5 times (ORIG, BR, DA, RO, SP).
- The mean multiplicity of a source image within this 20% fold is 1.776. Under random fold assignment that is exactly what **k ≈ 7 copies per source image** produces (the original plus the 6 suffixes). So each of those test images has about 6 siblings, and about 80% of them (≈ 4.8) are in the **training** set of the same fold.
- The probability that a numbered-family test image has *no* sibling in training is about 0.2⁶ ≈ 0.00006. In practice, **every one of these 856 test images has rotated, flipped or brightness-shifted copies of itself in training.**
- The errors are correlated too. G_712_SP_ and G_712_VF_ (two copies of the same slice) are both misclassified as meningioma.

This confirms, directly from your own output, the leakage that Version 7 of the dataset reports ("exact and near-duplicate removal"). No pHash is even needed for this part. The images can be grouped by filename.

## 3. Finding 2 (important): the filenames point to public Kaggle datasets

| Filename family (Fold 1 test) | n | Errors | Looks like |
|---|---|---|---|
| `G/M/P/N_###[_AUG_].jpg` | 856 | 10 (1.2%) | Numbered set with 7× offline augmentation |
| `Tr-xx_####.jpg` | 755 | 2 (0.3%) | Training-split naming of the Kaggle "Brain Tumor MRI Dataset" (M. Nickparvar) |
| `Te-xx_####.jpg` | 162 | 0 | Testing-split naming of the same Kaggle dataset |
| `#.jpg` | 370 | 0 | Plain numbered files |
| `gg (n)`, `m (n)`, `m#(n)`, `p (n)` | 211 | 0 | Naming used in the SARTAJ "Brain Tumor Classification (MRI)" Kaggle dataset |
| `image(n).jpg` | 46 | **5 (10.9%)** | Testing-folder naming of the same SARTAJ dataset |
| `# no.jpg`, `N#.jpg`, `no*.jpg` | 13 | 0 | Naming used in N. Chakrabarty's "Brain MRI Images for Brain Tumor Detection" |

Implications:
- **Provenance.** Version 7 describes the dataset as coming from "Epic & CSCR Hospital, Bangladesh", yet about half of Fold 1 (1,187 of 2,413 images, 49%) carries the filename conventions of well-known public Kaggle compilations, and a further 370 plain-numbered files are of unclear origin. Confirm this by hashing a sample against those datasets before writing anything. If it is confirmed, Section 3.1 must describe the dataset as a compilation, and the paper must not call it a single-hospital cohort.
- **External validation.** Figshare (Cheng), SARTAJ, Br35H and Nickparvar would *not* be independent test sets for this model. Their images are probably already inside the training data. The Nickparvar dataset is itself built from Figshare + SARTAJ + Br35H.
- **Cross-source duplicates.** The public compilations are known to share images with each other. That is a second leakage path, separate from the augmentation suffixes. For this part you do need perceptual hashing.
- **Error rate by source.** 5 of the 17 errors (29%) come from the 46 `image(n)` files (10.9% error rate). That family is known for label-quality problems. Look at those 5 images. Some may be mislabelled rather than misclassified, which is a nice point for the error-analysis section.

## 4. Finding 3: calibration, where ECE mostly reflects label smoothing

| Confidence bin | n | Accuracy | Mean confidence |
|---|---|---|---|
| (0.5, 0.6] | 5 | 0.600 | 0.554 |
| (0.6, 0.7] | 4 | 0.750 | 0.623 |
| (0.7, 0.8] | 7 | 0.714 | 0.771 |
| (0.8, 0.9] | 18 | 0.833 | 0.850 |
| (0.9, 1.0] | 2379 | **0.9962** | **0.9779** |

- The **highest probability the model outputs on any image is 0.984.** With label smoothing ε = 0.03 over 4 classes, the training target for the true class is 1 − 0.03 + 0.03/4 = **0.9775**, and the top bin's mean confidence is 0.9779.
- So the ECE of 0.0187 is almost entirely the gap between the confidence ceiling that label smoothing imposes and the 99.6% accuracy in that bin. **The model is slightly under-confident.** The ECE value comes from label smoothing, not from the MC-Dropout gate.
- On the positive side, low-confidence predictions really are less accurate (mean confidence 0.855 on errors vs 0.976 on correct predictions), so confidence is informative for ranking.

For the paper:
- describe ECE as under-confidence;
- add a reliability diagram and NLL;
- optionally report ECE after temperature scaling fitted on the validation split;
- do not claim that the uncertainty module produced the calibration unless the gate ablation shows it.

## 5. What to do with folds 2–5

Nothing needs to change in the runs already in progress. Upload each fold's 5 files the same way. Once all five are in I will:
1. check that the five test folds are disjoint and cover all 12,064 paths;
2. compute mean ± SD, pooled OOF metrics with bootstrap CIs, and the OOF confusion matrix;
3. **quantify the leakage exactly**: accuracy on test images with siblings in training vs without, and accuracy per filename family.

The fix (strongly recommended, since it is cheap): rerun with grouped folds, so that all copies of a slice stay on the same side.

```python
import re, os
from sklearn.model_selection import StratifiedGroupKFold

def group_key(path, label):
    name = os.path.basename(path)
    m = re.match(r'^([GMPN])_(\d+)', name)          # G_710, G_710_BR_, ... -> one group
    if m:
        return f'{label}/{m.group(1)}_{m.group(2)}'
    return f'{label}/{name}'                          # other files: own group (refine with pHash below)

groups = [group_key(p, y) for p, y in zip(all_paths, all_labels)]
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for k, (trainval_idx, test_idx) in enumerate(sgkf.split(all_paths, all_labels, groups)):
    ...
```

To also catch cross-source duplicates, compute `imagehash.phash` for every image and merge into one group any images whose hashes are within a Hamming distance of about 6 (union-find). This takes a few minutes on CPU for 12k images. Use the same grouping for the inner validation split, e.g. `GroupShuffleSplit`.

Report both protocols in the paper. The drop from image-level to grouped CV *is* the leakage estimate, and showing it openly is far stronger than hiding it. Alternatively, use Version 7 (already de-duplicated) as the main experiment.

---

## সংক্ষেপে (বাংলা)

- Fold 1-এর সব সংখ্যা আমি predictions থেকে নিজে হিসাব করে মিলিয়েছি, **হুবহু মিলেছে**। Final code-এ Fold 1 = **99.30%** (আগের 99.50% আর ব্যবহার করা যাবে না)।
- **বড় সমস্যা:** file name দেখে বোঝা যাচ্ছে dataset-এ একই image-এর ~৭টা করে augmented copy আছে (ORIG, RO, VF, HF, BR, SP, DA)। Test-এর 856টা image-এর প্রায় প্রতিটার rotate/flip করা copy training-এ আছে। এটা সরাসরি leakage।
- প্রায় অর্ধেক (৪৯%) file-এর নাম public Kaggle dataset-এর মতো (`Tr-gl_…`, `gg (n)` ইত্যাদি)। তাই "Bangladesh hospital dataset" দাবিটা যাচাই করতে হবে, আর ওই public dataset-গুলো external test হিসেবে ব্যবহার করা যাবে না।
- ECE 0.0187 মূলত label smoothing-এর কারণে (model-এর সর্বোচ্চ confidence 0.984)। Model একটু under-confident।
- বাকি ৪টা fold পাঠাও। তারপর leakage ঠিক কতটা তা মাপব। সবচেয়ে ভালো সমাধান হলো file name দিয়ে group করে StratifiedGroupKFold-এ আবার চালানো।
