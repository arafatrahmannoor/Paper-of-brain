# v2 grouped (duplicate-aware) CV — run log

Script: `code/brain_tumor_cv_v2.py`, `CV_MODE="grouped"`, pHash merge (Hamming ≤ 5), checkpoint selection on validation loss.
Hardware: Kaggle, Tesla T4.

## Grouping (identical for every run)
- 12,064 images → 3,873 groups (3,626 merged by filename stem); largest group 93 images; 2,002 groups contain more than one image.
- 108 near-duplicate pairs carry different class labels (`near_duplicate_label_conflicts.csv`). These are possible label noise.
- Test images per fold: 2343 / 2310 / 2539 / 2397 / 2475. Every fold shares 0 groups with its training set.

## Results

| Ablation | Fold | Acc | Macro-P | Macro-R | Macro-F1 | ECE | Brier | Best epoch | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| full | 1 (`FOLD_TO_RUN=0`) | 0.9706 | 0.9681 | 0.9685 | 0.9683 | 0.0140 | 0.0495 | 16 (stopped 21) | 6.81 h |
| full | 2 (`FOLD_TO_RUN=1`) | 0.9797 | 0.9768 | 0.9803 | 0.9784 | 0.0139 | 0.0374 | 15 (stopped 20) | n/a |
| full | 3 (`FOLD_TO_RUN=2`) | 0.9516 | 0.9502 | 0.9543 | 0.9511 | 0.0131 | 0.0809 | 8 (stopped 13) | n/a |
| full | 4 (`FOLD_TO_RUN=3`) | 0.9816 | 0.9815 | 0.9809 | 0.9811 | 0.0243 | 0.0301 | 11 (stopped 16) | n/a |
| full | 5 (`FOLD_TO_RUN=4`) | 0.9434 | 0.9457 | 0.9358 | 0.9389 | 0.0248 | 0.0944 | 11 (stopped 16) | n/a |

**Final 5-fold result (`full`, complete).** Regenerate with `python tools/aggregate_grouped_cv.py full`.
- Every image appears in exactly one test fold: n = 12,064, 423 errors.
- Pooled accuracy 0.9649. Group-cluster bootstrap 95% CI: [0.9550, 0.9739].
- Mean ± SD across folds:
  - accuracy 0.9654 ± 0.0171
  - macro-P 0.9645 ± 0.0159
  - macro-R 0.9639 ± 0.0191
  - macro-F1 0.9636 ± 0.0181
  - ECE 0.0180 ± 0.0060
  - Brier 0.0585 ± 0.0280

Pooled confusion matrix, rows true and columns predicted:
```
glioma     3644   84   39    6
meningioma   28 2658   23   20
notumor      69   13 2311   39
pituitary     4   68   30 3028
```
Per-class recall: 0.9658 / 0.9740 / 0.9502 / 0.9674. Per-class precision: 0.9730 / 0.9416 / 0.9617 / 0.9790.

**Errors cluster by source image.** The dataset contains offline-augmented copies of each source slice (`_BR_`, `_DA_`, `_HF_`, `_RO_`), and grouping keeps all copies of a slice in the same fold. When the model gets a source image wrong, it usually gets most of its copies wrong too.
- Fold 3 (95.16%) is one of the two low folds. 56% of its 123 errors come from 5 groups. In one, 19 of the 20 copies of `P_818` (pituitary) were predicted as notumor.
- Fold 3 also stopped early: best epoch 8, stopped at 13. That is a legitimate early-stopping outcome, not a failed run.
- The effective sample size is therefore closer to the number of groups than to the number of images. Report the group-cluster CI and the fold SD, not a binomial CI over images.
- In fold 3, hospital-style files (`G_/M_/P_*`) have an error rate of 9.2%, against 1.4% for `Tr-/Te-*` files.

- Fold 5 (94.34%) shows the same pattern. 78 of its 140 errors come from 5 source slices (four of them hospital no-tumour slices), each with 10–25 copies. Examples: `N_378` (23 of 25 copies predicted as glioma), `N_375` (17 of 20 predicted as pituitary) and `N_39`.

**Errors across all 5 folds.**

By file source:

| Source | Error rate | Errors / images |
|---|---|---|
| `G_/M_/N_/P_*` (hospital-style) | 5.89% | 250 / 4246 |
| `Tr-/Te-*` (Kaggle-style) | 1.57% | 73 / 4642 |
| Other names | 3.15% | 100 / 3176 |

By augmentation:

| Images | Error rate | Errors / images |
|---|---|---|
| Offline-augmented copies (`_XX_` suffix) | 5.93% | 216 / 3641 |
| Originals | 2.46% | 207 / 8423 |

- The 423 errors fall in 136 of 3,873 groups. In 91 of those groups, at least half the images are wrong.
- The 10 worst groups account for 151 of the 423 errors.

GNN path weights α and β stay at about 1/3 in all 5 folds. Mean `w_image` per fold: 0.82, 0.69, 0.72, 0.69, 0.73.

Fold 1, `full`: confusion matrix. Rows are true classes and columns are predicted classes, both in the order glioma, meningioma, notumor, pituitary. n = 2343, 69 errors.

```
glioma      754   8   4   0
meningioma    5 477  12   4
notumor      12   6 441   0
pituitary     1   9   8 602
```
Per-class recall: 0.9843 / 0.9578 / 0.9608 / 0.9710.

The same fold under the v1 image-level split scored 99.30% (17 errors), so near-duplicate leakage accounted for about 2.2 points of accuracy.

## Complexity (`full`, T4)
19,350,155 parameters; 2.2587 GFLOPs per single pass. Timings per image:

| Step | Time |
|---|---|
| Single forward pass | 21.05 ms |
| CPU graph construction | 232.07 ms |
| Full pipeline (10 TTA × 20 MC) | 2630.9 ms |

## Checks on the uploaded fold 1 files
- Recomputed from `grouped_full_fold_1_predictions.csv`, the predictions match the console output exactly: accuracy 0.9706, ECE 0.0139 (15 bins), Brier 0.0495, and the same confusion matrix.
- Leakage check using `groups.csv` and `fold_assignment_grouped.csv` (`test_fold` is 1-based): all 5 folds share 0 groups with their training data.
- **Label conflicts.** All 108 conflicting pairs fall inside a single group, so none of them leaks across the split.
  - 25 groups (538 images) contain more than one class.
  - By class pair: glioma/meningioma 49, glioma/notumor 45, meningioma/notumor 10, meningioma/pituitary 4.
  - The 4 pairs at Hamming distance 0 are the same image filed under two labels. Each pairs a `Tr-me_*` meningioma with a `P_*_HF_` pituitary image. These are certain label errors in the dataset.
  - 44 of the fold 1 test images belong to conflict pairs, and only 2 of them were misclassified.
- **Gate behaviour.**
  - Mean `w_image` is 0.82 (median 0.90). The image stream gets the larger weight for 96.8% of test images.
  - Mean `w_image` is 0.86 on misclassified images and 0.82 on correctly classified ones.
  - Learned τ = 0.764.
  - The GNN path weights stay at about 1/3 each: α = [0.329, 0.330, 0.341] and β = [0.333, 0.336, 0.330]. They do not specialise.

## Four-model ablation (same grouped folds, all 5 folds each)
Regenerate the figures and tests with `python tools/make_ablation_figures.py`; run `python tools/aggregate_grouped_cv.py <ablation>` for the per-fold checks.

| Model | Accuracy mean ± SD | Pooled | Macro-F1 | ECE | Full − variant (group-bootstrap 95% CI) | McNemar p |
|---|---|---|---|---|---|---|
| image_only | 96.58 ± 1.44 | 96.55 | 96.45 ± 1.51 | 0.015 | −0.06 pp [−0.61, +0.46] | 0.72 |
| graph_only | 87.48 ± 1.73 | 87.43 | 87.13 ± 1.79 | 0.030 | +9.06 pp [+7.76, +10.48] | <1e-200 |
| concat | 96.25 ± 1.74 | 96.21 | 96.14 ± 1.85 | 0.019 | +0.28 pp [−0.16, +0.72] | 0.049 |
| full | 96.54 ± 1.71 | 96.49 | 96.36 ± 1.81 | 0.018 | — | — |

- EfficientNet-B3 alone matches the full model, so the graph stream and the fusion modules add no measurable accuracy on this dataset.
- A repeated training of the full model on fold 1 gave 94.62% instead of 97.06%, so single-run differences of a point or two are within run-to-run noise.
