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
