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
