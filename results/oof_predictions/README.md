# Out-of-fold predictions (image-level stratified 5-fold CV)

`fold_k_predictions.csv` holds the held-out test predictions of fold k, written by `code/brain_tumor_cv.py` with 10-view TTA. The five files are pairwise disjoint and together cover all 12,064 images exactly once, so they also define the fold assignment.

| Column | Meaning |
|---|---|
| `sample_index` | row index within the fold's test set |
| `path` | image path on Kaggle (dataset `arafatrahmann/my-data221`); the part after `Epic and CSCR hospital Dataset/` is the path inside the Mendeley archive |
| `true_index`, `true_class` | label (0 glioma, 1 meningioma, 2 notumor, 3 pituitary) |
| `pred_index`, `pred_class` | predicted label (argmax of the TTA-averaged probabilities) |
| `prob_*` | TTA-averaged class probabilities |

Every metric in the paper can be recomputed from these files with `tools/final_cv_summary.py`.
