# Five-fold CV: running audit (updated as folds arrive)

Scripts:
- `tools/audit_fold.py`: per fold, recomputes every metric from the predictions.
- `tools/filename_families.py`: per fold, groups images by filename convention.
- `tools/cross_fold.py`: across folds, checks disjointness, confirmed sibling leakage and pooled out-of-fold (OOF) metrics.

## Per-fold metrics (all recomputed from per-image predictions; every value matches the saved JSON exactly)

| Fold | n | Accuracy | Macro P | Macro R | Macro F1 | ECE | Brier | NLL | Errors | Acc. 95% Wilson CI |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2413 | 0.99295 | 0.99259 | 0.99379 | 0.99317 | 0.01870 | 0.01322 | 0.0460 | 17 | [0.9887, 0.9956] |
| 2 | 2413 | 0.99254 | 0.99275 | 0.99235 | 0.99254 | 0.01903 | 0.01211 | 0.0434 | 18 | [0.9882, 0.9953] |
| 3 | | | | | | | | | | |
| 4 | | | | | | | | | | |
| 5 | | | | | | | | | | |
| **Pooled OOF (folds 1–2)** | 4826 | **0.99275** | – | – | 0.99285 | – | – | – | 35 | [0.9899, 0.9948] |

Mean ± SD will be filled in once all 5 folds are in. Use the sample SD (ddof = 1).

Pooled OOF confusion matrix (folds 1–2; rows = true, columns = predicted; order glioma, meningioma, notumor, pituitary):
```
[[1493,   14,    2,    1],
 [   5, 1080,    0,    7],
 [   1,    0,  969,    2],
 [   1,    1,    1, 1249]]
```
The dominant error is glioma → meningioma (14/35). The next is meningioma → pituitary (7/35).

## Cross-fold checks

- **Disjointness:** folds 1 and 2 share 0 paths, so 4,826 unique images so far. ✅
- **Fold composition:** each fold draws about 1,937 images from `Train/` and about 476 from `Test/`, as expected after merging the folders. ✅

### Confirmed augmented-copy leakage
A sibling is an offline-augmented copy of the same slice, for example `G_710.jpg` and `G_710_RO_.jpg`. A sibling that sits in **another** fold's test set was, by construction, in **this** fold's training or validation data.

| Fold | Numbered-family test images | Sibling confirmed in this fold's training data | Accuracy (confirmed-leaked) | Accuracy (rest of fold) |
|---|---|---|---|---|
| 1 | 856 | **658 (76.9%)** | 0.9894 | 0.9943 |
| 2 | 832 | **597 (71.8%)** | 0.9933 | 0.9923 |

- These are **lower bounds** computed from only 2 of the 5 folds. With all five folds, the confirmed share should approach the ≈100% predicted in `fold1-audit.md`.
- **A fair caveat:** so far the confirmed-leaked images are *not* more accurate than the rest of the fold. That hints the augmented-copy leak may inflate accuracy less than feared. But the "rest" group is not leak-free: the Nickparvar (`Tr-`/`Te-`) and SARTAJ files are known to share near-duplicates across sources. **Only a grouped-CV rerun gives the real, unbiased number.**

### High-confidence errors (likely label noise or source artefacts)
15 of the 35 errors (43%) have confidence ≥ 0.95. Given label smoothing, the model's maximum possible output is about 0.98, so these are essentially maximum-confidence errors.
- 6 are `Tr-me_*` meningioma images predicted as **pituitary** at 0.97–0.98: Tr-me_0269, 0293, 0828, 1025, 1153, 1336.
- 5 are `image(n)` glioma images predicted as **meningioma or no tumour** at 0.97–0.98: image(2), (16), (19), (54), (82).
- The rest are Tr-pi_1115 (pituitary → glioma), Tr-no_0093 (notumor → pituitary), G_709_RO_ and G_710_SP_.

**Action:** open these 15 images and have someone with radiology knowledge look at them. If several are mislabelled, that is a strong, honest error-analysis paragraph. It also corrects the Fold 1 note: confidence separates errors from correct predictions *on average* (0.83–0.85 vs 0.98), but almost half of the errors are confident. So the paper must **not** claim that low confidence reliably flags errors.

## Pending (when folds 3–5 arrive)
1. Coverage: union = 12,064 paths, all pairwise overlaps = 0.
2. Mean ± SD for every metric; pooled OOF metrics with bootstrap CIs; final OOF confusion matrix.
3. Final confirmed-leakage table, leakage per filename family, and error rate per source.
4. Reliability diagram (pooled OOF).
