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
| 3 | 2413 | 0.99254 | 0.99283 | 0.99230 | 0.99255 | 0.01879 | 0.01317 | 0.0467 | 18 | [0.9882, 0.9953] |
| 4 | | | | | | | | | | |
| 5 | | | | | | | | | | |
| **Pooled OOF (folds 1–3)** | 7239 | **0.99268** | – | – | 0.99280 | – | – | – | 53 | [0.9904, 0.9944] |

Mean ± SD will be filled in once all 5 folds are in. Use the sample SD (ddof = 1).

Pooled OOF confusion matrix (folds 1–3; rows = true, columns = predicted; order glioma, meningioma, notumor, pituitary):
```
[[2246,   14,    4,    1],
 [  13, 1614,    0,   10],
 [   1,    0, 1456,    2],
 [   1,    6,    1, 1870]]
```
Almost all errors (49/53) are among the three tumour classes:
- glioma ↔ meningioma: 27 errors;
- meningioma ↔ pituitary: 16 errors;
- **no tumour:** recall 1456/1459 = 99.8%; only 5 tumour images were called "no tumour", 4 of them gliomas.

## Cross-fold checks

- **Disjointness:** folds 1, 2 and 3 are pairwise disjoint (0 shared paths), giving 7,239 unique images so far. ✅
- **Fold composition:** each fold draws about 1,937 images from `Train/` and about 476 from `Test/`, as expected after merging the folders. ✅

### Confirmed augmented-copy leakage
A sibling is an offline-augmented copy of the same slice, for example `G_710.jpg` and `G_710_RO_.jpg`. A sibling that sits in **another** fold's test set was, by construction, in **this** fold's training or validation data.

| Fold | Numbered-family test images | Sibling confirmed in this fold's training data (folds 1–3 visible) | Accuracy (confirmed-leaked) | Accuracy (rest of fold) |
|---|---|---|---|---|
| 1 | 856 | **826 (96.5%)** | 0.9879 | 0.9956 |
| 2 | 832 | **804 (96.6%)** | 0.9950 | 0.9913 |
| 3 | 833 | **792 (95.1%)** | 0.9949 | 0.9914 |
| Pooled | 2521 | **2422 (96.1%)** | 0.9926 | 0.9927 |

- With 3 of the 5 folds visible, **96%** of augmented-family test images already have a copy confirmed in training. The ≈100% predicted in `fold1-audit.md` is effectively confirmed.
- Pooled accuracy is **identical** for confirmed-leaked images and for the rest (99.26% vs 99.27%). So the augmented-copy leak shows **no measurable inflation relative to the other images**. That is encouraging, but it is not proof of no inflation, because the comparison group (the Nickparvar `Tr-`/`Te-` and SARTAJ files) probably has cross-source near-duplicates of its own. A grouped-CV rerun remains the definitive test. The paper can now honestly say the leak was identified and quantified, and that no accuracy gap was observed between leaked and non-leaked images.

### High-confidence errors (likely label noise or source artefacts)
23 of the 53 errors (43%) have confidence ≥ 0.95. Given label smoothing, the model's maximum output is about 0.98, so these are essentially maximum-confidence errors.
- **10 `Tr-me_*` meningioma images predicted as pituitary at 0.95–0.98:** 0269, 0293, 0828, 0829, 0831, 0942, 1025, 1153, 1336. Three of these IDs are consecutive (0828, 0829, 0831), which suggests adjacent slices of one patient. Sellar/parasellar meningiomas can genuinely mimic pituitary adenomas, so this could be either a real diagnostic difficulty or label noise. Either way it makes a good clinical discussion point.
- **7 `image(n)` gliomas predicted as meningioma or no tumour at 0.97–0.98:** image(2), (16), (18), (19), (54), (56), (82). This SARTAJ testing-folder family has a 6.3% error rate (10 errors in 158 images so far), against 0.73% overall (53/7239). That is consistent with known label problems in that source.
- The rest: Tr-pi_1115, Tr-no_0093, P_834, P_844_VF_, m3 (134), G_709_RO_ and G_710_SP_.
- **Sibling errors:** copies of the same slice fail together. G_712_SP_ and G_712_VF_ fail in Fold 1; P_834 and P_834_HF_ fail in Fold 3. Errors are therefore not independent across images, which is one more reason the per-image confidence interval is optimistic.

**Action:** open these 23 images and have someone with radiology knowledge look at them. Confidence separates errors from correct predictions *on average* (about 0.85 vs 0.98), but almost half of the errors are confident, so the paper must **not** claim that low confidence reliably flags errors.

## Pending (when folds 3–5 arrive)
1. Coverage: union = 12,064 paths, all pairwise overlaps = 0.
2. Mean ± SD for every metric; pooled OOF metrics with bootstrap CIs; final OOF confusion matrix.
3. Final confirmed-leakage table, leakage per filename family, and error rate per source.
4. Reliability diagram (pooled OOF).
