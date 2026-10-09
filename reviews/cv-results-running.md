# Five-fold CV: running audit (updated as folds arrive)

> **All five folds are now in. The final, authoritative numbers are in [`final-cv-results.md`](final-cv-results.md).** This file is kept as the interim log written while folds 1–4 were arriving.

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
| 4 | 2413 | 0.99461 | 0.99456 | 0.99457 | 0.99456 | 0.02306 | 0.00910 | 0.0406 | 13 | [0.9908, 0.9968] |
| 5 | 2412 | 0.98466 | 0.98402 | 0.98490 | 0.98443 | 0.02480 | 0.02562 | 0.0696 | 37 | [0.9789, 0.9889] |
| *Interim mean ± SD (folds 1–4)* | | *0.99316 ± 0.00099* | *0.99318 ± 0.00092* | *0.99325 ± 0.00112* | *0.99320 ± 0.00095* | *0.01989 ± 0.00211* | *0.01190 ± 0.00193* | | | |
| **Pooled OOF (folds 1–4)** | 9652 | **0.99316** | – | – | 0.99320 | – | – | – | 66 | [0.9913, 0.9946] |

The interim row uses the sample SD (ddof = 1). Recompute it with all 5 folds before using it anywhere.

Fold 4 has the highest accuracy but also the highest ECE. Its mean confidence (0.9726) is lower and its accuracy (0.9946) higher, so the under-confidence gap that drives ECE is wider. This confirms that ECE here measures under-confidence (see `fold1-audit.md` §4), not over-confidence.

Pooled OOF confusion matrix (folds 1–4; rows = true, columns = predicted; order glioma, meningioma, notumor, pituitary):
```
[[2995,   16,    6,    2],
 [  15, 2156,    0,   12],
 [   2,    0, 1940,    4],
 [   1,    6,    2, 2495]]
```
- 52 of the 66 errors are among the three tumour classes: glioma ↔ meningioma 31, meningioma ↔ pituitary 18, glioma ↔ pituitary 3.
- **No tumour:** recall 1940/1946 = 99.7%. 8 tumour images were called "no tumour" (6 gliomas, 2 pituitary), and 6 no-tumour images were called tumour.

## Cross-fold checks

- **Disjointness:** folds 1–4 are pairwise disjoint (0 shared paths in all 6 pairs), giving 9,652 unique images so far. ✅
- **Fold composition:** each fold draws about 1,940 images from `Train/` and about 470 from `Test/`, as expected after merging the folders. ✅

### Confirmed augmented-copy leakage
A sibling is an offline-augmented copy of the same slice, for example `G_710.jpg` and `G_710_RO_.jpg`. A sibling that sits in **another** fold's test set was, by construction, in **this** fold's training or validation data.

| Fold | Numbered-family test images | Sibling confirmed in this fold's training data (folds 1–4 visible) | Accuracy (confirmed-leaked) | Accuracy (rest of fold) |
|---|---|---|---|---|
| 1 | 856 | **850 (99.3%)** | 0.9882 | 0.9955 |
| 2 | 832 | **830 (99.8%)** | 0.9952 | 0.9912 |
| 3 | 833 | **827 (99.3%)** | 0.9927 | 0.9924 |
| 4 | 853 | **847 (99.3%)** | 0.9965 | 0.9936 |
| Pooled | 3374 | **3354 (99.4%)** | 0.9931 | 0.9932 |

- **The leak is confirmed:** 99.4% of augmented-family test images have a copy of the same slice in that fold's training data. That is about 35% of every test fold.
- **No measurable accuracy gap:** leaked 99.31% vs the rest 99.32%. The per-fold differences go in both directions (fold 1 lower; folds 2–4 slightly higher), consistent with noise.
- **Caveat:** the comparison group (Nickparvar `Tr-`/`Te-`, SARTAJ, plain-numbered files) is not proven leak-free. Cross-source near-duplicates are likely, and they need pHash to detect. A grouped-CV rerun remains the definitive test.
- **What the paper can say:** the leak was identified, measured, and showed no accuracy gap between leaked and non-leaked images.

### High-confidence errors (likely label noise or source artefacts)
26 of the 66 errors (39%) have confidence ≥ 0.95. Given label smoothing, the model's maximum output is about 0.98, so these are essentially maximum-confidence errors.
- **9 `Tr-me_*` meningioma images predicted as pituitary at 0.95–0.98:** 0269, 0293, 0828, 0829, 0831, 0942, 1025, 1153, 1336. Tr-me_1263 was also predicted as glioma at 0.977. Three IDs are consecutive (0828, 0829, 0831), which suggests adjacent slices of one patient. Sellar/parasellar meningiomas can genuinely mimic pituitary adenomas, so this is either a real diagnostic difficulty or label noise. Either way it makes a good clinical discussion point.
- **8 `image(n)` gliomas predicted as meningioma or no tumour at 0.97–0.98:** image(2), (15), (16), (18), (19), (54), (56), (82). This SARTAJ testing-folder family has a **6.3% error rate (13 errors in 206 images)**, against **0.68% overall (66/9652)**, about nine times higher. That is consistent with known label problems in that source.
- `Tr-no_0093` and `Tr-no_0103`: no-tumour images predicted as pituitary at 0.96–0.98.
- The rest: Tr-pi_1115, P_834, P_844_VF_, m3 (134), G_709_RO_ and G_710_SP_.
- **Sibling errors:** copies of the same slice fail together (G_712_SP_ and G_712_VF_ in fold 1; P_834 and P_834_HF_ in fold 3). Errors are not independent across images, which is one more reason the per-image confidence interval is optimistic.

**Action:** open these 26 images and have someone with radiology knowledge look at them. Mean confidence on errors is 0.77–0.86 per fold, against 0.97–0.98 on correct predictions, so confidence ranks errors *on average*. But almost 40% of errors are made at maximum confidence, so the paper must **not** claim that low confidence reliably flags errors.

## Pending (when fold 5 arrives)
1. Coverage: union = 12,064 paths, all 10 pairwise overlaps = 0.
2. Final mean ± SD for every metric; pooled OOF metrics with bootstrap CIs; final OOF confusion matrix.
3. Final confirmed-leakage table, leakage per filename family, and error rate per source.
4. Reliability diagram (pooled OOF).
