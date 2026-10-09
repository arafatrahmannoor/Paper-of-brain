# Roadmap to submission

## Honest ceiling
No reviewer gives 10/10, and this study has limits that no amount of rework removes:
- the data are a public compilation with no patient IDs;
- each image is a single 2D slice;
- there is no prospective or clinical validation.

A **realistic target is 8–8.5/10 at a good journal**, for example Computers in Biology and Medicine, Biomedical Signal Processing and Control, or Scientific Reports.

| Stage | Expected score |
|---|---|
| Now (Sections 3–4 reviewed at 6.5; other sections not yet seen) | ~6 |
| + all writing fixes below (no new experiments) | ~7 |
| + grouped (leak-free) CV + ablations + same-protocol baselines | ~8 |
| + external test set + explainability + radiologist error review | ~8.5–9 |

## A. What I need from you (no new training)

| # | Item | Status |
|---|---|---|
| A1 | Final training code | ✅ Received and audited (`reviews/code-audit.md`). It matches the checkpoints exactly. Saved as `code/brain_tumor_cv.py` |
| A2 | The 5 `Best epoch: N` lines from the Kaggle logs, or the output of `tools/inspect_checkpoints.py` | ⏳ Diagnoses Fold 5 |
| A3 | Full current manuscript (`.docx` or PDF) | ⏳ `new paper.docx` in the repo is 0 bytes |
| A4 | Greyscale check and `tools/provenance_check.py` table (`reviews/code-audit.md` §4) | ⏳ Settles the TTA no-op question and the dataset provenance |
| A5 | URL and citation of the website where the dataset was published | ⏳ Needed for the Dataset section and reference [19] |

## B. Experiments, in priority order

### Critical (reject risk without them)
1. **Leak-free evaluation.** Ready to run: `code/brain_tumor_cv_v2.py` (`CV_MODE="grouped"`; see `code/README.md`). Rerun the same 5-fold protocol with `StratifiedGroupKFold`. Group by filename base (`G_710*` and so on) and merge pHash near-duplicates into the same group. Alternatively, rerun on the de-duplicated Version 7 (5,941 images). Report this as the **main result**, with the image-level CV (99.15%) alongside to show the leakage effect.
2. **Ablations** (`ABLATION=` switch in v2), using the same grouped folds and settings:
   - image-only (EfficientNet-B3);
   - graph-only;
   - concatenation fusion;
   - full model with the uncertainty gate fixed at 0.5/0.5;
   - full model with the transformer replaced by an MLP;
   - single GNN operator instead of three paths;
   - full model.

   Report mean ± SD plus **McNemar's test** on out-of-fold predictions vs the full model.
3. **Baselines** under the identical protocol: ResNet-50, DenseNet-121, ConvNeXt-T, and ViT-B/16 or Swin-T. Remove every "state-of-the-art" claim that rests on other papers' numbers from other datasets.

### High
4. **Complexity:** parameters (19.35 M, done), FLOPs, and latency for a single pass vs the full pipeline (10 TTA × MC-20).
5. **Gate behaviour:** the distribution of w_g and w_i on test images (overall and per class), and the learned τ for all folds.
6. **External test set.** Candidates: *PMRAM: Bangladeshi Brain Cancer MRI* and *Brain Cancer – MRI dataset* on Mendeley. Check class compatibility first, then pHash-check overlap with the training data. Do not use Figshare, SARTAJ, Br35H or Nickparvar; they are already inside the training data.
7. **Explainability:** Grad-CAM for the CNN stream and node importance for the graph stream, on correct and incorrect examples.
8. **Radiologist review** of the 29 high-confidence errors and the `image(n)` family, to separate label noise from genuine model errors.

### Optional
9. Temperature scaling fitted on the validation split (post-hoc ECE).
10. Validation-loss checkpoint selection instead of validation accuracy, if B1 is rerun anyway (B8 in the §3–4 review).

## C. Writing fixes (I can do these once A3 arrives)
- §3–4 items A1–A6 and B1–B12 from `paper6-sections-3-4-review.md`:
  - typeset all equations;
  - add Figure 1 (architecture) and Figure 2 (graph pyramid);
  - remove the self-notes;
  - fix the saturation/hue no-op;
  - use the gate hidden sizes from the checkpoint.
- Dataset section:
  - correct citation (Mendeley DOI, version);
  - real class counts (3,773 / 2,729 / 2,432 / 3,130);
  - describe the data as a compilation of sources;
  - a leakage-quantification paragraph.
- Use 19.35 M parameters and near-uniform α/β; do not claim that the network learns operator preferences.
- Write Results, Discussion, Limitations, Conclusion, Introduction and Abstract from the audited numbers (`final-cv-results.md`), in that order.
- Fully audit the references; [4], [6], [7] and [19] are already known to be wrong.
- Reproducibility statement: public code, fold indices, out-of-fold predictions and checkpoints (this repository).
