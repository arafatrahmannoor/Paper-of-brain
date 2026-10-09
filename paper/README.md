# IEEE version of the manuscript

| File | What it is |
|---|---|
| `main.tex` | Manuscript in IEEEtran journal format |
| `references.bib` | 38 references. IEEEtran.bst numbers them in order of first citation |
| `main.pdf` | Compiled PDF (the version sent for supervisor review) |
| `figures/` | Fig. 2 (architecture) and Fig. 4 (reliability diagram). Figs. 1, 3 and 5 are added as described below |

## Build
- **Overleaf:** upload the whole `paper/` folder (or a zip of it) as a new project and press Recompile.
- **Local:** `pdflatex main && bibtex main && pdflatex main && pdflatex main`

## Before submission
Red text in the PDF marks an item still to be filled in:
- author names, affiliations and corresponding author;
- journal name;
- GPU model and PyTorch / PyTorch Geometric versions;
- acknowledgment / author contributions / AI-tool disclosure;
- contributor names for the Version 7 Mendeley record (`bdneuro2025v7` in `references.bib`).

## Figures

| Fig. | Content | Status |
|---|---|---|
| 1 | Class samples | ✅ `fig1_classes.png`, re-composed from the first-submission panels; order matches the paper, internal title removed, (a)–(d) labels |
| 2 | Architecture | ✅ `fig_architecture.tex`: vector TikZ diagram, edit its text directly in Overleaf. The model is identical in `brain_tumor_cv_v2.py`, so the planned experiments do not change it |
| 3 | Graph pyramid on one slice | ✅ `fig3_graph_pyramid.png`, drawn on the glioma slice of Fig. 1(a) |
| 4 | Pooled out-of-fold confusion matrix | ✅ `fig_confusion.pdf`, from the audited pooled counts of all 5 folds |
| 5 | Reliability diagram | ✅ `fig4_reliability.pdf` |
| W | Study workflow (supervisor's request) | ✅ `fig_workflow.png`: the authors' diagram with two text fixes ("Public Brain MRI Dataset (Mendeley, v1)", "12,064 Images"). Replace it with a 300-dpi PowerPoint export that has the same two fixes |
| 6 | Most confident errors | ✅ `fig5_confident_errors.png`, generated on Kaggle from the dataset |

### Figures from the first submission that must NOT be reused
- **CLAHE before/after figure:** the training code applies no CLAHE, so the figure and the CLAHE paragraph describe a step the model never saw.
- **Fold-1 confusion matrix "Acc 0.9950":** it has 12 errors, while the audited final fold 1 has 17 (99.30%). It comes from an earlier configuration, not the reported model. It is replaced by Fig. 4.
- **Workflow figure:** usable only after editing. Save it as `figures/fig_workflow.png` and recompile. It then appears automatically at the start of Section IV as a full-width figure, with a sentence that refers to it. Without the file, neither the figure nor the sentence appears. Remove the "Ablation Study" box (no ablations have been run yet), change "Multi-center" to "Public compiled", and move TTA from "Performance Evaluation" to inference. Export it at 300 dpi.

### Generating Fig. 6
These are generated from the dataset on Kaggle (CPU is enough):
```bash
python tools/make_paper_figures.py \
  --data-root "/kaggle/input/datasets/arafatrahmann/my-data221/Epic and CSCR hospital Dataset" \
  --pred fold_1_predictions.csv fold_2_predictions.csv fold_3_predictions.csv fold_4_predictions.csv fold_5_predictions.csv \
  --out figures
```
Copy only `fig5_confident_errors.png` into `paper/figures/` and recompile. The red placeholder box is then replaced automatically. Do not copy the script's `fig1_classes.png` or `fig3_graph_pyramid.png`, because Figs. 1 and 3 are already in place.

The eight images the figure will show (from the uploaded prediction CSVs), all under `Train/`:

| Fold | File | True → predicted | Confidence |
|---|---|---|---|
| 1 | `pituitary/Tr-pi_1115.jpg` | pituitary → glioma | 0.982 |
| 3 | `glioma/image(56).jpg` | glioma → no tumour | 0.981 |
| 3 | `glioma/image(18).jpg` | glioma → no tumour | 0.980 |
| 2 | `meningioma/Tr-me_1025.jpg` | meningioma → pituitary | 0.980 |
| 1 | `glioma/image(2).jpg` | glioma → meningioma | 0.978 |
| 2 | `glioma/image(54).jpg` | glioma → meningioma | 0.978 |
| 2 | `notumor/Tr-no_0093.jpg` | no tumour → pituitary | 0.977 |
| 4 | `meningioma/Tr-me_1263.jpg` | meningioma → glioma | 0.977 |
