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
- repository URL;
- funding and competing-interests statement;
- acknowledgment / author contributions / AI-tool disclosure;
- contributor names for the Version 7 Mendeley record (`bdneuro2025v7` in `references.bib`).

## Figures 1, 3 and 5
These are generated from the dataset on Kaggle (CPU is enough):
```bash
python tools/make_paper_figures.py \
  --data-root "/kaggle/input/datasets/arafatrahmann/my-data221/Epic and CSCR hospital Dataset" \
  --pred fold_1_predictions.csv fold_2_predictions.csv fold_3_predictions.csv fold_4_predictions.csv fold_5_predictions.csv \
  --out figures
```
Copy the three PNGs (`fig1_classes.png`, `fig3_graph_pyramid.png`, `fig5_confident_errors.png`) into `paper/figures/` and recompile. The red placeholder boxes are then replaced automatically; no edit to `main.tex` is needed.
