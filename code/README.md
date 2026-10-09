# Code

| File | What it is |
|---|---|
| `brain_tumor_cv.py` | **v1**: the exact script that produced the five image-level folds (99.15 ± 0.39%). Kept unchanged for reproducibility. |
| `brain_tumor_cv_v2.py` | **v2**: same model and hyperparameters, plus duplicate-aware grouped CV, validation-loss checkpoint selection, TTA/augmentation fixes, ablation switch, gate logging and complexity measurement. See the header of the file and `reviews/code-audit.md` for why each change was made. |

## Running v2 on Kaggle
1. Add `imagehash` (first cell: `!pip install -q imagehash`).
2. In the CONFIGURATION block, set `FOLD_TO_RUN` (0–4) and `ABLATION`, then run. Everything else stays fixed.
3. Download `/kaggle/working/brain_tumor_results_v2/` after each run. Files are prefixed with `{CV_MODE}_{ABLATION}_fold_{k}_…`, so runs never overwrite each other. Upload these to me:
   - `*_predictions.csv`, which now includes the gate weights `w_graph`/`w_image`, the uncertainties and the group id;
   - `*_metrics.json`, which now includes the best epoch, the per-epoch history, τ, softmax(α/β) and complexity;
   - `groups.csv`;
   - `near_duplicate_label_conflicts.csv`;
   - `fold_assignment_grouped.csv`.

   You don't need to upload the `.pth` files.
4. **After the first run, before anything else:** send me the `✅ Groups: …` line, the `near_duplicate_label_conflicts.csv` file and the 5 lines `Fold k: … groups shared with training: 0`. If pHash produced one giant group, lower `PHASH_MAX_HAMMING` and rerun. This costs about 3 minutes of CPU, so check it before spending GPU hours.

## Run plan (priority order)

| Priority | `ABLATION` | Folds | Purpose |
|---|---|---|---|
| 1 | `full` | 0–4 | **New main result** (leak-free grouped CV) |
| 2 | `image_only` | 0–4 | EfficientNet-B3 baseline; does the graph branch add anything? |
| 3 | `concat` | 0–4 | Plain fusion vs the full fusion stack |
| 4 | `no_uncert` | 0–4 | Does the MC-dropout gate help? (gate fixed at 0.5/0.5) |
| 5 | `no_trans` | 0–4 | Is the 5.3 M-parameter, 2-token transformer needed? |
| 6 | `graph_only` | 0–4 | Graph branch alone |
| 7 | `single_gcn` | 0–4 | Three GNN operators vs GCN only (α/β learned ≈ ⅓) |
| 8 | `no_recal` | 0–4 | Recalibration gates |

- **Compute:** Kaggle gives about 30 GPU-hours per week. Note the runtime of one `full` fold, which is printed at the end, and plan from it. Priorities 1–4 are the minimum for a credible paper. 5–8 are strongly recommended.
- **Statistics:** all runs share the same grouped folds (`fold_assignment_grouped.csv`), so the out-of-fold predictions can be compared pairwise with McNemar's test. I'll compute this from the uploaded `*_predictions.csv` files.

## Smoke test (any machine, CPU)
```bash
BT_SMOKE=1 BT_PRETRAINED=0 BT_DATA_PATH=/path/to/tiny_dataset BT_OUTPUT_DIR=/tmp/out BT_ABLATION=full python code/brain_tumor_cv_v2.py
```
`BT_SMOKE=1` shrinks the run (2 epochs, batch 8, 3 TTA views, 3 MC passes, 0 workers). `BT_COMPLEXITY=0` skips the latency measurement.
