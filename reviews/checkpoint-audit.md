# Checkpoint audit: `brain_tumor_model_fold_{1..5}.pth`

Sources:
- The five checkpoint reports you supplied. Fold 2's includes the full key/shape manifest.
- The audited prediction files (`final-cv-results.md`).
- Section 3 of `Paper6.pdf`.

I could not load the `.pth` files myself. Everything below was derived from the reported shapes, counts and values, and every count was cross-checked arithmetically.

## 1. Bottom line

1. **The checkpoints match the Methodology section in every point that can be checked.** No discrepancy was found between Paper6 §3 and the saved weights (§2).
2. **The paper needs the real parameter count: 19,350,155 (19.35 M).** The old draft's 15.86 M is wrong. The cross-modal transformer alone holds **27.5%** of all parameters, while the graph branch holds only **3.2%** (§3).
3. **The "learnable multi-path GNN weighting" learned essentially nothing.** softmax(α) and softmax(β) stay within ±0.006 of an equal ⅓/⅓/⅓ split in both inspected folds. The paper must not claim the model learns to prefer GCN, GAT or TransformerConv (§4).
4. **The checkpoints can diagnose Fold 5 without any training log.** BatchNorm's `num_batches_tracked` counter records how many training batches ran before the best checkpoint was saved. Run `tools/inspect_checkpoints.py` on all five files (§6).
5. **The handoff reports contain a few errors.** For example, they call GAT1 "concatenated (512)", but its bias is 128-wide, which means the heads are averaged, as the paper says. They also miss things already established: the class order is known from the prediction files, and the metrics are already independently verified (§5).

## 2. Paper §3 vs checkpoint: consistency check

| Paper6 claim | Checkpoint evidence | Verdict |
|---|---|---|
| EfficientNet-B3, 1,536-d pooled features, ImageNet-pretrained, fine-tuned end-to-end | 40-channel stem `[40,3,3,3]`, final conv `[1536,384,1,1]`. Feature-extractor parameters total **10,696,232**, exactly the torchvision EfficientNet-B3 `features` count | ✅ torchvision EfficientNet-B3 confirmed |
| Projection 1536→768 (BN, GELU, dropout 0.3) →384 (BN, GELU) | `projection.1 [768,1536]`, `.2` BN(768), `.5 [384,768]`, `.6` BN(384) | ✅ |
| 24-d node descriptors → 128 (BN, GELU) | `g_enc.input_proj.0 [128,24]`, `.1` BN(128) | ✅ |
| Two stages of GCN + GAT + TransformerConv at width 128 | `gcn1/gcn2.lin [128,128]`, `gat1/gat2`, `trans1/trans2` | ✅ |
| GAT and TransformerConv: 4 heads (stage 1), 2 heads (stage 2), **non-concatenated** | `gat1.att_src [1,4,128]`, `gat2 [1,2,128]`; **biases are [128]** and `lin_skip` is `[128,128]`, so `concat=False` (heads averaged). No `lin_beta`, so `beta=False` | ✅ |
| Learnable path weights α, β via softmax | `g_enc.alpha [3]`, `g_enc.beta [3]` | ✅ (but see §4) |
| Residual between stages with BN | `g_enc.bn_residual` | ✅ |
| Mean ‖ max ‖ add pooling → 3×128 = 384 → 384 (BN, GELU) | `output_head.0 [384,384]`, `.1` BN(384) | ✅ |
| Recalibration gates (Eqs. 5–8) | `cg_g`, `cg_i`: 384→**24**→384; `sg`: 384→**96**→384; `cmg`: 768→384→384; `ln_g`, `ln_i` LayerNorm(384) | ✅ The hidden sizes are now known (§7). |
| Two-token transformer, 3 layers, FFN ×4 | `trans.modality_embedding [1,2,384]`, `layers.0–2`, `ffn 384→1536→384`, `ln1`, `ln2` | ✅ The head count (8) is not stored in a state dict. Keep the code's value. |
| Auxiliary heads 384→192→4 and learnable temperature | `uncert.g_head`, `uncert.i_head` 384→192→4; `uncert.log_tau` (scalar) | ✅ (see §4 on naming) |
| Classifier 768→512 (BN, GELU, dropout 0.4) →256 (BN, GELU, dropout 0.3) →4 | `clf.0 [512,768]`, `clf.1` BN, `clf.4 [256,512]`, `clf.5` BN, `clf.8 [4,256]`. Indices 3 and 7 are parameter-free (dropout) | ✅ |
| All 5 folds use the same architecture | 751 keys, identical shapes, 78,064,422 bytes each. SHA-256 prefixes differ (fold 2 `cbf380d3…`, fold 3 `a44d7929…`, fold 4 `ddf6a9af…`, fold 5 `25d1d124…`), and all tensor values differ between folds | ✅ Five distinct trained models |

Class index order, verified from the prediction files rather than the checkpoint: **0 = glioma, 1 = meningioma, 2 = notumor, 3 = pituitary** (alphabetical folder order).

## 3. Parameter count (for the paper's complexity table)

All values are computed from the fold-2 manifest and exclude BatchNorm running statistics. Each module total reconciles exactly with the reported per-prefix element counts.

| Module | Parameters | Share |
|---|---|---|
| Image stream (`i_enc`): EfficientNet-B3 features 10,696,232 + projection 1,478,016 | 12,174,248 | 62.9% |
| Graph stream (`g_enc`) | 617,222 | 3.2% |
| Dynamic recalibration (`recal`) | 557,520 | 2.9% |
| Cross-modal transformer (`trans`) | **5,324,160** | **27.5%** |
| Uncertainty heads + τ (`uncert`) | 149,385 | 0.8% |
| Classifier (`clf`) | 527,620 | 2.7% |
| **Total** | **19,350,155** | 100% |

- **Buffers:** 94,043 elements (91 BatchNorm layers × running mean, running variance and step counter).
- **Size:** about 78 MB in float32.
- **Action:** replace "15.86 M parameters" everywhere with **19.35 M**. FLOPs and latency must be measured again; they cannot be read from weights.
- **Reviewer impact:** the 5.3 M-parameter transformer mixes only **two tokens**. It is larger than everything except the CNN backbone, while the graph branch, the paper's main novelty, is 3% of the model. This strengthens review point B6 (`paper6-sections-3-4-review.md`). An ablation "transformer → simple MLP/gated fusion" is now essential.

## 4. What the learned values say

### 4.1 Multi-path weights α, β are effectively uniform

| | GCN | GAT | TransformerConv |
|---|---|---|---|
| Fold 1 softmax(α) (stage 1) | 0.3281 | 0.3332 | 0.3386 |
| Fold 1 softmax(β) (stage 2) | 0.3328 | 0.3368 | 0.3304 |
| Fold 2 softmax(α) | 0.3310 | 0.3336 | 0.3354 |
| Fold 2 softmax(β) | 0.3329 | 0.3367 | 0.3305 |

(The path order assumes the code orders α as GCN, GAT, TR. Confirm this in `forward()`.)

- All eight learned weights lie within 0.328–0.339, i.e. ±0.006 of ⅓.
- The "learnable fusion coefficients" stayed at their initialisation in practice. The likely causes are weak gradients through the softmax combined with weight decay of 0.01 pulling the logits back towards 0.
- **Paper change (§3.4):** do not describe this as the network learning which graph operator matters. Either state honestly that the learned coefficients converged to near-uniform weights (≈ ⅓ each, reported in a table), so the multi-path block behaves as an equal-weight ensemble of the three operators, or replace it with a fixed average. Then run the ablation "single operator vs three-path average" to show whether three paths help at all.
- Fold 2's β is almost identical to fold 1's (to about 0.0005), so the coefficients move in the same tiny, data-independent way in every fold. That is further evidence they carry no meaningful selection.

### 4.2 Uncertainty temperature τ

| | `log_tau` | τ if `exp(log_tau)` | τ if `softplus(log_tau)` |
|---|---|---|---|
| Fold 1 | 0.1377 | 1.148 | 0.764 |
| Fold 2 | 0.1537 | 1.166 | 0.773 |

- **Naming conflict:** the parameter is called `log_tau`, which suggests τ = exp(log_tau), but Paper6 Eq. 15 says τ is parameterised by softplus. Check `forward()` and make the paper match the code.
- **What τ allows:** with Eq. 14 the normalised uncertainties always sum to 2, so the gate can move the weights only from 0.50/0.50 up to about 0.93/0.07 with softplus, or about 0.85/0.15 with exp, and only when one modality has near-zero uncertainty. A modest 10% uncertainty difference (ũ = 0.9 vs 1.1) gives only about 0.56/0.44.
- **Action:** log the per-image weights `w_g`, `w_i` on one fold's test set and report their distribution. If they sit at about 0.5/0.5 for almost every image, the uncertainty gate is not doing meaningful work, and the paper must say so, or the ablation will show it anyway.

## 5. Corrections to the handoff reports

| Report statement | Correction |
|---|---|
| Fold 1: "First GAT block: 4 heads … concatenated width 512" | `lin.weight [512,128]` always has heads × out rows in PyG. The **bias [128]** proves `concat=False` (heads averaged), exactly as the paper says. |
| Fold 1: "floating-point parameter values: 19,350,155" | 19,350,155 is the **non-buffer parameter count**. Floating-point elements total 19,444,107 (fold 2 report, correct). |
| All: "class names / index order unknown" | Known and verified from the five prediction files: 0 glioma, 1 meningioma, 2 notumor, 3 pituitary. |
| All: "no metrics can be derived; supply held-out predictions" | Already done. All 12,064 out-of-fold predictions were audited and every metric was reproduced exactly (`final-cv-results.md`). The `.pth` files are not needed for the paper's metrics. |
| All: "epoch number is not stored" | Not stored directly, but **recoverable** from BatchNorm `num_batches_tracked` (§6). |
| All: "check patient-level splitting" | Patient IDs do not exist in this dataset. The real issue, augmented-copy leakage, has already been measured (35.2% of images; no accuracy gap). |

The generic "copy-and-paste prompts" in those reports ask for work that is already complete: the fold audit, out-of-fold metrics, leakage analysis and the Section 3–4 review. The only new evidence the checkpoints add is in §3, §4 and §6.

## 6. Diagnose Fold 5 from the checkpoints (2 minutes on Kaggle)

```bash
python tools/inspect_checkpoints.py brain_tumor_model_fold_{1,2,3,4,5}.pth
```

- `num_batches_tracked` increases by 1 for every training-mode batch. Because the saved file is the deep-copied best checkpoint, the counter shows **how many batches had run when the best validation accuracy was reached**. With about 247 batches per epoch (8,879 training images ÷ 36), that is the best epoch.
- **If Fold 5's best epoch is much earlier than folds 1–4**, early stopping picked an immature checkpoint. Validation accuracy saturates on about 770 images (review point B8), so this is plausible. Report it as a documented cause: "Fold 5's checkpoint was selected at epoch k because validation accuracy plateaued early." The real fix is a methodological change, applied to all folds, not just Fold 5: select on validation loss.
- **If the epochs are similar**, the difference is training variability, and Fold 5 is reported as it is.
- The script also prints α, β, τ and the per-module parameter counts for all five folds, so §3 and §4 can be confirmed for folds 3–5.

## 7. Paper edits enabled by this audit

1. §3.5–3.9 / complexity table: **19.35 M parameters** with the per-module table from §3. Remove 15.86 M.
2. §3.6: add the gate hidden sizes. Modality gates 384→24→384 (reduction 16); shared gate 384→96→384 with LayerNorm; cross-modal gate 768→384→384 with LayerNorm. These were missing (review point B7).
3. §3.4: report the learned α/β values and drop any wording that implies the network learned operator preferences.
4. §3.8: make "softplus" vs "exp(log_tau)" match the code, report the learned τ (≈ 0.76 or ≈ 1.15), and report the distribution of w_g and w_i.
5. Ablations are now clearly justified, given the parameter shares and flat path weights:
   - (a) transformer → MLP fusion;
   - (b) three-path GNN → single GCN;
   - (c) uncertainty gate → fixed 0.5/0.5;
   - (d) image-only.

---

## সংক্ষেপে (বাংলা)

- ৫টা checkpoint-ই paper-এর Methodology (Section 3)-র সঙ্গে পুরোপুরি মিলে। কোনো অমিল পাইনি। ৫টাই আলাদা আলাদা trained model।
- **আসল parameter সংখ্যা 19.35M**, আগের draft-এর 15.86M ভুল। শুধু Transformer-এই 27.5% parameter (মাত্র ২টা token মেশাতে), আর graph branch মাত্র 3.2%।
- **GCN/GAT/Transformer path-এর learnable weight (α, β) আসলে কিছুই শেখেনি**, সব ⅓-এর কাছাকাছি রয়ে গেছে। তাই paper-এ "model শিখেছে কোন path বেশি গুরুত্বপূর্ণ" বলা যাবে না।
- `log_tau` নাম আর paper-এর "softplus" মিলছে না। Code দেখে ঠিক করো।
- **Fold 5 কেন দুর্বল তা training log ছাড়াই বের করা যায়।** `tools/inspect_checkpoints.py` Kaggle-এ চালাও, এটা দেখাবে best checkpoint কোন epoch-এ save হয়েছিল। Output আমাকে পাঠাও।
- অন্য AI-এর report-গুলোতে কিছু ভুল আছে (যেমন GAT "concatenated 512" নয়, heads গড় করা হয়েছে)। আর ওরা যা চেয়েছে (metrics, class order) সেগুলো আমরা আগেই verify করে ফেলেছি।
