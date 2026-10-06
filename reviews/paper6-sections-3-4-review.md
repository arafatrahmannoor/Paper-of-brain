# Reviewer report: Paper6.pdf (Sections 3–4, 8 pages)

**Scope:** Section 3 (Proposed Methodology) and Section 4 (Experimental Setup and Evaluation Protocol), the only sections in the uploaded draft. I did not have the final notebook, so I checked internal consistency between the two sections, the arithmetic, the methodology, and the presentation. I could not check every hyperparameter against the code.

**Overall score for Sections 3–4: 6.5 / 10.** In the original manuscript, Methodology scored about 4.5 and the evaluation protocol about 2.

This is a large improvement. The text now reads like a description of real code, the numbers agree between Sections 3 and 4, and the overclaims ("unbiased", "patient-level proxy", "dynamic k-NN", "512-d", CLAHE) are gone. What keeps it from 8+ is mostly one issue that is still not addressed at all, near-duplicate leakage in Version 1, plus several smaller technical issues a careful reviewer will find.

---

## A. Must fix before submission (blocking)

### A1. Version 1 near-duplicates are not mentioned anywhere (§3.1, §4.2)
The dataset authors' own Version 7 (BDNeuro-MRI) removed exact and near-duplicates from this release: **12,064 → 5,941 images, about 51% removed.** Sections 3.1 and 4.2 talk about patient identifiers only. Image-level StratifiedKFold on Version 1 will put near-copies of the same image on both sides of a fold. A reviewer who checks the dataset page will see Version 7 and its "leakage-free" wording, and will treat the five-fold result as inflated.

Options, best first:
1. Main experiment on Version 7 (official 70/15/15 split, or CV on train+val with the test set untouched).
2. Stay on Version 1, but run a perceptual-hash (pHash/dHash) near-duplicate audit. Report how many near-duplicate pairs cross train and test per fold, and add a **StratifiedGroupKFold run with groups = duplicate clusters**. Report both results.
3. At minimum, if neither is possible: state the Version 7 finding explicitly in §3.1 and in Limitations, and do not call the CV estimate a measure of generalization.

Suggested text for §3.1 if you take option 2:
> "A later release of the same dataset (Version 7) reports the removal of exact and near-duplicate images. To quantify the effect of such duplicates on the present protocol, perceptual hashes were computed for all 12,064 images, near-duplicate clusters were formed at a Hamming-distance threshold of ≤ X, and a duplicate-aware StratifiedGroupKFold evaluation was performed in addition to the standard image-level protocol."

### A2. Dataset counts and citation (§3.1)
- The per-class counts are missing. Add a small table of the counts the code actually read from disk. From the Fold-1 matrix these are most likely Glioma 3773, Meningioma 2729, No Tumor 2432, Pituitary 3130, and the Mendeley page lists the Meningioma/Pituitary test counts the other way round. Confirm against the notebook printout.
- Reference [19] must be the Mendeley dataset itself (DOI 10.17632/zwr4ntf94j, Version 1, with its authors), not Cheng 2017.
- The old manuscript said Version 1 already contains offline augmentation. If that is true, say so here, because it makes the duplicate problem worse.

### A3. All five folds must come from the final code
The 99.50% Fold-1 result in the old draft was produced *before* MC-Dropout, the two-token transformer and the TTA fix were added. Confirm that Fold 1 was re-run with the final code. Otherwise the five folds are not the same model.

Section 4.4 says the folds were run as separate notebooks. Add one sentence saying the fold assignment (file paths per fold) was saved and checked: the five test folds are pairwise disjoint and together cover all 12,064 images.

### A4. Saturation/hue jitter and the saturation TTA probably do nothing (§3.1, §3.10, §4.5)
The MRI images are grayscale, converted to RGB by copying the one channel into all three, so R = G = B. On such an image:
- **Saturation jitter (0.26) and hue jitter (0.05) have no effect.** A gray pixel has zero saturation, and hue is undefined.
- **The TTA "saturation factor 1.10" transform is the identity**, so "ten-way TTA" is really nine distinct views, with the identity view counted twice.

Check this quickly: apply `TF.adjust_saturation(img, 1.1)` to one image and compare it with the original. If they are identical, remove these from the paper (and ideally the code), or replace the saturation view with a genuinely different one (e.g. a small scale or crop) and re-run. A reviewer who knows image processing will spot this.

### A5. The equations are broken in the PDF
- Eqs. 18, 21, 22, 23 and 24 show **missing-glyph boxes (☐)** where the Unicode subscripts ᵢ, ₘ, ₖ should be. The font has no glyphs for them.
- All the equations are plain-text linear notation (`g_c`, `X_(l−1)`, `Σ_m α_m · h_m^(1)`), and there are hidden zero-width characters after several of them.

Retype every equation with the Google Docs or Word equation editor (or LaTeX). Journals will not accept the current form.

### A6. Sentences addressed to yourself are still in the paper
Remove or rewrite:
- §4.6: *"The current evaluation code does not compute ROC-AUC; consequently, AUC should not be reported unless it is explicitly added and recomputed…"* This is a note to yourself, not paper text. Either add AUC (one-vs-rest from the saved probabilities, which is easy) or say nothing.
- §3.7: *"rather than as independent single-token cross-attention sequences"* refers to an earlier version of your code that the reader never saw.
- §3.8 last paragraph (*"Accordingly, the resulting quantity is a feature-level MC-dropout uncertainty estimate…"*) reads defensive. Merge it into the description.
- In general, phrases like "in the implemented training pipeline", "the notebook", "deep-copied", "torch.Generator", "PIL image", "test_size = 0.08", "random_state = 42 + k" read like a code audit. Describe the method in plain scientific terms, and put the exact API-level settings in a short reproducibility paragraph or a supplementary table.

---

## B. Important technical weaknesses (a reviewer will ask)

### B1. MC-Dropout scope is overstated (§3.8)
Dropout is applied only to the 384-d embedding in front of a small 384→192→4 auxiliary head. That estimates the uncertainty of **that head only**. It is not the epistemic uncertainty of the backbone or GNN in the sense of Gal & Ghahramani, which requires dropout throughout the network. Use wording like *"head-level (last-layer) MC-dropout uncertainty"* and *"a lightweight proxy for epistemic uncertainty"*.

Also state:
- whether u_m is **detached** from the gradient when computing the weights;
- the value of ε in Eq. 14;
- the learned value of τ after training.

Note also that ũ_g + ũ_i = 2 by construction, so the weights depend only on the *ratio* of the two uncertainties. That is fine, but say it.

### B2. The calibration claim does not follow from the uncertainty module
ECE and Brier are computed from the **main classifier's** TTA-averaged probabilities, not from the MC-averaged auxiliary-head probabilities. Label smoothing (0.03) and TTA averaging both affect calibration. Do not attribute good ECE to the uncertainty gate unless an ablation (gate vs no gate) shows it. Report the distribution of w_g and w_i, e.g. a histogram or the mean ± SD per class, so readers can see whether the gate actually varies or collapses to about 0.5/0.5.

### B3. The graph topology is identical for every image (§3.3)
With a fixed 8-neighbour grid and fixed parent-child links, every image has the same graph: 336 nodes and **3,004 directed edges**. Intra-scale undirected edges are 42 + 210 + 930 = 1,182, and inter-scale edges are 64 + 256 = 320, giving 1,502 undirected, or 3,004 directed.

Only the 24-d node features change. A reviewer will ask why this beats a small CNN on a 3-level map of the same statistics. You need:
- (a) an ablation showing the graph branch adds accuracy or calibration over the CNN alone, and
- (b) ideally, one line arguing why message passing over a multi-resolution grid with cross-scale links is the right tool.

Also say:
- whether self-loops are added (PyG's GCNConv adds them by default);
- that the scale ratio r is always 2 here.

### B4. Node descriptor details (§3.3.1)
- Two features are redundant: the **median equals the 50th percentile**, and **variance is std²**. Either drop them or say they are kept intentionally.
- The thresholds defining "bright", "dark", "very bright", "mid-range" and "above-mean" pixels are not given, nor is the normalization (÷255? per image?). Without these the graph branch cannot be reproduced.

### B5. Sum pooling is redundant (§3.4)
Every graph has exactly 336 nodes, so global add pooling = 336 × global mean pooling. Mean and sum pooling carry the same information. This is harmless but noticeable: drop it, or justify it as a scale cue.

Also, Eq. 3 omits the BN, GELU and dropout (p = 0.20) that the text describes, so make it match Eq. 4.

### B6. The two-token transformer needs justification (§3.7)
With only two tokens, each attention head computes a 2×2 weighting, and three 384-d blocks with a 1,536 FFN add about **5.3 M parameters** to mix two vectors. Either:
- (a) justify it with an ablation (transformer vs a plain MLP or gated fusion at equal parameters), or
- (b) make it a true cross-modal transformer with several tokens per modality: EfficientNet spatial tokens (e.g. the 7×7 feature map gives 49 tokens) and graph node or scale tokens.

### B7. Dynamic recalibration is under-specified and unjustified (§3.6)
Hidden sizes and reduction ratios of MLP_g, MLP_i, MLP_s and MLP_c are not given. There are four gating networks stacked before a transformer and an uncertainty gate. Expect "over-engineered" comments unless the ablation shows each block helps.

### B8. Model selection on a saturated validation set (§4.2–4.3)
The validation set is about 6.4% of 12,064, roughly 770 images. At about 99.5% accuracy, validation accuracy differs between epochs by only one or two images, and ties, including at 100%, are likely. Early stopping on accuracy with patience 5 is therefore noisy. Use validation loss or NLL, or at least use loss as a tie-breaker, and state how ties are handled.

Also note: with OneCycleLR planned for 30 epochs, early stopping at epoch ~10–15 means the selected checkpoint is taken before the learning rate has annealed. Mention this, or report the epoch each fold stopped at.

### B9. Test-time stochasticity vs "reproducibility" (§4.4–4.5)
MC-Dropout is active during evaluation, so test predictions are random unless the dropout RNG is seeded at test time. PyG scatter operations on CUDA are also non-deterministic. Say that inference was seeded, or report how much the metrics vary across two evaluation runs.

### B10. Missing implementation facts (§4.1)
Even though the Kaggle GPU varied, reviewers expect:
- the GPU per fold (T4 or P100);
- PyTorch, torchvision and PyG versions;
- training time per fold;
- parameter count and FLOPs;
- inference time per image, both single pass and with 10 TTA views × 20 MC samples.

### B11. Ablation and baseline protocol is absent from Section 4
Section 4 should define *how* ablations and baselines are run: the same folds and seeds, the same epochs and early-stopping rule, and the same TTA (or none, for every model). Otherwise the later Results tables cannot be audited. Minimum ablation set:
- CNN only;
- GNN only;
- concatenation fusion;
- full model without the uncertainty gate (equal weights);
- full model without the transformer;
- full model.

### B12. Statistics (§4.6–4.7)
Mean ± SD over 5 folds is good. With about 12 errors per fold, also add:
- **95% bootstrap (or Wilson) CIs** on the pooled out-of-fold metrics;
- **McNemar's test** on out-of-fold predictions for each ablation or baseline comparison;
- per-class sensitivity and specificity, which matter clinically;
- optionally, one-vs-rest ROC-AUC from the saved probabilities.

State that ECE is top-label ECE, and that the Brier score sums over classes (range 0–2).

---

## C. Section-by-section ratings

| Section | Strengths | Weak points | Score |
|---|---|---|---|
| 3 (intro) | Clear summary, and synchronized augmentation across branches is a good detail | — | 8.0 |
| 3.1 Dataset & CV | Honest image-level framing; merging folders before CV is correct | Version 1 duplicates (A1); no class counts; wrong [19]; saturation/hue jitter no-op; "RGB" needs "grayscale replicated to 3 channels"; jitter schedule shape and original image resolution not stated | 5.0 |
| 3.2 Architecture | Clear sequence | Refers to an "overall architecture diagram" but the draft has **no figures**; add Figure 1 (architecture) and Figure 2 (graph pyramid) | 7.0 |
| 3.3 Graph pyramid | Fully specified scales, node count, patch sizes and edges; reproducible | Fixed topology (B3); redundant features and missing thresholds (B4); self-loops not stated | 7.0 |
| 3.4 Multi-path GNN | Learnable softmax path weights, heads and dims given | Eq. 3 incomplete; sum pooling redundant (B5); report learned α, β | 7.5 |
| 3.5 EfficientNet-B3 | Precise and correct (1,536-d, end-to-end fine-tuning) | B3's native resolution is 300; 224 is a deviation worth one clause | 8.5 |
| 3.6 Recalibration | Equations given | MLP sizes missing; heavy gating not justified (B7) | 6.5 |
| 3.7 Two-token transformer | Honest about two tokens; equations clear | Limited mixing with 2 tokens, ~5.3 M params, needs ablation (B6); stray reference to old design (A6) | 6.0 |
| 3.8 MC-dropout gate | Scalar uncertainty now properly defined (Eq. 13), which fixes the old gap | Scope overstated (B1); detach, ε and τ not given; calibration attribution (B2) | 6.5 |
| 3.9 Head & loss | Complete, with λ and label smoothing given | λ = 0.15 not justified (sensitivity check welcome) | 8.0 |
| 3.10 TTA | Deterministic list and Eq. 17 | Saturation view is identity (A4); was the TTA set chosen after seeing test results? | 7.0 |
| 4 (intro) | Good principle: protocol defined from code | — | 7.5 |
| 4.1 Environment | AMP noted | No GPU, versions, training time or complexity (B10) | 5.5 |
| 4.2 CV & test isolation | Correct test isolation; 73.6/6.4/20 split arithmetic right; clear limitation sentence | Duplicates (A1); saturated validation set (B8) | 6.0 |
| 4.3 Optimization + Table 2 | Complete, matches §3; Table 2 useful | Early stopping on accuracy; OneCycle + early stop interplay | 8.0 |
| 4.4 Reproducibility | Honest about non-bitwise determinism | Add fold disjointness/coverage check (A3); test-time dropout seeding (B9) | 7.0 |
| 4.5 Inference | Clear | Repeats §3.10 almost word for word; shorten to a reference plus the MC note | 6.5 |
| 4.6 Metrics | Macro averaging justified; ECE/Brier defined | Broken equations (A5); self-instruction sentence (A6); no CIs or tests (B12) | 6.0 |
| 4.7 Aggregation / OOF | Correct: no ensembling on the CV data; OOF pooling | Add CI, McNemar, and ablation/baseline protocol (B11–B12) | 7.5 |
| Presentation | Readable English, consistent numbers across §3 and §4 | Equations not typeset and show missing glyphs; no figures; code-audit tone | 4.5 |
| **Overall (§3–4)** | | | **6.5** |

**Consistency check (§3 vs §4):**
- The following match between the two sections: 30 epochs, batch 36, patience 5, the AdamW learning rates, weight decay 0.01, OneCycle pct_start 0.10, label smoothing 0.03, λ = 0.15, T = 20, p = 0.20, the 10 TTA transforms, the 8% validation split and seed 42.
- I found no internal contradictions.

---

## D. Path from 6.5 to 8+

1. Deal with duplicates: Version 7 rerun, or a pHash audit plus grouped CV (A1).
2. Add class counts, the correct dataset citation, and the fold-coverage check (A2, A3).
3. Fix the no-op saturation/hue augmentation and TTA, and re-run if you change the code (A4).
4. Retype all equations; add Figure 1 (architecture) and Figure 2 (graph pyramid) (A5).
5. Remove the self-notes and code-audit phrasing (A6).
6. Tone down the MC-dropout wording; report τ, the weight distribution, and detach behaviour (B1, B2).
7. Add the ablation/baseline protocol and the statistics plan (CI, McNemar) to Section 4 (B11, B12).
8. Add hardware, software, time and complexity details (B10).

---

## সংক্ষেপে (বাংলা)

- আগের draft-এর তুলনায় Section 3–4 অনেক ভালো হয়েছে: code-এর সঙ্গে মিলে, numbers consistent, ভুল claim সরানো হয়েছে। **স্কোর 6.5/10।**
- **সবচেয়ে বড় সমস্যা এখনো রয়ে গেছে:** Version 1-এ প্রায় ৫১% duplicate বা near-duplicate (Version 7 প্রমাণ করে), কিন্তু paper-এ এর কোনো উল্লেখ নেই।
- **Saturation/hue jitter আর TTA-র "saturation 1.10" grayscale MRI-তে কিছুই করে না।** তাই 10-way TTA আসলে 9-way। Code-এ একবার test করে দেখো।
- **Equation-গুলো PDF-এ ভাঙা (☐ box দেখাচ্ছে)।** Equation editor দিয়ে আবার লিখতে হবে। কোনো figure নেই, architecture diagram যোগ করো।
- "AUC should not be reported…" এ ধরনের নিজের জন্য লেখা note paper থেকে মুছে দাও।
- MC-Dropout শুধু ছোট auxiliary head-এ আছে, তাই এটাকে "head-level" uncertainty বলো, পুরো model-এর epistemic uncertainty না।
- Fold 1 কি final code দিয়ে আবার চালানো হয়েছে? না হলে Fold 1 আবার চালাতে হবে।
