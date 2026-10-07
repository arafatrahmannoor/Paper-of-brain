# Usage: python3 tools/cross_fold.py fold_1_predictions.csv fold_2_predictions.csv [...]
# Cross-fold checks on the per-image prediction files (pass them in fold order):
#   - test folds are pairwise disjoint (by path), coverage so far
#   - offline-augmentation siblings (G_710, G_710_RO_, ...) of each test image found in OTHER folds' test sets,
#     i.e. images that were in this fold's training/validation data -> direct, confirmed leakage
#   - accuracy of test images with vs without a confirmed sibling in training
#   - pooled out-of-fold metrics so far
import csv, sys, re, os, math, collections
C = ["glioma", "meningioma", "notumor", "pituitary"]
folds = [list(csv.DictReader(open(p))) for p in sys.argv[1:]]
K = len(folds)
paths = [set(r["path"] for r in f) for f in folds]
print("folds loaded:", K, "sizes:", [len(f) for f in folds])
for a in range(K):
    for b in range(a + 1, K):
        print(f"overlap fold {a+1} & fold {b+1}: {len(paths[a] & paths[b])}")
allp = set().union(*paths)
print("unique paths so far:", len(allp), "of 12064 (expected", sum(len(p) for p in paths), "if disjoint)")

def base(r):
    m = re.match(r"^([GMPN])_(\d+)", os.path.basename(r["path"]))
    return f'{r["true_class"]}/{m.group(1)}_{m.group(2)}' if m else None

where = collections.defaultdict(set)          # base id -> folds whose TEST set holds a copy
for k, f in enumerate(folds):
    for r in f:
        b = base(r)
        if b: where[b].add(k)
print()
for k, f in enumerate(folds):
    fam = [r for r in f if base(r)]
    leaked = [r for r in fam if where[base(r)] - {k}]
    clean = [r for r in f if not (base(r) and where[base(r)] - {k})]
    acc = lambda rs: sum(r["true_index"] == r["pred_index"] for r in rs) / max(1, len(rs))
    print(f"fold {k+1}: numbered-family test images {len(fam)}; with a sibling CONFIRMED in this fold's training data "
          f"(found in another fold's test set): {len(leaked)} ({len(leaked)/max(1,len(fam)):.1%})")
    print(f"         accuracy: confirmed-leaked {acc(leaked):.4f} (n={len(leaked)})  vs rest of fold {acc(clean):.4f} (n={len(clean)})")
mult = collections.Counter(len(v) for v in where.values())
print("\nnumbered source images seen so far:", len(where), "spread over how many test folds:", dict(sorted(mult.items())))
# pooled OOF so far
rows = [r for f in folds for r in f]
n = len(rows); y = [int(r["true_index"]) for r in rows]; yp = [int(r["pred_index"]) for r in rows]
cm = [[0]*4 for _ in range(4)]
for a, b in zip(y, yp): cm[a][b] += 1
F = []
for c in range(4):
    tp = cm[c][c]; p = tp / sum(cm[j][c] for j in range(4)); rc = tp / sum(cm[c]); F.append(2*p*rc/(p+rc))
acc = sum(a == b for a, b in zip(y, yp)) / n
z = 1.96; den = 1 + z*z/n; cen = (acc + z*z/(2*n)) / den; half = z*math.sqrt(acc*(1-acc)/n + z*z/(4*n*n)) / den
print(f"\npooled OOF so far: n={n} accuracy={acc:.4f} (95% Wilson [{cen-half:.4f}, {cen+half:.4f}]) macro-F1={sum(F)/4:.4f}")
print("pooled confusion matrix:", cm)
hc = [(os.path.basename(r["path"]), r["true_class"], r["pred_class"], max(float(r["prob_"+c]) for c in C))
      for r in rows if r["true_index"] != r["pred_index"]]
print("errors:", len(hc), " high-confidence errors (p>=0.95):", sum(1 for h in hc if h[3] >= 0.95))
for h in sorted(hc, key=lambda h: -h[3]):
    if h[3] >= 0.95: print("   %-18s %-10s -> %-10s p=%.3f" % h)
