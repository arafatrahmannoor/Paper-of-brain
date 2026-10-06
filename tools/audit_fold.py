# Usage: python3 tools/audit_fold.py fold_k_predictions.csv fold_k_confusion_matrix.csv fold_k_metrics.json fold_k_classification_report.json
# Recomputes the confusion matrix, macro P/R/F1, top-label ECE (10 bins), multiclass Brier, NLL and a Wilson CI
# from the per-image predictions, compares them with the saved files, and lists every error.
import csv, json, sys, re, math, collections
pred_path, cm_path, metrics_path, report_path = sys.argv[1:5]
C = ["glioma","meningioma","notumor","pituitary"]
rows = list(csv.DictReader(open(pred_path)))
n = len(rows); print("rows", n)
# integrity
idx = [int(r["sample_index"]) for r in rows]
print("unique sample_index", len(set(idx)), "unique paths", len(set(r["path"] for r in rows)))
probs = [[float(r["prob_"+c]) for c in C] for r in rows]
sums = [sum(p) for p in probs]
print("prob row sum min/max %.6f %.6f" % (min(sums), max(sums)))
bad_argmax = sum(1 for r,p in zip(rows,probs) if int(r["pred_index"]) != max(range(4), key=lambda k:p[k]))
print("pred_index != argmax(prob):", bad_argmax)
lab_mismatch = sum(1 for r in rows if C[int(r["true_index"])]!=r["true_class"] or C[int(r["pred_index"])]!=r["pred_class"])
print("index/name mismatch:", lab_mismatch)
folder_mismatch = sum(1 for r in rows if r["path"].split("/")[-2]!=r["true_class"])
print("folder != true_class:", folder_mismatch)
split = collections.Counter(r["path"].split("/")[-3] for r in rows); print("source split folders:", dict(split))
# recompute CM
cm = [[0]*4 for _ in range(4)]
for r in rows: cm[int(r["true_index"])][int(r["pred_index"])] += 1
saved = [[int(x) for x in l.split(",")] for l in open(cm_path).read().split() if l]
print("CM recomputed == saved:", cm == saved); print(cm)
y = [int(r["true_index"]) for r in rows]; yp=[int(r["pred_index"]) for r in rows]
acc = sum(a==b for a,b in zip(y,yp))/n
P=[];R=[];F=[]
for k in range(4):
    tp=cm[k][k]; fp=sum(cm[j][k] for j in range(4))-tp; fn=sum(cm[k])-tp
    p=tp/(tp+fp); rr=tp/(tp+fn); P.append(p); R.append(rr); F.append(2*p*rr/(p+rr))
# ECE top-label 10 bins, Brier
conf=[max(p) for p in probs]; ece=0
for b in range(10):
    lo,hi=b/10,(b+1)/10
    ids=[i for i,c in enumerate(conf) if (c>lo and c<=hi) or (b==0 and c==0)]
    if ids: ece+=len(ids)/n*abs(sum(y[i]==yp[i] for i in ids)/len(ids)-sum(conf[i] for i in ids)/len(ids))
brier=sum(sum((p[k]-(1 if y[i]==k else 0))**2 for k in range(4)) for i,p in enumerate(probs))/n
nll=-sum(math.log(max(probs[i][y[i]],1e-12)) for i in range(n))/n
m=json.load(open(metrics_path))
for name,val in [("accuracy",acc),("macro_precision",sum(P)/4),("macro_recall",sum(R)/4),("macro_f1",sum(F)/4),("ece",ece),("brier",brier)]:
    print(f"{name:16s} recomputed {val:.6f}  saved {m[name]:.6f}  diff {abs(val-m[name]):.2e}")
print("NLL %.4f" % nll)
print("mean max-prob %.4f, mean conf on correct %.4f, on wrong %.4f" % (sum(conf)/n, sum(c for c,a,b in zip(conf,y,yp) if a==b)/sum(a==b for a,b in zip(y,yp)), sum(c for c,a,b in zip(conf,y,yp) if a!=b)/max(1,sum(a!=b for a,b in zip(y,yp)))))
hist=collections.Counter(min(9,int(c*10)) for c in conf); print("confidence histogram (bin:count)", dict(sorted(hist.items())))
# Wilson CI
z=1.96; k=sum(a==b for a,b in zip(y,yp)); ph=k/n
den=1+z*z/n; cen=(ph+z*z/(2*n))/den; half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/den
print("accuracy Wilson 95%% CI [%.4f, %.4f]" % (cen-half, cen+half))
print("errors:")
for r,p in zip(rows,probs):
    if r["true_index"]!=r["pred_index"]:
        print("  ", r["path"].split("/")[-1], r["true_class"],"->",r["pred_class"], "p=%.3f"%max(p))
# filename patterns
names=[r["path"].split("/")[-1] for r in rows]
print("sample names:", names[:5], names[-3:])
suf=collections.Counter()
base=collections.Counter()
pat=re.compile(r"^(.*?)_(\d+)_?([A-Za-z]*)_?\.(\w+)$")
unmatched=[]
for nm in names:
    mm=pat.match(nm)
    if mm: suf[mm.group(3)]+=1; base[(mm.group(1),mm.group(2))]+=1
    else: unmatched.append(nm)
print("suffix counts:", suf.most_common(20))
print("unmatched names:", len(unmatched), unmatched[:10])
print("base ids appearing >1 times in this test fold:", sum(1 for v in base.values() if v>1), "of", len(base))
print("multiplicity dist:", collections.Counter(base.values()))
