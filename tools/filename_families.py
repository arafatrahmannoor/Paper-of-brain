# Usage: python3 tools/filename_families.py fold_k_predictions.csv
# Groups test images by filename convention and offline-augmentation suffix (ORIG/RO/VF/HF/BR/SP/DA).
import csv, sys, re, collections
rows=list(csv.DictReader(open(sys.argv[1])))
def family(nm):
    if re.match(r"^[GMPN]_\d+", nm): return "X_###[_AUG_].jpg (numbered)"
    if re.match(r"^Tr-\w\w_\d+", nm): return "Tr-xx_####.jpg"
    if re.match(r"^Te-\w\w_\d+", nm): return "Te-xx_####.jpg"
    if re.match(r"^gg \(\d+\)", nm): return "gg (n).jpg"
    if re.match(r"^m\d? \(\d+\)", nm): return "m (n).jpg"
    if re.match(r"^p \(\d+\)", nm): return "p (n).jpg"
    if re.match(r"^image ?\(\d+\)", nm): return "image(n).jpg"
    if re.match(r"^no ?\d*", nm, re.I): return "no*.jpg"
    return "other"
fam=collections.Counter(); famcls=collections.defaultdict(collections.Counter); other=collections.Counter()
err=collections.Counter()
for r in rows:
    nm=r["path"].split("/")[-1]; f=family(nm); fam[f]+=1; famcls[f][r["true_class"]]+=1
    if f=="other": other[re.sub(r"\d+","#",nm)]+=1
    if r["true_index"]!=r["pred_index"]: err[f]+=1
for f,c in fam.most_common(): print(f"{f:30s} {c:5d}  errors={err[f]:2d}  {dict(famcls[f])}")
print("other patterns:", other.most_common(15))
# numbered family: prefix per class, suffix groups
groups=collections.defaultdict(list)
for r in rows:
    nm=r["path"].split("/")[-1]
    m=re.match(r"^([GMPN])_(\d+)_?([A-Z]*)_?\.", nm)
    if m: groups[(r["true_class"],m.group(1),int(m.group(2)))].append(m.group(3) or "ORIG")
mult=[len(v) for v in groups.values()]
print("numbered bases in fold:", len(groups), "images:", sum(mult), "mean multiplicity %.3f"%(sum(mult)/len(mult)))
# estimate k (copies per base) assuming p=0.2 zero-truncated binomial
p=0.2
for k in range(1,9):
    e=k*p/(1-(1-p)**k); print(f"  k={k}: E[mult|>=1]={e:.3f}")
ex=[ (k,v) for k,v in groups.items() if len(v)>=3][:8]
for k,v in ex: print("  ",k,v)
idrange=collections.defaultdict(list)
for (c,pref,i) in groups: idrange[(c,pref)].append(i)
for k,v in idrange.items(): print("  id range",k,min(v),max(v),len(v))
