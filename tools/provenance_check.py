# Usage (Kaggle: add the public dataset(s) as inputs first):
#   pip install -q imagehash
#   python tools/provenance_check.py OUR_DATASET_ROOT REFERENCE_ROOT [REFERENCE_ROOT ...] [--max-hamming 4]
# Example:
#   python tools/provenance_check.py \
#     "/kaggle/input/datasets/arafatrahmann/my-data221/Epic and CSCR hospital Dataset" \
#     /kaggle/input/brain-tumor-mri-dataset          # masoudnickparvar/brain-tumor-mri-dataset
#
# For every image in OUR dataset, finds the closest image (64-bit perceptual hash) in the reference
# dataset(s) and reports how many of our images also exist there (Hamming distance <= threshold),
# broken down by our filename family. Also reports exact byte-identical files (MD5).
# Output: provenance_matches.csv in the current directory.
import os, re, sys, csv, hashlib, collections
import numpy as np
from PIL import Image
import imagehash

args = [a for a in sys.argv[1:] if not a.startswith("--")]
max_h = int(sys.argv[sys.argv.index("--max-hamming") + 1]) if "--max-hamming" in sys.argv else 4
if "--max-hamming" in sys.argv:
    args.remove(str(max_h))
ours_root, ref_roots = args[0], args[1:]
EXT = (".jpg", ".jpeg", ".png")


def files(root):
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(EXT))


def fingerprint(path):
    with open(path, "rb") as fh:
        md5 = hashlib.md5(fh.read()).hexdigest()
    with Image.open(path) as im:
        ph = int(str(imagehash.phash(im.convert("L"))), 16)
    return md5, ph


def family(name):
    for pat, lab in [(r"^[GMPN]_\d+", "G/M/P/N_### (+aug)"), (r"^Tr-\w\w_\d+", "Tr-xx_####"), (r"^Te-\w\w_\d+", "Te-xx_####"),
                     (r"^(gg|m\d?|p) ?\(\d+\)", "gg/m/p (n)"), (r"^image ?\(\d+\)", "image(n)"),
                     (r"^(\d+ ?no|N\d+|no ?\d+)", "no/N"), (r"^\d+\.jpe?g$", "#.jpg")]:
        if re.match(pat, name, re.I):
            return lab
    return "other"


ours = files(ours_root)
refs = [p for r in ref_roots for p in files(r)]
print(f"our images: {len(ours)} | reference images: {len(refs)}")
o_fp = [fingerprint(p) for p in ours]
r_fp = [fingerprint(p) for p in refs]
r_md5 = {m: p for p, (m, _) in zip(refs, r_fp)}
r_ph = np.array([h for _, h in r_fp], dtype=np.uint64)
lut = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)

stats = collections.defaultdict(lambda: [0, 0, 0])  # n, exact md5, phash<=max_h
with open("provenance_matches.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["our_path", "family", "exact_md5_match", "nearest_ref_path", "hamming"])
    for p, (md5, ph) in zip(ours, o_fp):
        x = np.bitwise_xor(r_ph, np.uint64(ph))
        d = np.bitwise_count(x) if hasattr(np, "bitwise_count") else lut[x.view(np.uint8).reshape(-1, 8)].sum(1)
        j = int(np.argmin(d))
        fam = family(os.path.basename(p))
        exact = md5 in r_md5
        stats[fam][0] += 1
        stats[fam][1] += int(exact)
        stats[fam][2] += int(d[j] <= max_h)
        w.writerow([p, fam, int(exact), refs[j], int(d[j])])

print(f"\n| Our filename family | n | exact file in reference | near-identical (pHash <= {max_h}) |")
print("|---|---|---|---|")
tot = [0, 0, 0]
for fam, (n, e, m) in sorted(stats.items(), key=lambda t: -t[1][0]):
    print(f"| {fam} | {n} | {e} ({e / n:.0%}) | {m} ({m / n:.0%}) |")
    tot = [a + b for a, b in zip(tot, (n, e, m))]
print(f"| **total** | {tot[0]} | {tot[1]} ({tot[1] / tot[0]:.0%}) | {tot[2]} ({tot[2] / tot[0]:.0%}) |")
print("\nDetails: provenance_matches.csv")
