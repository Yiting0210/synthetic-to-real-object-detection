"""Confidence threshold on top of the 21-class model (no retraining).

Rule, fixed before looking at test or unseen results:
    the model predicts a course ID only if that ID's probability is at least T; otherwise the answer is OTHERS.
    T is chosen on VALIDATION ONLY: the largest T at which at most 1% of the validation course images are rejected
    (false rejection <= 1%), so recognition of the 20 objects stays practically unchanged.
Then T is applied once to the test images and to the unseen-object photos.

    python classification/src/others_threshold.py        (repository root, laptop CPU, ~2 min)

Rebuilds the exact inputs of the Colab run m1o_20261107_113839 (M1 split for the 20 classes; same import, square crop,
pHash groups and seeded split for OTHERS) and first checks that the plain 21-class predictions reproduce the Colab numbers.
Writes classification/results/others/threshold/.
"""
import csv, json, math, re
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "detection" / "cluster" / "data" / "raw"
M1_DATA = ROOT / "detection" / "cluster" / "IE7615_project" / "runs" / "m1_20261018_022316" / "data"
OTHERS_ROOT = ROOT / "classification" / "others"
MODEL = ROOT / "models" / "mobilenetv2_others.keras"
CLASSES_FILE = ROOT / "models" / "classes_with_others.json"
OUT = ROOT / "classification" / "results" / "others" / "threshold"
OUT.mkdir(parents=True, exist_ok=True)
WORK = OUT / "_work"
SEED, IMG_SIZE, PHASH_THRESHOLD, SPLIT_RATIOS = 42, 224, 4, (0.70, 0.15, 0.15)
MAX_FALSE_REJECTION = 0.01
COLAB = dict(val_known=(301, 303), test_known=(302, 303), test_others=(12, 15), unseen=(15, 36))   # m1o_20261107_113839
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}

import tensorflow as tf
import keras

CLASSES = {int(k): v for k, v in json.loads(CLASSES_FILE.read_text(encoding="utf-8")).items()}
OTH = [k for k, v in CLASSES.items() if v == "OTHERS"][0]

# ---------------------------------------------------------------- inputs, exactly as in m1_others.ipynb
def load_square(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((left, top, left + s, top + s)).resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

def kind_of(name):
    stem = re.sub(r"^(oth|uns)_", "", re.sub(r"\s*\(\d+\)$", "", Path(name).stem.lower().strip()))
    return "empty" if stem.startswith("empty") else (re.sub(r"[_\-\s]*\d+$", "", stem) or stem)

def import_folder(folder, prefix):
    (WORK / prefix).mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT and not f.name.startswith("."))
    out = []
    for i, f in enumerate(files):
        dst = WORK / prefix / f"{prefix}_{i:04d}.jpg"
        load_square(f).save(dst, quality=95)
        out.append(dict(name=f.name, kind=kind_of(f.name), path=str(dst)))
    return out

D = np.cos(np.pi * (2 * np.arange(32)[None, :] + 1) * np.arange(32)[:, None] / 64) * np.sqrt(2 / 32)
D[0] /= np.sqrt(2)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
def phash(path):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = (D @ a @ D.T)[:8, :8]
    return np.packbits((low > np.median(low)).flatten())

others = import_folder(OTHERS_ROOT / "images_OTHERS", "OTHERS")
unseen = import_folder(OTHERS_ROOT / "unseen_test", "UNSEEN")
P = np.stack([phash(r["path"]) for r in others])
H = POP[np.bitwise_xor(P[:, None, :], P[None, :, :])].sum(-1)
parent = list(range(len(others)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a, b in np.argwhere(H <= PHASH_THRESHOLD):
    if a < b:
        parent[find(a)] = find(b)
groups = {}
for i in range(len(others)):
    groups.setdefault(find(i), []).append(i)
rng = np.random.default_rng([SEED, OTH])
order = rng.permutation(len(groups))
glist = sorted((list(groups.values())[k] for k in order), key=len, reverse=True)
n = len(others); targets = [round(n * SPLIT_RATIOS[0]), round(n * SPLIT_RATIOS[1])]; targets.append(n - sum(targets))
counts = [0, 0, 0]
for m in glist:
    k = int(np.argmax([(targets[j] - counts[j]) / max(targets[j], 1) for j in range(3)]))
    counts[k] += len(m)
    for i in m:
        others[i]["split"] = ["train", "val", "test"][k]

folders = {p.name.strip(): p for p in RAW.iterdir() if p.is_dir()}
course = []
for r in csv.DictReader(open(M1_DATA / "split_manifest.csv", newline="", encoding="utf-8")):
    if r["split"] in ("val", "test"):
        folder, name = r["src_path"].split("/", 1)
        course.append(dict(name=r["image_id"], split=r["split"], label=int(r["class_idx"]), path=str(folders[folder.strip()] / name)))

# ---------------------------------------------------------------- predictions (same decode as the notebook)
model = keras.models.load_model(MODEL, compile=False)
def probs(paths):
    def dec(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize_with_pad(tf.cast(img, tf.float32), IMG_SIZE, IMG_SIZE)
        return tf.clip_by_value(tf.round(img), 0, 255)
    ds = tf.data.Dataset.from_tensor_slices(paths).map(dec).batch(32)
    return model.predict(ds, verbose=0)

def subset(split):
    rows = [r for r in course if r["split"] == split] + [dict(r, label=OTH) for r in others if r["split"] == split]
    return rows, np.array([r["label"] for r in rows]), probs([r["path"] for r in rows])

val_rows, y_val, p_val = subset("val")
test_rows, y_test, p_test = subset("test")
p_uns = probs([r["path"] for r in unseen])

# ---------------------------------------------------------------- 1) reproduce the Colab numbers (plain argmax)
def counts_at(y, p, t):
    """Predictions with threshold t: a course ID only if its probability >= t, else OTHERS."""
    pred = p.argmax(1)
    pred = np.where((pred != OTH) & (p.max(1) < t), OTH, pred)
    known = y != OTH
    return dict(known_correct=int(((pred == y) & known).sum()), known=int(known.sum()),
                rejected_known=int(((pred == OTH) & known).sum()),
                others_correct=int(((pred == OTH) & ~known).sum()), others=int((~known).sum()))
def unseen_at(t):
    pred = p_uns.argmax(1)
    pred = np.where((pred != OTH) & (p_uns.max(1) < t), OTH, pred)
    return int((pred == OTH).sum()), pred

v0, t0 = counts_at(y_val, p_val, 0), counts_at(y_test, p_test, 0)
got = dict(val_known=(v0["known_correct"], v0["known"]), test_known=(t0["known_correct"], t0["known"]),
           test_others=(t0["others_correct"], t0["others"]), unseen=(unseen_at(0)[0], len(unseen)))
print("reproduces the Colab run:", got == COLAB, got)
if got != COLAB:
    raise SystemExit("Local predictions differ from the Colab run; the threshold would not apply to that model. Stop.")

# ---------------------------------------------------------------- 2) choose T on validation only
cand = np.round(np.arange(0.50, 0.996, 0.01), 2)
sweep = []
for t in cand:
    v = counts_at(y_val, p_val, t)
    sweep.append(dict(t=float(t), val_false_rejection=v["rejected_known"] / v["known"], val_known_acc=v["known_correct"] / v["known"],
                      val_others_recall=v["others_correct"] / v["others"]))
ok = [s for s in sweep if s["val_false_rejection"] <= MAX_FALSE_REJECTION]
T = max(s["t"] for s in ok) if ok else 0.0
chosen = next(s for s in sweep if s["t"] == T)
print(f"T chosen on validation: {T} (val false rejection {chosen['val_false_rejection']:.4f}, val OTHERS recall {chosen['val_others_recall']:.3f})")
(OUT / "threshold.json").write_text(json.dumps(dict(
    rule=f"course ID only if its probability >= T, else OTHERS; T = largest value in 0.50..0.99 with validation false rejection <= {MAX_FALSE_REJECTION:.0%}",
    T=T, chosen_on="validation", validation=chosen, model_sha256_prefix="9deb2081"), indent=2), encoding="utf-8")

# ---------------------------------------------------------------- 3) apply T once to test and unseen
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0, c - h), min(1, c + h)
def fmt(k, n):
    lo, hi = wilson(k, n); return f"{k / n:.4f} ({k}/{n}; CI {lo:.3f}–{hi:.3f})"

rows = []
for name, t in [("no threshold", 0.0), (f"T = {T}", T)]:
    ts = counts_at(y_test, p_test, t); u, _ = unseen_at(t)
    rows.append([name, fmt(ts["known_correct"], ts["known"]), fmt(ts["rejected_known"], ts["known"]),
                 fmt(ts["others_correct"], ts["others"]), fmt(u, len(unseen))])
table = ("| | test known accuracy | test false rejection | test OTHERS recall | unseen rejected as OTHERS |\n|---|---|---|---|---|\n"
         + "\n".join("| " + " | ".join(r) + " |" for r in rows))
print(table)

_, pred_T = unseen_at(T)
kind_rows = {}
for r, pr in zip(unseen, pred_T):
    k = kind_rows.setdefault(r["kind"], [0, 0, set()])
    k[1] += 1
    if pr == OTH: k[0] += 1
    else: k[2].add(CLASSES[int(pr)])
kind_table = ("| unseen kind | rejected | mistaken for |\n|---|---|---|\n"
              + "\n".join(f"| {kd} | {a}/{b} | {', '.join(sorted(s)) or '–'} |" for kd, (a, b, s) in sorted(kind_rows.items())))
print(kind_table)
with open(OUT / "unseen_predictions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["file", "kind", "pred_no_threshold", "top_prob", f"pred_T{T}"])
    for r, p, pr in zip(unseen, p_uns, pred_T):
        w.writerow([r["name"], r["kind"], CLASSES[int(p.argmax())], round(float(p.max()), 4), CLASSES[int(pr)]])

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3.6))
plt.plot(cand, [s["val_false_rejection"] for s in sweep], label="val false rejection (course images)")
plt.plot(cand, [s["val_others_recall"] for s in sweep], label="val OTHERS recall")
plt.axvline(T, color="gray", ls="--", lw=1); plt.text(T, 0.5, f" T={T}", fontsize=8)
plt.xlabel("threshold T"); plt.ylabel("rate"); plt.legend(fontsize=7); plt.title("Threshold chosen on validation")
plt.tight_layout(); plt.savefig(OUT / "threshold_sweep_val.png", dpi=120)

(OUT / "threshold_results.md").write_text(
    f"# Confidence threshold on the 21-class model\n\nRule (fixed before test): {json.loads((OUT / 'threshold.json').read_text())['rule']}.\n\n"
    f"Chosen on validation: **T = {T}** (validation false rejection {chosen['val_false_rejection']:.4f}, validation OTHERS recall "
    f"{chosen['val_others_recall']:.3f}).\n\n## Test and unseen (T applied once)\n\n{table}\n\n## Unseen kinds at T = {T}\n\n{kind_table}\n",
    encoding="utf-8")
import shutil; shutil.rmtree(WORK, ignore_errors=True)
print("written:", OUT)
