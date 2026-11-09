"""Compare the five rejection methods of openset_methods.py on the same data with the same threshold rule.

    python classification/src/openset_eval.py        (repository root, laptop CPU, a few minutes, no training)

1. Inputs: the M1 split for the 20 course classes (train for the prototypes, val for the thresholds, test for testing),
   the OTHERS photos (same import, square crop, pHash groups and seeded split as m1_others.ipynb) and the 36 unseen-kind photos.
2. Models: V1 = models/mobilenetv2.keras (20 classes), V2 = models/mobilenetv2_others.keras (21 classes).
   For every image: softmax probabilities and the 1280-d feature before the classification layer.
3. Check: the plain predictions must reproduce the earlier runs (V1 test 301/303; V2 test 302/303, unseen 15/36), else stop.
4. Threshold, per method, chosen on VALIDATION ONLY: the largest value rejecting at most 1% of the 303 val course images.
5. Applied once to the 303 test course images, the 15 OTHERS test photos and the 36 unseen-kind photos.
   AUROC (test course images vs unseen photos) measures each method without any threshold.
Writes classification/results/openset/<run>/.
"""
import csv, hashlib, json, math, re, subprocess, sys, shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openset_methods import (ModelOutputs, METHODS, class_prototypes, rejected, choose_threshold,
                             ranking_score, auroc, auroc_ci)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "detection" / "cluster" / "data" / "raw"
M1_DATA = ROOT / "classification" / "results" / "m1" / "data"
OTHERS_ROOT = ROOT / "classification" / "others"
MODELS = ROOT / "models"
RUN = datetime.now().strftime("openset_%Y%m%d_%H%M%S")
OUT = ROOT / "classification" / "results" / "openset" / RUN
WORK = OUT / "_work"
OUT.mkdir(parents=True, exist_ok=True)
SEED, IMG_SIZE, PHASH_THRESHOLD, SPLIT_RATIOS = 42, 224, 4, (0.70, 0.15, 0.15)
MAX_FALSE_REJECTION = 0.01
DEPLOYED_T = 0.50                                  # threshold of the current demo (method D), for reference
EXPECTED = dict(v1_test=(301, 303), v2_test=(302, 303), v2_unseen=(15, 36))
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}

import tensorflow as tf
import keras

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
classes20 = {int(k): v for k, v in json.loads((M1_DATA / "classes.json").read_text(encoding="utf-8")).items()}
OTH = len(classes20)

# ---------------------------------------------------------------- 1. inputs (exactly as in m1_others.ipynb)
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
    rows = []
    for i, f in enumerate(files):
        dst = WORK / prefix / f"{prefix}_{i:04d}.jpg"
        load_square(f).save(dst, quality=95)
        rows.append(dict(name=f.name, kind=kind_of(f.name), path=str(dst)))
    return rows

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
order = np.random.default_rng([SEED, OTH]).permutation(len(groups))
glist = sorted((list(groups.values())[k] for k in order), key=len, reverse=True)
n = len(others); targets = [round(n * SPLIT_RATIOS[0]), round(n * SPLIT_RATIOS[1])]; targets.append(n - sum(targets))
counts = [0, 0, 0]
for m in glist:
    k = int(np.argmax([(targets[j] - counts[j]) / max(targets[j], 1) for j in range(3)]))
    counts[k] += len(m)
    for i in m:
        others[i]["split"] = ["train", "val", "test"][k]

folders = {p.name.strip(): p for p in RAW.iterdir() if p.is_dir()}
course = {s: [] for s in ("train", "val", "test")}
for r in csv.DictReader(open(M1_DATA / "split_manifest.csv", newline="", encoding="utf-8")):
    folder, name = r["src_path"].split("/", 1)
    course[r["split"]].append(dict(name=r["image_id"], label=int(r["class_idx"]), path=str(folders[folder.strip()] / name)))

SETS = {"train": course["train"], "val": course["val"], "test": course["test"],
        "others_test": [r for r in others if r["split"] == "test"], "unseen": unseen}

# ---------------------------------------------------------------- 2. model outputs (probabilities + features)
def outputs(model_file):
    m = keras.models.load_model(MODELS / model_file, compile=False)
    two = keras.Model(m.inputs, [m.layers[-1].input, m.output])          # input of the last Dense = pooled feature
    def dec(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize_with_pad(tf.cast(img, tf.float32), IMG_SIZE, IMG_SIZE)
        return tf.clip_by_value(tf.round(img), 0, 255)
    res = {}
    for name, rows in SETS.items():
        ds = tf.data.Dataset.from_tensor_slices([r["path"] for r in rows]).map(dec).batch(32)
        f, p = two.predict(ds, verbose=0)
        res[name] = ModelOutputs(probs=p, feats=f)
    return res

V1, V2 = outputs("mobilenetv2.keras"), outputs("mobilenetv2_others.keras")
y = {s: np.array([r["label"] for r in SETS[s]]) for s in ("train", "val", "test")}

# ---------------------------------------------------------------- 3. reproduce the earlier runs
got = dict(v1_test=(int((V1["test"].probs.argmax(1) == y["test"]).sum()), len(y["test"])),
           v2_test=(int((V2["test"].probs.argmax(1) == y["test"]).sum()), len(y["test"])),
           v2_unseen=(int((V2["unseen"].probs.argmax(1) == OTH).sum()), len(unseen)))
print("reproduces the earlier runs:", got == EXPECTED, got)
if got != EXPECTED:
    raise SystemExit("Inputs or models differ from the earlier runs; the comparison would not be about the same models. Stop.")

# prototypes from the course TRAIN images only
PROTO = dict(prototypes_v1=class_prototypes(V1["train"].feats, y["train"], OTH),
             prototypes_v2=class_prototypes(V2["train"].feats, y["train"], OTH))

def run(method, s):
    spec = METHODS[method]
    out = (V1 if spec["model"] == "V1" else V2)[s]
    return spec["fn"](**{spec["model"].lower(): out}, others_idx=OTH, **PROTO)

# ---------------------------------------------------------------- 4–5. thresholds on val, then test
def wilson(k, n, z=1.96):
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)
def fmt(k, n):
    lo, hi = wilson(k, n); return f"{k}/{n} = {k / n:.3f} ({lo:.3f}–{hi:.3f})"

results, thresholds, per_image = {}, {}, []
rows_to_run = list(METHODS) + ["D*"]                       # D* = method D at the deployed threshold 0.50
for key in rows_to_run:
    m = "D" if key == "D*" else key
    spec = METHODS[m]
    pv, sv, fv = run(m, "val")
    if key == "D*":
        t = DEPLOYED_T
    else:
        t = choose_threshold(sv, fv, MAX_FALSE_REJECTION) if spec["uses_threshold"] else -np.inf
    thresholds[key] = t
    r = {}
    for s in ("val", "test", "others_test", "unseen"):
        pred, score, forced = run(m, s)
        rej = rejected(score, forced, t)
        r[s] = dict(pred=pred, score=score, forced=forced, rej=rej)
    kt = r["test"]
    correct = int(((~kt["rej"]) & (kt["pred"] == y["test"])).sum())
    s_known = ranking_score(kt["score"], kt["forced"]); s_uns = ranking_score(r["unseen"]["score"], r["unseen"]["forced"])
    finite = np.isfinite(np.concatenate([s_known, s_uns]))
    lo_fill = np.nanmin(np.concatenate([s_known, s_uns])[finite]) - 1 if finite.any() else -1
    s_known, s_uns = np.where(np.isfinite(s_known), s_known, lo_fill), np.where(np.isfinite(s_uns), s_uns, lo_fill)
    results[key] = dict(
        label=spec["label"] + (" (deployed T = 0.50)" if key == "D*" else ""), model=spec["model"],
        threshold=None if not np.isfinite(t) else round(float(t), 4),
        val_false_rejection=int(r["val"]["rej"].sum()),
        test_known_correct=correct, test_false_rejection=int(kt["rej"].sum()),
        others_test_rejected=int(r["others_test"]["rej"].sum()), unseen_rejected=int(r["unseen"]["rej"].sum()),
        auroc=round(auroc(s_known, s_uns), 4), auroc_ci=[round(v, 4) for v in auroc_ci(s_known, s_uns)],
        curve_known=s_known, curve_unseen=s_uns)
    for i, row in enumerate(unseen):
        per_image.append(dict(method=key, set="unseen", file=row["name"], kind=row["kind"],
                              pred=("OTHERS" if r["unseen"]["rej"][i] else classes20[int(r["unseen"]["pred"][i])]),
                              score=round(float(r["unseen"]["score"][i]), 4)))

nk, no, nu = len(y["test"]), len(SETS["others_test"]), len(unseen)
header = "| | method | threshold (val) | test known accuracy | test false rejection | OTHERS test rejected | unseen rejected | AUROC (test vs unseen) |"
lines = [header, "|---|---|---|---|---|---|---|---|"]
for key, r in results.items():
    lines.append(f"| {key} | {r['label']} | {r['threshold'] if r['threshold'] is not None else '–'} | {fmt(r['test_known_correct'], nk)} | "
                 f"{fmt(r['test_false_rejection'], nk)} | {fmt(r['others_test_rejected'], no)} | {fmt(r['unseen_rejected'], nu)} | "
                 f"{r['auroc']:.3f} ({r['auroc_ci'][0]:.3f}–{r['auroc_ci'][1]:.3f}) |")
table = "\n".join(lines)
print(table)

# per-kind table for the unseen photos
kinds = sorted({r["kind"] for r in unseen})
kt_lines = ["| unseen kind | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
for kd in kinds:
    cells = []
    for key in results:
        ps = [p for p in per_image if p["method"] == key and p["kind"] == kd]
        cells.append(f"{sum(p['pred'] == 'OTHERS' for p in ps)}/{len(ps)}")
    kt_lines.append(f"| {kd} | " + " | ".join(cells) + " |")
kind_table = "\n".join(kt_lines)

# ---------------------------------------------------------------- outputs
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.figure(figsize=(6.5, 4.5))
for key, r in results.items():
    if key in ("C", "D*"):
        continue
    sk, su = r["curve_known"], r["curve_unseen"]
    ts = np.unique(np.concatenate([sk, su, [np.inf]]))
    fr = [(sk < t).mean() for t in ts]; rr = [(su < t).mean() for t in ts]
    plt.plot(fr, rr, label=f"{key}: {r['label']} (AUROC {r['auroc']:.3f})")
for key in ("C", "D*"):
    r = results[key]
    plt.scatter([r["test_false_rejection"] / nk], [r["unseen_rejected"] / nu], zorder=3, label=f"{key}: {r['label']}")
plt.xlim(0, 0.3); plt.xlabel("test course images rejected (false rejection)"); plt.ylabel("unseen photos rejected")
plt.title("Rejection trade-off (test; thresholds for the table chosen on val)"); plt.legend(fontsize=6.5); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "curves.png", dpi=130)

with open(OUT / "predictions_unseen.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["method", "set", "file", "kind", "pred", "score"]); w.writeheader(); w.writerows(per_image)
(OUT / "thresholds.json").write_text(json.dumps(
    {k: (None if not np.isfinite(v) else float(v)) for k, v in thresholds.items()}, indent=2), encoding="utf-8")
try:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    commit = "not recorded"
record = dict(run=RUN, time=datetime.now(timezone.utc).isoformat(), git_commit=commit,
              rule=f"threshold per method = largest value rejecting at most {MAX_FALSE_REJECTION:.0%} of the val course images",
              models={f: sha(MODELS / f)[:12] for f in ["mobilenetv2.keras", "mobilenetv2_others.keras"]},
              data=dict(split_manifest=sha(M1_DATA / "split_manifest.csv")[:8], n=dict(train=len(y["train"]), val=len(y["val"]),
                        test=nk, others_test=no, unseen=nu)),
              results={k: {kk: vv for kk, vv in v.items() if not kk.startswith("curve")} for k, v in results.items()})
(OUT / "run_record.json").write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
(OUT / "comparison.md").write_text(
    f"# Rejection methods compared ({RUN})\n\n{record['rule']}; applied once to test. Rates with 95% Wilson intervals; AUROC with a bootstrap 95% interval.\n"
    f"OTHERS test photos are kinds V2 saw in training (unknown only for V1).\n\n{table}\n\n## Unseen photos rejected, per kind\n\n{kind_table}\n",
    encoding="utf-8")
shutil.rmtree(WORK, ignore_errors=True)
print("written:", OUT)
