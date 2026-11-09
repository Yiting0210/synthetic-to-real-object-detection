"""Evaluate the M2 comparison models (arms A, B and C if present) on the real multi-object photos (detection/data/real_photos).

Run:
    E:/CodeTools/anaconda3/envs/ie7615-demo/python.exe detection/src/eval_real_photos.py      (from the repository root)

Ground truth: detection/data/real_photos/labels/*.txt (YOLO format, drawn by hand with src/label_real_photos.py).
Only OBJ011 (mug, class 6) and OBJ002 (paper roll, class 2) are in the photos; every other
object is a distractor, so any detection of another class is a false alarm.

Per model:
  AP@0.5 for the two classes (all detections, conf >= 0.001) and their mean
  at the run's own threshold (chosen on validation, read from M2_results_for_report.md):
    recall per class, precision,
    false alarms = detections of the 18 classes that are not in the photos,
    other false positives = mug/roll detections that match no box (bad location or duplicate)
Writes detection/results/real_photos/README.md, and detection/results/real_photos/predictions/<run>/ (images with boxes,
seed 1 of each arm; not in git).
"""
import re
import statistics as st
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
REAL = HERE / "data" / "real_photos"
OUT = HERE / "results" / "real_photos"
RUNS = HERE.parent / "runs"                    # run folders downloaded from the cluster (not in git)
CLASSES = {6: "OBJ011", 2: "OBJ002"}
IOU = 0.5


def load_gt():
    gt = {}
    for lf in sorted((REAL / "labels").glob("*.txt")):
        rows = [l.split() for l in lf.read_text().split("\n") if l.strip()]
        gt[lf.stem] = [(int(r[0]), *map(float, r[1:])) for r in rows]
    return gt


def xyxy(b, w, h):
    _, x, y, bw, bh = b
    return np.array([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(preds, gts):
    """preds: list of (conf, box); gts: list of boxes. Greedy by confidence. Returns list of (conf, is_tp)."""
    used, out = set(), []
    for conf, box in sorted(preds, key=lambda p: -p[0]):
        best, bi = 0, -1
        for i, g in enumerate(gts):
            if i not in used:
                v = iou(box, g)
                if v > best:
                    best, bi = v, i
        if best >= IOU:
            used.add(bi)
            out.append((conf, True))
        else:
            out.append((conf, False))
    return out


def ap(scored, n_gt):
    """VOC-style all-point AP from (conf, is_tp) over all images."""
    scored = sorted(scored, key=lambda s: -s[0])
    tp = np.cumsum([s[1] for s in scored])
    fp = np.cumsum([not s[1] for s in scored])
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(run, gt, save_dir=None):
    from ultralytics import YOLO
    thr = float(re.search(r"General threshold ([0-9.]+)", (run / "M2_results_for_report.md").read_text()).group(1))
    model = YOLO(str(run / "models" / "d1_yolov8n_best.pt"))
    per_cls_scored = {c: [] for c in CLASSES}
    n_gt = {c: 0 for c in CLASSES}
    tp_thr = {c: 0 for c in CLASSES}
    false_alarm = other_fp = 0
    alarm_classes = {}
    for stem, boxes in gt.items():
        img = REAL / "images" / f"{stem}.jpg"
        r = model.predict(str(img), conf=0.001, imgsz=640, verbose=False)[0]
        h, w = r.orig_shape
        d_cls = r.boxes.cls.cpu().numpy().astype(int)
        d_conf = r.boxes.conf.cpu().numpy()
        d_box = r.boxes.xyxy.cpu().numpy()
        for c in CLASSES:
            g = [xyxy(b, w, h) for b in boxes if b[0] == c]
            n_gt[c] += len(g)
            preds = [(cf, bx) for k, cf, bx in zip(d_cls, d_conf, d_box) if k == c]
            per_cls_scored[c] += match(preds, g)
            m = match([p for p in preds if p[0] >= thr], g)
            tp_thr[c] += sum(t for _, t in m)
            other_fp += sum(not t for _, t in m)
        for k, cf in zip(d_cls, d_conf):
            if cf >= thr and k not in CLASSES:
                false_alarm += 1
                name = model.names[int(k)]
                alarm_classes[name] = alarm_classes.get(name, 0) + 1
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            keep = r.boxes.conf >= thr
            r.boxes = r.boxes[keep]
            r.save(filename=str(save_dir / f"{stem}.jpg"))
    aps = {c: ap(per_cls_scored[c], n_gt[c]) for c in CLASSES}
    n_det = sum(tp_thr.values()) + other_fp + false_alarm
    return {
        "thr": thr,
        "AP50 OBJ011": aps[6], "AP50 OBJ002": aps[2], "mAP50 (2 classes)": float(np.mean(list(aps.values()))),
        "recall OBJ011": tp_thr[6] / n_gt[6], "recall OBJ002": tp_thr[2] / n_gt[2],
        "precision": sum(tp_thr.values()) / max(n_det, 1),
        "false alarms (other classes)": false_alarm, "other FP (bad box / duplicate)": other_fp,
        "alarm_classes": alarm_classes,
    }


def main():
    gt = load_gt()
    runs = sorted(RUNS.glob("m2_20261108_*_arm*_seed*"))
    res = {}
    for run in runs:
        arm = re.search(r"arm([ABC])_seed(\d+)", run.name)
        key = (arm.group(1), int(arm.group(2)))
        save = OUT / "predictions" / run.name if key[1] == 1 else None
        res[key] = evaluate(run, gt, save)
        print(run.name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[key].items()}, flush=True)
    metrics = [k for k in next(iter(res.values())) if k not in ("thr", "alarm_classes")]
    n_img = len(gt)
    n_box = sum(len(v) for v in gt.values())
    arms = sorted({a for a, _ in res})
    label = {"A": "arm A (D1)", "B": "arm B (D1+D3)", "C": "arm C (D3)"}
    out = ["# Real multi-object photos: " + " vs ".join(label[a] for a in arms), "",
           f"{n_img} phone photos (one mug OBJ011 + one paper roll OBJ002 each, {n_box} boxes drawn by hand), "
           "with distractor objects (bottles, jars, boxes, coaster, pencil case) that are not in the 20 classes. "
           "Desk, carpet and chair backgrounds, partial occlusion. Mean ± sample std over seeds 42 / 1 / 2; "
           "thresholds are each run's own, chosen on validation.", "",
           "| metric | " + " | ".join(label[a] for a in arms) + " |", "|---|" + "---|" * len(arms)]
    for m in metrics:
        p = 1 if "alarm" in m or "FP" in m else 3
        cells = []
        for arm in arms:
            v = [res[(arm, s)][m] for s in (42, 1, 2) if (arm, s) in res]
            cells.append(f"{st.mean(v):.{p}f} ± {st.stdev(v):.{p}f}" if len(v) > 1 else f"{v[0]:.{p}f}")
        out.append(f"| {m} | " + " | ".join(cells) + " |")
    out += ["", "## Per run", "", "| run | thr | " + " | ".join(metrics) + " | false-alarm classes |",
            "|---|---|" + "---|" * len(metrics) + "---|"]
    for (arm, s), r in sorted(res.items()):
        vals = " | ".join(f"{r[m]:.3f}" if isinstance(r[m], float) else str(r[m]) for m in metrics)
        out.append(f"| {arm} seed {s} | {r['thr']} | {vals} | {r['alarm_classes']} |")
    out += ["", "Caveat: only two of the 20 classes can be tested (the other objects belong to classmates); "
            "the paper roll is a different roll from the one in training (same category, different instance)."]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", OUT / "README.md")


if __name__ == "__main__":
    main()
