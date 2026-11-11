"""Re-evaluation check (proposal §7.5): load registered model versions from the MLflow Model Registry, evaluate them again
with a fresh command, and compare with the metrics their training runs reported.

Detector versions (lab2real-detector):
    D1-test       Ultralytics val on the run's own D1-test split (runs/<run>/data/d1_dataset.zip), conf 0.001, NMS IoU 0.7,
                  imgsz 640 - the same call as detection/src/m2_detection.py, but on this machine's CPU
                  → mAP@0.5 and mAP@0.5:0.95 vs. test.d1.map50 / test.d1.map50_95
    real photos   the 29 hand-labelled phone photos, AP@0.5 of OBJ011 and OBJ002 (helpers from detection/src/eval_real_photos.py)
                  → mean vs. test.real.map50_2cls
Classifier versions (lab2real-classifier):
    M1 test split (303 images, found by the MD5 in the run's split_manifest.csv under detection/cluster/data/raw or the
    cached course_images_m1.zip), same preprocessing as M1 → accuracy and macro-F1 vs. test.accuracy / test.macro_f1

Tolerances (absolute): D1-test mAP 0.01 (training runs evaluated on a cluster GPU with Ultralytics 8.4.202; here CPU and
8.4.192); real photos 0.005 (the reported numbers were computed on this machine); classifier 0.004 (≈ 1 of 303 images).

Each check is logged as a run in the MLflow experiment `re-evaluation`, the version gets the tags reeval_passed /
reeval_run_id / reeval_date, and a summary table is written to mlops/logs/reevaluation.md.

Usage:
    python mlops/tools/reevaluate.py                                   every registered version of both models
    python mlops/tools/reevaluate.py --model lab2real-detector --alias champion
    python mlops/tools/reevaluate.py --model lab2real-detector --version 3
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlops_common import CLASSIFIER_MODEL, DETECTOR_MODEL, LOGS, REPO, RUNS, STORE, client, experiment_id  # noqa: E402
from mlflow.entities import Metric, Param, RunTag  # noqa: E402

EXP = "re-evaluation"
TOL = {"d1": 0.01, "real": 0.005, "cls": 0.004}
RAW_DIRS = [REPO / "detection/cluster/data/raw"]
RAW_ZIPS = [REPO / "detection/cluster/IE7615_project/cache/course_images_m1.zip"]


def download(c, mv):
    import mlflow
    tmp = Path(tempfile.mkdtemp(prefix=f"reeval-{mv.name}-v{mv.version}-"))
    return Path(mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(tmp))), tmp


# ---------- detector ----------

def d1_test_yaml(run_tag, classes):
    """Extract images/test + labels/test of the run's D1 dataset once; returns the data yaml."""
    out = STORE / "_reeval" / run_tag
    yml = out / "d1_test.yaml"
    if not yml.exists():
        with zipfile.ZipFile(RUNS / run_tag / "data/d1_dataset.zip") as z:
            for n in z.namelist():
                if n.startswith(("images/test/", "labels/test/")) and not n.endswith("/"):
                    z.extract(n, out)
        names = "\n".join(f"  {k}: {v}" for k, v in sorted(classes.items()))
        yml.write_text(f"path: {out.as_posix()}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n{names}\n",
                       encoding="utf-8")
    return yml


def real_photo_map50(weights):
    sys.path.insert(0, str(REPO / "detection/src"))
    import eval_real_photos as erp
    from ultralytics import YOLO
    model = YOLO(str(weights))
    gt = erp.load_gt()
    scored = {k: [] for k in erp.CLASSES}
    n_gt = {k: 0 for k in erp.CLASSES}
    for stem, boxes in gt.items():
        r = model.predict(str(erp.REAL / "images" / f"{stem}.jpg"), conf=0.001, imgsz=640, device="cpu", verbose=False)[0]
        h, w = r.orig_shape
        cls, conf, box = r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()
        for k in erp.CLASSES:
            g = [erp.xyxy(b, w, h) for b in boxes if b[0] == k]
            n_gt[k] += len(g)
            scored[k] += erp.match([(cf, bx) for kk, cf, bx in zip(cls, conf, box) if kk == k], g)
    aps = {erp.CLASSES[k]: erp.ap(scored[k], n_gt[k]) for k in erp.CLASSES}
    return float(np.mean(list(aps.values()))), aps


def check_detector(c, mv):
    from ultralytics import YOLO
    run = c.get_run(mv.run_id)
    tag = mv.tags.get("source_run_tag") or run.data.tags["run_tag"]
    local, tmp = download(c, mv)
    try:
        serving = json.loads((local / "serving.json").read_text(encoding="utf-8"))
        classes = {int(k): v for k, v in json.loads((local / "classes.json").read_text(encoding="utf-8")).items()}
        weights = local / serving["weights"]
        yml = d1_test_yaml(tag, classes)
        m = YOLO(str(weights)).val(data=str(yml), split="test", imgsz=serving["imgsz"], batch=16, conf=0.001,
                                   iou=serving["nms_iou"], device="cpu", plots=False, verbose=False,
                                   project=str(STORE / "_reeval" / "_val"), name=f"v{mv.version}", exist_ok=True)
        real, aps = real_photo_map50(weights)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    rm = run.data.metrics
    return [("test.d1.map50", float(m.box.map50), rm.get("test.d1.map50"), TOL["d1"]),
            ("test.d1.map50_95", float(m.box.map), rm.get("test.d1.map50_95"), TOL["d1"]),
            ("test.real.map50_2cls", real, rm.get("test.real.map50_2cls"), TOL["real"])], {
        "data": f"runs/{tag}/data/d1_dataset.zip (images/test, 300) + detection/data/real_photos (29)",
        "real_ap50": json.dumps({k: round(v, 4) for k, v in aps.items()})}


# ---------- classifier ----------

def image_index():
    """MD5 → bytes getter over the local raw course images and the cached zip."""
    idx = {}
    for d in RAW_DIRS:
        for p in d.rglob("*.jpg") if d.exists() else []:
            idx.setdefault(hashlib.md5(p.read_bytes()).hexdigest(), lambda p=p: p.read_bytes())
    for zp in RAW_ZIPS:
        if zp.exists():
            z = zipfile.ZipFile(zp)
            for n in z.namelist():
                if n.lower().endswith(".jpg"):
                    idx.setdefault(hashlib.md5(z.read(n)).hexdigest(), lambda n=n, z=z: z.read(n))
    return idx


def check_classifier(c, mv):
    import keras
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score
    run = c.get_run(mv.run_id)
    local, tmp = download(c, mv)
    try:
        model = keras.models.load_model(next(local.glob("*.keras")), compile=False)
        manifest = Path(c.download_artifacts(mv.run_id, "data/split_manifest.csv", str(tmp)))
        rows = [r for r in csv.DictReader(open(manifest, encoding="utf-8")) if r["split"] == "test"]
        idx = image_index()
        missing = [r["image_id"] for r in rows if r["md5"] not in idx]
        if missing:
            raise SystemExit(f"{len(missing)} test images not found locally by MD5, e.g. {missing[:3]}")
        xs = []
        for r in rows:          # same as M1: decode, resize with padding to 224, round, clip to 0-255
            img = tf.io.decode_image(idx[r["md5"]](), channels=3, expand_animations=False)
            img = tf.image.resize_with_pad(tf.cast(img, tf.float32), 224, 224)
            xs.append(tf.clip_by_value(tf.round(img), 0, 255))
        pred = model.predict(tf.stack(xs), batch_size=32, verbose=0).argmax(1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    y = np.array([int(r["class_idx"]) for r in rows])
    rm = run.data.metrics
    acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
    return [("test.accuracy", float(acc), rm.get("test.accuracy"), TOL["cls"]),
            ("test.macro_f1", float(f1), rm.get("test.macro_f1"), TOL["cls"])], {
        "data": f"M1 test split, {len(rows)} images (by MD5 from split_manifest.csv)", "correct": f"{int((pred == y).sum())}/{len(y)}"}


# ---------- driver ----------

def record(c, mv, checks, extra, seconds):
    passed = all(r is not None and abs(n - r) <= tol for _, n, r, tol in checks)
    now = int(time.time() * 1000)
    run = c.create_run(experiment_id(c, EXP), run_name=f"{mv.name} v{mv.version}",
                       tags={"model_name": mv.name, "model_version": str(mv.version), "source_run_id": mv.run_id,
                             "passed": str(passed), "mlflow.note.content": extra.get("data", "")})
    params = [Param("model_name", mv.name), Param("model_version", str(mv.version)), Param("source_mlflow_run", mv.run_id),
              Param("device", "cpu")] + [Param(f"tolerance.{k}", str(t)) for k, _, _, t in checks] + \
             [Param(k, str(v)[:6000]) for k, v in extra.items()]
    metrics = [Metric(f"reeval.{k}", n, now, 0) for k, n, _, _ in checks] + \
              [Metric(f"reported.{k}", r, now, 0) for k, _, r, _ in checks if r is not None] + \
              [Metric(f"diff.{k}", n - r, now, 0) for k, n, r, _ in checks if r is not None] + [Metric("seconds", seconds, now, 0)]
    c.log_batch(run.info.run_id, metrics=metrics, params=params)
    c.set_terminated(run.info.run_id, "FINISHED")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for k, v in (("reeval_passed", str(passed)), ("reeval_run_id", run.info.run_id), ("reeval_date", date)):
        c.set_model_version_tag(mv.name, mv.version, k, v)
    return passed, run.info.run_id


def write_log(results):
    lines = ["# Re-evaluation of registered model versions", "",
             f"Written by `mlops/tools/reevaluate.py` on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} (CPU). "
             "Each version was loaded from the MLflow Model Registry and evaluated again; reported = the metric its training "
             "run logged. Tolerances: D1-test mAP 0.01, real photos 0.005, classifier 0.004.", "",
             "| model | version | aliases | metric | re-evaluated | reported | difference | within tolerance |",
             "|---|---|---|---|---|---|---|---|"]
    for mv, aliases, checks, _, _ in results:
        for k, n, r, tol in checks:
            ok = r is not None and abs(n - r) <= tol
            lines.append(f"| {mv.name} | {mv.version} | {', '.join(aliases) or '-'} | {k} | {n:.4f} | "
                         f"{'-' if r is None else f'{r:.4f}'} | {'-' if r is None else f'{n - r:+.4f}'} | {'yes' if ok else 'NO'} |")
    lines += ["", "| model | version | MLflow re-evaluation run | details |", "|---|---|---|---|"]
    for mv, _, _, extra, rid in results:
        lines.append(f"| {mv.name} | {mv.version} | `{rid}` | {'; '.join(f'{k}: {v}' for k, v in extra.items())} |")
    out = LOGS / "reevaluation.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=[DETECTOR_MODEL, CLASSIFIER_MODEL])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--alias")
    g.add_argument("--version", type=int)
    a = ap.parse_args(argv)
    c = client()
    names = [a.model] if a.model else [DETECTOR_MODEL, CLASSIFIER_MODEL]
    results = []
    for name in names:
        aliases_of = {}
        for al, v in c.get_registered_model(name).aliases.items():
            aliases_of.setdefault(str(v), []).append(al)
        if a.alias:
            mvs = [c.get_model_version_by_alias(name, a.alias)]
        elif a.version:
            mvs = [c.get_model_version(name, str(a.version))]
        else:
            mvs = sorted(c.search_model_versions(f"name = '{name}'"), key=lambda m: int(m.version))
        for mv in mvs:
            t0 = time.perf_counter()
            checks, extra = (check_detector if name == DETECTOR_MODEL else check_classifier)(c, mv)
            passed, rid = record(c, mv, checks, extra, time.perf_counter() - t0)
            results.append((mv, sorted(aliases_of.get(str(mv.version), [])), checks, extra, rid))
            print(f"{name} v{mv.version}: {'PASS' if passed else 'FAIL'}  " +
                  ", ".join(f"{k} {n:.4f} (reported {r:.4f})" for k, n, r, _ in checks if r is not None))
    out = write_log(results)
    print(f"summary written to {out.relative_to(REPO).as_posix()}")
    if not all(all(r is not None and abs(n - r) <= tol for _, n, r, tol in ch) for _, _, ch, _, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
