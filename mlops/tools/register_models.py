"""Register the models chosen for use as numbered versions in the MLflow Model Registry (proposal §7.7).

    lab2real-detector     the seed-42 detector of each ablation arm: v1 = arm A (D1), v2 = arm B (D1+D3), v3 = arm C (D3)
    lab2real-classifier   the M1 classifier selected on validation (MobileNetV2, Keras): v1

Each version points to the `model/` artifact folder of its imported MLflow run and carries tags for the class map,
preprocessing, thresholds, data manifest and runtime. Run mlops/tools/import_runs.py first. Re-running does not create
duplicate versions.

Detector versions also get a readiness fixture, added to the run's `model/readiness/` folder (so every bundle exported from
the version carries it): the first D1-val image (development set, from the run's own d1_dataset.zip) on which the model
finds every labelled object, plus expected.json with the detections this version gives on it (CPU, at its threshold).
The service compares against it at startup (service/model_loader.py, readiness_check).

Aliases: `champion` = version intended for use; `previous` = last good version. On first registration champion → arm B
(best arm of the A/B/C ablation, detection/results/ablation) and previous → arm A. mlops/tools/rollback_demo.py moves them.
Arm C is registered for comparison only and gets no alias. The classifier gets champion → v1.

Usage:  python mlops/tools/register_models.py
"""
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlops_common import (CLASSIFIER_MODEL, DETECTOR_MODEL, MODEL_SUBDIR, READINESS_SUBDIR, REPO, RUNS, STORE,  # noqa: E402
                          client, experiment_id)

TO_REGISTER = [("A", 42), ("B", 42), ("C", 42)]       # registration order = version numbers 1, 2, 3
ARM_DATA = {"A": "D1", "B": "D1+D3", "C": "D3"}
READINESS_TRIES = 40


def find_version(c, name, run_id):
    for mv in c.search_model_versions(f"name = '{name}'"):
        if mv.run_id == run_id:
            return mv
    return None


def ensure_registered_model(c, name, description):
    try:
        c.get_registered_model(name)
    except Exception:
        c.create_registered_model(name, description=description)


# ---------- readiness fixture ----------

def yolo_boxes(label_text, w, h):
    out = []
    for line in label_text.splitlines():
        if line.strip():
            k, cx, cy, bw, bh = map(float, line.split())
            out.append((int(k), [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def ensure_readiness(c, run):
    """Add model/readiness/ (image + expected.json) to a detector run if it is not there yet."""
    rid, tag = run.info.run_id, run.data.tags["run_tag"]
    if any(f.path.endswith("expected.json") for f in c.list_artifacts(rid, f"{MODEL_SUBDIR}/{READINESS_SUBDIR}")):
        return "present"
    zpath = RUNS / tag / "data/d1_dataset.zip"
    if not zpath.exists():
        raise SystemExit(f"{zpath} not found: the readiness fixture is cut from the run's D1-val images")
    sys.path.insert(0, str(REPO))
    from PIL import Image
    from service.model_loader import Detector, _box_iou, read_classes
    tmp = STORE / "_tmp_readiness" / tag
    shutil.rmtree(tmp, ignore_errors=True)
    model_dir = Path(c.download_artifacts(rid, MODEL_SUBDIR, str(tmp)))
    serving = json.loads((model_dir / "serving.json").read_text(encoding="utf-8"))
    det = Detector(model_dir / serving["weights"], read_classes(model_dir / "classes.json"), serving["conf_threshold"],
                   serving["nms_iou"], serving["imgsz"])
    with zipfile.ZipFile(zpath) as z:
        names = sorted(n for n in z.namelist() if n.startswith("images/val/") and n.endswith(".jpg"))
        chosen = None
        for n in names[:READINESS_TRIES]:
            data = z.read(n)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            gt = yolo_boxes(z.read(n.replace("images/", "labels/").replace(".jpg", ".txt")).decode(), *image.size)
            dets, _ = det(image)
            found = sum(any(d["class_id"] == k and _box_iou(d["box_xyxy"], b) >= 0.5 for d in dets) for k, b in gt)
            if chosen is None or found / len(gt) > chosen[3] / len(chosen[2]):
                chosen = (n, data, gt, found, dets)
            if found == len(gt):
                break
    n, data, gt, found, dets = chosen
    out = tmp / READINESS_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    name = Path(n).name
    (out / name).write_bytes(data)
    import torch
    import ultralytics
    expected = {"image": name, "source": f"D1-val (development set) of {tag}", "conf": serving["conf_threshold"],
                "ground_truth": [{"class_id": k, "box_xyxy": [round(v, 1) for v in b]} for k, b in gt],
                "ground_truth_found_at_iou_0.5": f"{found}/{len(gt)}", "detections": dets,
                "recorded_with": {"device": "cpu", "torch": torch.__version__, "ultralytics": ultralytics.__version__},
                "rule": "same classes, IoU >= 0.9, confidence within 0.05; detections within 0.1 of the threshold are optional"}
    (out / "expected.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    c.log_artifacts(rid, str(out), f"{MODEL_SUBDIR}/{READINESS_SUBDIR}")
    shutil.rmtree(tmp, ignore_errors=True)
    return f"added {name} ({found}/{len(gt)} labelled objects found, {len(dets)} detections)"


# ---------- detector ----------

def register_detector(c, arm, seed):
    exp = experiment_id(c, "M2-detection")
    runs = c.search_runs([exp], filter_string=f"tags.arm = '{arm}' and tags.seed = '{seed}'")
    if len(runs) != 1:
        raise SystemExit(f"expected one imported run for arm {arm} seed {seed}, found {len(runs)}; run import_runs.py")
    run = runs[0]
    print(f"  arm {arm} seed {seed}: readiness fixture {ensure_readiness(c, run)}")
    mv = find_version(c, DETECTOR_MODEL, run.info.run_id)
    if mv is not None:
        print(f"  arm {arm} seed {seed}: already version {mv.version}")
        return mv
    serving = json.loads(Path(c.download_artifacts(run.info.run_id, f"{MODEL_SUBDIR}/serving.json")).read_text(encoding="utf-8"))
    source = f"{run.info.artifact_uri}/{MODEL_SUBDIR}"
    t = run.data.tags
    p = run.data.params
    tags = {
        "arm": arm, "seed": str(seed), "source_run_tag": t.get("run_tag"), "source_run_id": t.get("source_run_id"),
        "classes_json_sha256": serving["classes_json_sha256"], "weights_sha256": serving["weights_sha256"],
        "preprocessing": f"RGB, letterbox to {serving['imgsz']}x{serving['imgsz']} (Ultralytics), boxes mapped back to input pixels",
        "conf_threshold": str(serving["conf_threshold"]), "nms_iou": str(serving["nms_iou"]),
        "d1_train_manifest_sha256": p.get("checksum.file.d1_train_manifest.csv", ""),
        "masks_manifest_sha256": p.get("checksum.file.masks_manifest.csv", ""),
        "runtime": serving["runtime"], "git_commit": t.get("git_commit"),
    }
    m = run.data.metrics
    desc = (f"YOLOv8n, arm {arm} ({ARM_DATA[arm]}), seed {seed}, threshold {serving['conf_threshold']}. "
            f"D1-test mAP50-95 {m.get('test.d1.map50_95', float('nan')):.3f}, T-hard {m.get('test.thard.map50_95', float('nan')):.3f}, "
            f"real photos mAP50 (2 classes) {m.get('test.real.map50_2cls', float('nan')):.3f}. "
            "Inputs: RGB photos of the 20 course objects. Limitations: no OTHERS class; real-photo check covers 2 classes only.")
    mv = c.create_model_version(DETECTOR_MODEL, source=source, run_id=run.info.run_id, tags=tags, description=desc)
    print(f"  arm {arm} seed {seed}: registered as version {mv.version} (run {run.info.run_id})")
    return mv


# ---------- classifier ----------

def register_classifier(c):
    exp = experiment_id(c, "M1-classification")
    runs = [r for r in c.search_runs([exp], filter_string="tags.selected_as_best = 'True'")]
    if len(runs) != 1:
        raise SystemExit(f"expected one selected M1 run, found {len(runs)}; run import_runs.py")
    run = runs[0]
    if not str(run.data.tags.get("model_artifact", "")).startswith(MODEL_SUBDIR + "/"):
        raise SystemExit("the selected M1 run has no model artifact (models/mobilenetv2.keras missing at import time)")
    mv = find_version(c, CLASSIFIER_MODEL, run.info.run_id)
    if mv is not None:
        print(f"  classifier {run.data.params.get('model')}: already version {mv.version}")
        return mv
    rec = json.loads(Path(c.download_artifacts(run.info.run_id, f"records/run_{run.data.params['model']}.json")).read_text(encoding="utf-8"))
    p, m = run.data.params, run.data.metrics
    tags = {
        "model": p["model"], "seed": p.get("seed", ""), "source_run_tag": run.data.tags.get("run_tag"),
        "source_run_id": run.data.tags.get("source_run_id"), "weights_sha256": rec["artifact"]["sha256"],
        "classes_json_sha256": p.get("data.classes_json_sha256", ""), "split_manifest_sha256": p.get("data.split_manifest_sha256", ""),
        "preprocessing": "RGB 0-255, resize with padding to 224x224; model-specific scaling is the model's first layer",
        "selection_rule": p.get("selection_rule", "")[:500], "runtime": "keras / tensorflow", "git_commit": run.data.tags.get("git_commit"),
    }
    desc = (f"M1 single-object classifier ({p['model']}, 20 course classes), selected on validation. "
            f"Validation accuracy {m.get('val.accuracy', float('nan')):.4f}, test accuracy {m.get('test.accuracy', float('nan')):.4f} "
            f"(303 test images). Not served by the API yet (no /classify endpoint).")
    mv = c.create_model_version(CLASSIFIER_MODEL, source=f"{run.info.artifact_uri}/{MODEL_SUBDIR}", run_id=run.info.run_id,
                                tags=tags, description=desc)
    print(f"  classifier {p['model']}: registered as version {mv.version} (run {run.info.run_id})")
    return mv


def main():
    c = client()
    ensure_registered_model(c, DETECTOR_MODEL, "Lab-to-Real multi-object detector (YOLOv8n, 20 course objects).")
    versions = {arm: register_detector(c, arm, s) for arm, s in TO_REGISTER}
    if "champion" not in c.get_registered_model(DETECTOR_MODEL).aliases:
        c.set_registered_model_alias(DETECTOR_MODEL, "champion", versions["B"].version)
        c.set_registered_model_alias(DETECTOR_MODEL, "previous", versions["A"].version)
        print(f"  champion → version {versions['B'].version} (arm B), previous → version {versions['A'].version} (arm A)")
    print("detector aliases:", dict(c.get_registered_model(DETECTOR_MODEL).aliases))
    ensure_registered_model(c, CLASSIFIER_MODEL, "Lab-to-Real single-object classifier (M1, Keras, 20 course objects).")
    cmv = register_classifier(c)
    if "champion" not in c.get_registered_model(CLASSIFIER_MODEL).aliases:
        c.set_registered_model_alias(CLASSIFIER_MODEL, "champion", cmv.version)
    print("classifier aliases:", dict(c.get_registered_model(CLASSIFIER_MODEL).aliases))
    shutil.rmtree(STORE / "_tmp_readiness", ignore_errors=True)


if __name__ == "__main__":
    main()
