"""Import existing run records (Colab M1, HPC M2, open-set study) into the local MLflow store. No retraining.

Sources (all already in the repository / run folders):
    M1 classification   classification/results_m1/records/run_*.json  (+ HPC copy of its data in runs/m1_*)
    open-set rejection  classification/results_openset/openset_*/run_record.json
    M2 detection        detection/cluster/IE7615_project/runs/m2_20261108_*_arm{A,B}_seed{42,1,2}/
                        (records/run_yolov8n_d1.json, figures/train_args.yaml, figures/train_results.csv,
                         models/d1_yolov8n_best.pt, M2_results_for_report.md)
    real photos         detection/real_multi/real_multi_results.md (per-run table)

Each source run becomes one MLflow run. MLflow's own run ID is never overwritten: the original ID is the tag
`source_run_id`, and mlops/run_id_map.csv maps source run ↔ MLflow run. Re-running skips runs already imported
(use --force to delete and re-import them).

Metric names: val.* = validation (used for selection), test.* = final test sets (D1-test, T-hard, real photos),
train/* = per-epoch Ultralytics curves (step = epoch).

Usage:
    python tools/import_runs.py [--force]
    mlflow ui --backend-store-uri sqlite:///mlops/mlflow.db --port 5000
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlops_common import MODEL_SUBDIR, REPO, STORE, client, experiment_id, sha256  # noqa: E402

from mlflow.entities import Metric, Param, RunTag  # noqa: E402

RUNS = REPO / "detection/cluster/IE7615_project/runs"
M1_DIR = REPO / "classification/results_m1"
OPENSET_DIR = REPO / "classification/results_openset"
REAL_MD = REPO / "detection/real_multi/real_multi_results.md"
M2_PATTERN = "m2_20261108_*_arm[AB]_seed*"

EXP_M1, EXP_OPENSET, EXP_M2 = "M1-classification", "openset-rejection", "M2-detection"
MAX_PARAM = 6000
BAD_KEY = re.compile(r"[^0-9A-Za-z_\-. /]")


# ---------- small helpers ----------

def key(s):
    """MLflow-safe key: (B) suffixes dropped, other unsupported characters → '_'."""
    return BAD_KEY.sub("_", str(s).replace("(B)", "")).strip("_")


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        k = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(flatten(v, k + "."))
        else:
            out[k] = v
    return out


def ms(iso):
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


def rel(p):
    return Path(p).resolve().relative_to(REPO).as_posix()


def git_head():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


class RunWriter:
    """Collects params / metrics / tags and writes them with log_batch (MLflow caps a batch at 1000 metrics, 100 params)."""

    def __init__(self, c, run_id, ts):
        self.c, self.run_id, self.ts = c, run_id, ts
        self.params, self.metrics, self.tags = {}, [], {}

    def param(self, k, v):
        v = "None" if v is None else (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v))
        self.params[key(k)] = v[:MAX_PARAM]

    def params_from(self, d, prefix=""):
        for k, v in flatten(d, prefix).items():
            self.param(k, v)

    def metric(self, k, v, step=0):
        if v is None or isinstance(v, bool):
            return
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        self.metrics.append(Metric(key(k), v, self.ts, step))

    def tag(self, k, v):
        self.tags[key(k)] = "not recorded" if v is None else str(v)

    def flush(self):
        p = [Param(k, v) for k, v in self.params.items()]
        t = [RunTag(k, v) for k, v in self.tags.items()]
        for i in range(0, len(p), 100):
            self.c.log_batch(self.run_id, params=p[i:i + 100])
        for i in range(0, len(self.metrics), 1000):
            self.c.log_batch(self.run_id, metrics=self.metrics[i:i + 1000])
        self.c.log_batch(self.run_id, tags=t)


def existing_run(c, exp_id, source_run_id):
    found = c.search_runs([exp_id], filter_string=f"tags.source_run_id = '{source_run_id}'", max_results=5)
    return found[0] if found else None


def new_run(c, exp_name, source_run_id, run_name, start_iso, force, parent=None):
    exp_id = experiment_id(c, exp_name)
    old = existing_run(c, exp_id, source_run_id)
    if old is not None:
        if not force:
            print(f"  skip (already imported): {run_name} → {old.info.run_id}")
            return None, old.info.run_id
        c.delete_run(old.info.run_id)
    tags = {"source_run_id": source_run_id, "imported_by": "tools/import_runs.py", "import_repo_commit": git_head()}
    if parent:
        tags["mlflow.parentRunId"] = parent
    run = c.create_run(exp_id, start_time=ms(start_iso), tags=tags, run_name=run_name)
    return run, run.info.run_id


def log_files(c, run_id, files, artifact_path=None):
    for f in files:
        if Path(f).exists():
            c.log_artifact(run_id, str(f), artifact_path)


# ---------- M1 classification ----------

def import_m1(c, force, mapping):
    sel = json.loads((M1_DIR / "records/selection.json").read_text(encoding="utf-8"))
    test_log = [json.loads(l) for l in (M1_DIR / "records/test_run_log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    hpc_m1 = sorted(RUNS.glob("m1_*"))
    hpc_check = {}
    for d in hpc_m1:         # the M1 split copied to the cluster for M2 must be the same split
        for f in ("classes.json", "split_manifest.csv"):
            if (d / "data" / f).exists():
                hpc_check[f"{d.name}/{f}"] = sha256(d / "data" / f) == sha256(M1_DIR / "data" / f)
    for rec_path in sorted((M1_DIR / "records").glob("run_*.json")):
        r = json.loads(rec_path.read_text(encoding="utf-8"))
        src = f"{r['run_id']}:{r['model']}"
        run, rid = new_run(c, EXP_M1, src, f"m1_{r['model']}", sel["decided_at"], force)
        mapping.append((EXP_M1, src, r["run_tag"], rel(rec_path), rid))
        if run is None:
            continue
        w = RunWriter(c, rid, ms(sel["decided_at"]))
        w.params_from(r["config"], "config.")
        w.param("seed", r["config"]["seed"])
        w.param("model", r["model"])
        w.param("selection_rule", r["selection_rule"])
        w.param("chosen_stage", r["chosen_stage"])
        w.param("params_total", r["params_total"])
        w.param("params_trainable_final", r.get("params_trainable_final"))
        w.params_from(r["data"], "data.")
        w.params_from({"epochs_run": r["epochs_run"]})
        w.metric("val.accuracy", r["validation_metrics"]["accuracy"])
        w.metric("val.macro_f1", r["validation_metrics"]["macro_f1"])
        tm = r["final_test_metrics"]
        w.metric("test.accuracy", tm["accuracy"])
        w.metric("test.macro_f1", tm["macro_f1"])
        for cls, f1 in tm.get("per_class_f1", {}).items():
            w.metric(f"test.f1.{cls}", f1)
        w.metric("train_minutes", r["train_minutes"])
        w.tag("task", r["task"])
        w.tag("run_tag", r["run_tag"])
        w.tag("git_commit", r["git_commit"])
        w.tag("git_dirty", r["git_dirty"])
        w.tag("selected_as_best", r.get("selected_as_best", False))
        w.tag("test_seen_before_decision", sel.get("test_seen_before_decision"))
        w.tag("test_evaluations_logged", len(test_log))
        w.tag("imported_from", rel(rec_path))
        for k, v in flatten(r["environment"]).items():
            w.tag(f"env.{k}", v)
        for k, v in hpc_check.items():
            w.tag(f"hpc_copy_matches.{k}", v)
        w.flush()
        log_files(c, rid, [rec_path, M1_DIR / "records/selection.json", M1_DIR / "records/results.csv",
                           M1_DIR / "records/test_run_log.jsonl", M1_DIR / "M1_results_for_report.md"], "records")
        log_files(c, rid, [M1_DIR / "data/classes.json", M1_DIR / "data/split_manifest.csv",
                           M1_DIR / "data/data_report.md"], "data")
        weights = REPO / r["artifact"]["path"]   # only the selected classifier is kept locally (models/)
        if weights.exists() and sha256(weights) == r["artifact"]["sha256"]:
            log_files(c, rid, [weights, REPO / "models/classes.json"], MODEL_SUBDIR)
            c.set_tag(rid, "model_artifact", f"{MODEL_SUBDIR}/{weights.name}")
        else:
            c.set_tag(rid, "model_artifact", "not available locally (weights kept on Drive)")
        c.set_terminated(rid, "FINISHED", end_time=ms(sel["decided_at"]))
        print(f"  M1 {r['model']:15s} → {rid}")


# ---------- open-set rejection study ----------

def import_openset(c, force, mapping):
    for d in sorted(OPENSET_DIR.glob("openset_*")):
        r = json.loads((d / "run_record.json").read_text(encoding="utf-8"))
        run, rid = new_run(c, EXP_OPENSET, r["run"], r["run"], r["time"], force)
        mapping.append((EXP_OPENSET, r["run"], r["run"], rel(d / "run_record.json"), rid))
        if run is not None:
            w = RunWriter(c, rid, ms(r["time"]))
            w.param("rule", r["rule"])
            w.params_from(r["models"], "model_sha256_prefix.")
            w.params_from(r["data"], "data.")
            w.tag("git_commit", r["git_commit"])
            w.tag("task", "M1 extension: rejecting unknown objects (open set)")
            w.tag("imported_from", rel(d / "run_record.json"))
            w.flush()
            log_files(c, rid, [d / f for f in ("run_record.json", "thresholds.json", "comparison.md",
                                               "predictions_unseen.csv", "curves.png")])
            c.set_terminated(rid, "FINISHED", end_time=ms(r["time"]))
            print(f"  open-set {r['run']} → {rid}")
        for m, res in r["results"].items():      # one child run per rejection method, so the UI can compare them
            name = "D_deployed" if m == "D*" else m
            src = f"{r['run']}:{name}"
            child, cid = new_run(c, EXP_OPENSET, src, f"method {name}: {res['label']}", r["time"], force, parent=rid)
            mapping.append((EXP_OPENSET, src, r["run"], rel(d / "run_record.json"), cid))
            if child is None:
                continue
            cw = RunWriter(c, cid, ms(r["time"]))
            cw.param("method", m)
            cw.param("label", res["label"])
            cw.param("model", res["model"])
            cw.param("threshold", res["threshold"])
            for k, v in res.items():
                if k == "auroc_ci" and v:
                    cw.metric("test.auroc_ci_low", v[0])
                    cw.metric("test.auroc_ci_high", v[1])
                elif k.startswith("val_"):
                    cw.metric(f"val.{k[4:]}", v)
                elif k not in ("label", "model", "threshold"):
                    cw.metric(f"test.{k}", v)
            cw.flush()
            c.set_terminated(cid, "FINISHED", end_time=ms(r["time"]))


# ---------- M2 detection ----------

def real_photo_rows():
    """Per-run rows of the real-photo table, keyed by (arm, seed)."""
    rows, header = {}, None
    in_table = False
    for line in REAL_MD.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Per run"):
            in_table = True
            continue
        if not in_table or not line.startswith("|"):
            continue
        cells = [x.strip() for x in line.strip("|").split("|")]
        if header is None:
            header = cells
        elif not set(cells[0]) <= set("-: "):
            m = re.match(r"([AB]) seed (\d+)", cells[0])
            if m:
                rows[(m.group(1), int(m.group(2)))] = dict(zip(header, cells))
    return rows


REAL_KEYS = {"thr": "threshold", "AP50 OBJ011": "ap50_OBJ011", "AP50 OBJ002": "ap50_OBJ002",
             "mAP50 (2 classes)": "map50_2cls", "recall OBJ011": "recall_OBJ011", "recall OBJ002": "recall_OBJ002",
             "precision": "precision", "false alarms (other classes)": "false_alarms_other_classes",
             "other FP (bad box / duplicate)": "other_fp_bad_box_or_duplicate"}


def log_det_block(w, prefix, block, classes):
    w.metric(f"{prefix}.map50", block.get("map50"))
    w.metric(f"{prefix}.map50_95", block.get("map50_95"))
    at = block.get("at_threshold") or {}
    for k in ("precision", "recall", "macro_f1", "tp", "fp", "fn"):
        w.metric(f"{prefix}.{k}", at.get(k))
    for i, f1 in enumerate(at.get("per_class_f1") or []):
        w.metric(f"{prefix}.f1.{classes[i]}", f1)
    for k, v in (at.get("errors") or {}).items():
        w.metric(f"{prefix}.errors.{k}", v)


def model_card(r, sel, classes):
    return f"""# {r['run_tag']}

- Model: {sel['model']} (Ultralytics YOLOv8n, {len(classes)} classes {', '.join(classes[i] for i in sorted(classes))})
- Input: RGB image of any size; letterboxed to {r['config']['imgsz']}×{r['config']['imgsz']} inside Ultralytics; boxes are returned in original-image pixels
- Confidence threshold: {sel['general_threshold']} ({sel['threshold_rule']}); NMS IoU {sel['nms_iou']}
- Checkpoint rule: {sel['checkpoint_rule']}
- Trained on synthetic composites (D1{' + D3' if sel.get('arm') == 'B' else ''}) of single-object photos; seed {sel.get('train_seed')}
- Limitations: only the 20 course objects; anything else may be reported as one of them (no OTHERS class);
  real multi-object photos were only tested for OBJ011 and OBJ002 (detection/real_multi); CPU latency not benchmarked yet
"""


def import_m2(c, force, mapping):
    real = real_photo_rows()
    for d in sorted(RUNS.glob(M2_PATTERN)):
        r = json.loads((d / "records/run_yolov8n_d1.json").read_text(encoding="utf-8"))
        sel = r["selection"]
        arm, seed = sel["arm"], int(sel["train_seed"])
        classes = {int(k): v for k, v in json.loads((d / "data/classes.json").read_text(encoding="utf-8")).items()}
        run, rid = new_run(c, EXP_M2, r["run_id"], r["run_tag"], sel["decided_at"], force)
        mapping.append((EXP_M2, r["run_id"], r["run_tag"], rel(d), rid))
        if run is None:
            continue
        ts = ms(sel["decided_at"])
        w = RunWriter(c, rid, ts)

        # params: generator / training config, Ultralytics train args, selection rules
        w.params_from(r["config"], "config.")
        train_args = yaml.safe_load((d / "figures/train_args.yaml").read_text(encoding="utf-8"))
        w.params_from({k: v for k, v in train_args.items() if k not in ("data", "project", "save_dir")}, "yolo.")
        w.param("seed", seed)
        w.param("arm", arm)
        w.param("arm_description", r["arm"])
        w.param("model", r["model"])
        for k in ("checkpoint_rule", "threshold_rule", "general_threshold", "nms_iou", "best_epoch", "epochs_run", "train_aug"):
            w.param(f"selection.{k}", sel.get(k))
        w.params_from({k: v for k, v in r["data"].items() if not k.endswith("sha256")}, "data.")

        # dataset checksums: recorded values + recomputed hashes of the manifests in the run folder
        recorded = flatten({k: v for k, v in r["data"].items() if k.endswith("sha256")})
        for k, v in recorded.items():
            w.param(f"checksum.recorded.{k}", v)
        files = sorted(d.glob("data/*manifest.csv")) + [d / "data/classes.json"]
        local = {f.name: sha256(f) for f in files}
        for name, h in local.items():
            w.param(f"checksum.file.{name}", h)
        pairs = {"classes.json": recorded.get("classes_json_sha256"), "masks_manifest.csv": recorded.get("masks_manifest_sha256"),
                 "d1_train_manifest.csv": recorded.get("d1_manifest_sha256.train"),
                 "d1_val_manifest.csv": recorded.get("d1_manifest_sha256.val"),
                 "d1_test_manifest.csv": recorded.get("d1_manifest_sha256.test")}
        checked = {n: local.get(n) == h for n, h in pairs.items() if h}
        w.tag("checksums_match_record", all(checked.values()) if checked else "no recorded checksums")
        w.tag("checksums_checked", ",".join(sorted(checked)))

        # metrics
        vm, fm = r["validation_metrics"], r["final_test_metrics"]
        log_det_block(w, "val.d1", vm, classes)
        log_det_block(w, "val.hard", vm.get("hardval") or {}, classes)
        w.metric("val.score", vm.get("val_score"))
        w.metric("val.single_object.top1_acc", (vm.get("single_object") or {}).get("top1_acc"))
        log_det_block(w, "test.d1", fm.get("d1_test") or {}, classes)
        log_det_block(w, "test.thard", fm.get("t_hard") or {}, classes)
        w.metric("test.single_object.top1_acc", (fm.get("single_object_test") or {}).get("top1_acc"))
        w.metric("training.minutes", r["training"]["minutes"])
        w.metric("selection.general_threshold", sel["general_threshold"])
        row = real.get((arm, seed))
        if row:
            for col, k in REAL_KEYS.items():
                w.metric(f"test.real.{k}", row.get(col))
        w.tag("real_photo_results", "logged as test.real.*" if row else "not in real_multi_results.md")
        with open(d / "figures/train_results.csv", encoding="utf-8") as f:
            for e in csv.DictReader(f):
                step = int(float(e["epoch"]))
                for k, v in e.items():
                    if k != "epoch":
                        w.metric(k, v, step=step)

        # tags
        w.tag("task", r["task"])
        w.tag("run_tag", r["run_tag"])
        w.tag("arm", arm)
        w.tag("seed", seed)
        w.tag("git_commit", r["git_commit"])
        w.tag("git_dirty", r["git_dirty"])
        w.tag("gpu", r["training"].get("gpu"))
        w.tag("imported_from", rel(d))
        for k, v in flatten(r["environment"]).items():
            w.tag(f"env.{k}", v)
        w.flush()

        # artifacts
        log_files(c, rid, [d / "M2_results_for_report.md"])
        log_files(c, rid, sorted(d.glob("records/*")), "records")
        log_files(c, rid, sorted(p for p in d.glob("figures/*")), "figures")
        log_files(c, rid, files + sorted(d.glob("data/*.yaml")), "data")
        weights = d / "models/d1_yolov8n_best.pt"
        ok = sha256(weights) == r["artifact"]["sha256"]
        c.set_tag(rid, "weights_sha256", r["artifact"]["sha256"])
        c.set_tag(rid, "weights_sha256_verified", ok)
        if not ok:
            raise SystemExit(f"{weights}: SHA-256 differs from the run record")
        tmp = STORE / "_tmp_model" / r["run_tag"]
        tmp.mkdir(parents=True, exist_ok=True)
        serving = {"weights": weights.name, "weights_sha256": r["artifact"]["sha256"], "imgsz": r["config"]["imgsz"],
                   "letterbox": True, "conf_threshold": sel["general_threshold"], "nms_iou": sel["nms_iou"],
                   "classes_json_sha256": sha256(d / "data/classes.json"), "runtime": "pytorch (ultralytics)",
                   "ultralytics_trained_with": r["environment"].get("ultralytics"), "source_run_tag": r["run_tag"],
                   "arm": arm, "seed": seed}
        (tmp / "serving.json").write_text(json.dumps(serving, indent=2), encoding="utf-8")
        (tmp / "MODEL_CARD.md").write_text(model_card(r, sel, classes), encoding="utf-8")
        log_files(c, rid, [weights, d / "data/classes.json", tmp / "serving.json", tmp / "MODEL_CARD.md"], MODEL_SUBDIR)
        c.set_terminated(rid, "FINISHED", end_time=ts)
        print(f"  M2 {r['run_tag']} (arm {arm}, seed {seed}, thr {sel['general_threshold']}) → {rid}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="delete and re-import runs that were already imported")
    args = ap.parse_args(argv)
    c = client()
    mapping = []
    print("M1 classification"); import_m1(c, args.force, mapping)
    print("Open-set rejection"); import_openset(c, args.force, mapping)
    print("M2 detection"); import_m2(c, args.force, mapping)
    out = STORE / "run_id_map.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["experiment", "source_run_id", "source_run_tag", "source_record", "mlflow_run_id"])
        wr.writerows(mapping)
    print(f"{len(mapping)} runs; mapping written to {rel(out)}")


if __name__ == "__main__":
    main()
