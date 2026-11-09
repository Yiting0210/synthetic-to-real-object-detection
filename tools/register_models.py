"""Register the seed-42 detectors of arm A (D1) and arm B (D1+D3) as versions of one registered model (proposal §7.7).

Each version points to the `model/` artifact folder of its imported MLflow run (weights, classes.json, serving.json,
MODEL_CARD.md) and carries tags for the class map, preprocessing, thresholds, data manifest and runtime.
Run tools/import_runs.py first. Re-running does not create duplicate versions.

Aliases: `champion` = version intended for use; `previous` = last good version. On first registration champion → arm A
(the M2 report's reference arm). tools/rollback_demo.py moves them.

Usage:  python tools/register_models.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlops_common import DETECTOR_MODEL, MODEL_SUBDIR, client, experiment_id  # noqa: E402

TO_REGISTER = [("A", 42), ("B", 42)]       # registration order = version numbers 1, 2


def register(c, arm, seed):
    exp = experiment_id(c, "M2-detection")
    runs = c.search_runs([exp], filter_string=f"tags.arm = '{arm}' and tags.seed = '{seed}'")
    if len(runs) != 1:
        raise SystemExit(f"expected one imported run for arm {arm} seed {seed}, found {len(runs)}; run import_runs.py")
    run = runs[0]
    for mv in c.search_model_versions(f"name = '{DETECTOR_MODEL}'"):
        if mv.run_id == run.info.run_id:
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
    desc = (f"YOLOv8n, arm {arm} ({'D1+D3' if arm == 'B' else 'D1'}), seed {seed}, threshold {serving['conf_threshold']}. "
            f"D1-test mAP50-95 {m.get('test.d1.map50_95', float('nan')):.3f}, T-hard {m.get('test.thard.map50_95', float('nan')):.3f}, "
            f"real photos mAP50 (2 classes) {m.get('test.real.map50_2cls', float('nan')):.3f}. "
            "Inputs: RGB photos of the 20 course objects. Limitations: no OTHERS class; real-photo check covers 2 classes only.")
    mv = c.create_model_version(DETECTOR_MODEL, source=source, run_id=run.info.run_id, tags=tags, description=desc)
    print(f"  arm {arm} seed {seed}: registered as version {mv.version} (run {run.info.run_id})")
    return mv


def main():
    c = client()
    try:
        c.get_registered_model(DETECTOR_MODEL)
    except Exception:
        c.create_registered_model(DETECTOR_MODEL, description="Lab-to-Real multi-object detector (YOLOv8n, 20 course objects).")
    versions = [register(c, a, s) for a, s in TO_REGISTER]
    rm = c.get_registered_model(DETECTOR_MODEL)
    if "champion" not in rm.aliases:
        c.set_registered_model_alias(DETECTOR_MODEL, "champion", versions[0].version)
        print(f"  champion → version {versions[0].version}")
    print("aliases:", dict(c.get_registered_model(DETECTOR_MODEL).aliases))


if __name__ == "__main__":
    main()
