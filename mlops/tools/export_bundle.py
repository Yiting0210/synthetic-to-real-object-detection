"""Export a registered model version as a self-contained model bundle (proposal §7.7).

The alias is resolved to a concrete version number at export time, so later alias moves never change a bundle.

    model_bundles/<model>-v<N>/
        d1_yolov8n_best.pt  classes.json  serving.json  MODEL_CARD.md
        bundle.json         model name, version, alias at export, source run ID, SHA-256 of every file

With --to-models it also refreshes the repository's models/ folder, which the demo and the service fallback read:
    models/d1_yolov8n_best.pt   weights of that version (not in git)
    models/selection.json       the run's selection record (threshold, rules, validation results) + a "registry" block
So models/ always shows the registry's choice instead of a hand-copied file.

Usage:
    python mlops/tools/export_bundle.py --alias champion --to-models
    python mlops/tools/export_bundle.py --version 1
Prints the bundle path on the last line.
"""
import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlops_common import BUNDLES, DETECTOR_MODEL, REPO, client, sha256  # noqa: E402


def export(c, model=DETECTOR_MODEL, alias=None, version=None, out_root=BUNDLES):
    import mlflow
    mv = c.get_model_version_by_alias(model, alias) if alias else c.get_model_version(model, str(version))
    out = Path(out_root) / f"{model}-v{mv.version}"
    tmp = Path(tempfile.mkdtemp(prefix="bundle-"))
    try:
        local = Path(mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(tmp)))
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(local, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    files = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "bundle.json"}
    meta = {"model_name": model, "version": int(mv.version), "alias_at_export": alias, "run_id": mv.run_id,
            "source": mv.source, "tags": dict(mv.tags), "exported_at": datetime.now(timezone.utc).isoformat(),
            "files_sha256": files}
    (out / "bundle.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, mv


def to_models(c, out, mv, alias, models_dir=REPO / "models"):
    """Copy the bundle's weights into models/ and write models/selection.json from the source run."""
    sel = json.loads(Path(c.download_artifacts(mv.run_id, "records/selection.json")).read_text(encoding="utf-8"))
    sel["registry"] = {"model_name": mv.name, "version": int(mv.version), "alias_at_export": alias, "run_id": mv.run_id,
                       "arm": mv.tags.get("arm"), "seed": mv.tags.get("seed"), "source_run_tag": mv.tags.get("source_run_tag"),
                       "bundle": out.relative_to(REPO).as_posix(), "exported_at": datetime.now(timezone.utc).isoformat()}
    assert sha256(out / "d1_yolov8n_best.pt") == sel["weights_sha256"], "bundle weights do not match the run's selection record"
    shutil.copy2(out / "d1_yolov8n_best.pt", models_dir / "d1_yolov8n_best.pt")
    (models_dir / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    return sel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--alias")
    g.add_argument("--version", type=int)
    ap.add_argument("--model", default=DETECTOR_MODEL)
    ap.add_argument("--out", type=Path, default=BUNDLES)
    ap.add_argument("--to-models", action="store_true", help="also update models/d1_yolov8n_best.pt and models/selection.json")
    a = ap.parse_args(argv)
    c = client()
    out, mv = export(c, a.model, a.alias, a.version, a.out)
    print(f"exported {a.model} version {mv.version}" + (f" (alias {a.alias})" if a.alias else ""))
    if a.to_models:
        sel = to_models(c, out, mv, a.alias)
        print(f"models/ updated: version {mv.version} (arm {sel['registry']['arm']}), threshold {sel['general_threshold']}")
    print(out)


if __name__ == "__main__":
    main()
