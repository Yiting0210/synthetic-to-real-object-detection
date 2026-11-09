"""Export a registered model version as a self-contained model bundle (proposal §7.7).

The alias is resolved to a concrete version number at export time, so later alias moves never change a bundle.

    model_bundles/<model>-v<N>/
        d1_yolov8n_best.pt  classes.json  serving.json  MODEL_CARD.md
        bundle.json         model name, version, alias at export, source run ID, SHA-256 of every file

Usage:
    python tools/export_bundle.py --alias champion
    python tools/export_bundle.py --version 1
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
from mlops_common import BUNDLES, DETECTOR_MODEL, client, sha256  # noqa: E402


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--alias")
    g.add_argument("--version", type=int)
    ap.add_argument("--model", default=DETECTOR_MODEL)
    ap.add_argument("--out", type=Path, default=BUNDLES)
    a = ap.parse_args(argv)
    out, mv = export(client(), a.model, a.alias, a.version, a.out)
    print(f"exported {a.model} version {mv.version}" + (f" (alias {a.alias})" if a.alias else ""))
    print(out)


if __name__ == "__main__":
    main()
