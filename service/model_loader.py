"""Load the YOLOv8n detector for the inference service.

Three ways to point the service at a model (first one set wins):
    MODEL_BUNDLE=<dir>     model bundle exported by mlops/tools/export_bundle.py (bundle.json + checksums). Used in Docker.
    MODEL_ALIAS=<alias>    resolve <alias> (e.g. champion) in the local MLflow Model Registry at startup
                           (MLFLOW_TRACKING_URI, default sqlite:///<repo>/mlops/store/mlflow.db; MODEL_NAME, default lab2real-detector)
    MODEL_PATH=<file.pt>   plain weights file (fallback). MODEL_CLASSES=<classes.json> and MODEL_CONF=<float> are optional.
If MODEL_ALIAS is set but the registry cannot be read and MODEL_PATH is also set, the service falls back to MODEL_PATH.
With nothing set it uses the repository's models/d1_yolov8n_best.pt (+ models/classes.json, models/selection.json).

A model directory (bundle, or a registry version's artifact folder) holds:
    <weights>.pt   classes.json   serving.json (imgsz, conf_threshold, nms_iou, weights, weights_sha256)   [bundle.json]
"""
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("service")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "lab2real-detector"
DEFAULT_IMGSZ, DEFAULT_CONF, DEFAULT_NMS_IOU = 640, 0.25, 0.7


class ModelLoadError(RuntimeError):
    pass


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def read_classes(path):
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): v for k, v in m.items()}


@dataclass
class ModelInfo:
    name: str
    version: str
    source: str                 # bundle | registry | path
    location: str
    run_id: str = None
    alias: str = None
    extra: dict = field(default_factory=dict)


class Detector:
    """YOLOv8n wrapper. Boxes come back in the pixel coordinates of the image passed in
    (Ultralytics letterboxes to imgsz internally and maps boxes back)."""

    def __init__(self, weights, classes, conf, nms_iou=DEFAULT_NMS_IOU, imgsz=DEFAULT_IMGSZ, info=None):
        from ultralytics import YOLO
        if not Path(weights).exists():
            raise ModelLoadError(f"model weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.classes = classes
        names = {int(k): v for k, v in self.model.names.items()}
        if names != classes:
            raise ModelLoadError("detector class names do not match classes.json (model and class map are from different runs)")
        self.conf, self.nms_iou, self.imgsz = float(conf), float(nms_iou), int(imgsz)
        self.info = info

    def __call__(self, image: Image.Image, conf=None):
        conf = self.conf if conf is None else float(conf)
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        res = self.model.predict(bgr, conf=conf, iou=self.nms_iou, imgsz=self.imgsz, device="cpu", verbose=False)[0]
        dets = [dict(class_id=int(c), class_name=self.classes[int(c)], confidence=round(float(s), 4),
                     box_xyxy=[round(float(v), 1) for v in b])
                for c, s, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy)]
        dets.sort(key=lambda d: -d["confidence"])
        return dets, conf

    def self_check(self):
        """Readiness: one inference on a fixed blank image must run and return a list."""
        dets, _ = self(Image.new("RGB", (self.imgsz, self.imgsz), (114, 114, 114)))
        return isinstance(dets, list)


def _from_model_dir(d, info, verify=None):
    d = Path(d)
    serving = json.loads((d / "serving.json").read_text(encoding="utf-8"))
    weights = d / serving["weights"]
    if verify:
        for rel, expected in verify.items():
            p = d / rel
            if not p.exists():
                raise ModelLoadError(f"bundle file missing: {rel}")
            if sha256(p) != expected:
                raise ModelLoadError(f"bundle checksum mismatch: {rel}")
    elif serving.get("weights_sha256") and sha256(weights) != serving["weights_sha256"]:
        raise ModelLoadError("weights SHA-256 differs from serving.json")
    classes = read_classes(d / "classes.json")
    if serving.get("classes_json_sha256") and sha256(d / "classes.json") != serving["classes_json_sha256"]:
        raise ModelLoadError("classes.json SHA-256 differs from serving.json")
    info.extra.update({k: serving.get(k) for k in ("arm", "seed", "source_run_tag")})
    return Detector(weights, classes, conf=serving["conf_threshold"], nms_iou=serving.get("nms_iou", DEFAULT_NMS_IOU),
                    imgsz=serving.get("imgsz", DEFAULT_IMGSZ), info=info)


def load_bundle(bundle_dir):
    b = Path(bundle_dir)
    if not (b / "bundle.json").exists():
        raise ModelLoadError(f"not a model bundle (no bundle.json): {b}")
    meta = json.loads((b / "bundle.json").read_text(encoding="utf-8"))
    info = ModelInfo(meta["model_name"], str(meta["version"]), "bundle", str(b), meta.get("run_id"),
                     meta.get("alias_at_export"))
    return _from_model_dir(b, info, verify=meta["files_sha256"])


def load_from_registry(alias, model_name=DEFAULT_MODEL_NAME, tracking_uri=None):
    import mlflow
    from mlflow import MlflowClient
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + (REPO / "mlops/store/mlflow.db").as_posix()
    c = MlflowClient(tracking_uri=uri, registry_uri=uri)
    mv = c.get_model_version_by_alias(model_name, alias)
    tmp = Path(tempfile.mkdtemp(prefix=f"{model_name}-v{mv.version}-"))
    mlflow.set_tracking_uri(uri)
    local = mlflow.artifacts.download_artifacts(artifact_uri=mv.source, dst_path=str(tmp), tracking_uri=uri)
    info = ModelInfo(model_name, str(mv.version), "registry", f"models:/{model_name}@{alias}", mv.run_id, alias)
    try:
        return _from_model_dir(local, info)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_from_path(weights, classes_file=None, conf=None):
    weights = Path(weights)
    if not weights.exists():
        raise ModelLoadError(f"model weights not found: {weights}")
    classes_file = Path(classes_file) if classes_file else weights.with_name("classes.json")
    sel = weights.with_name("selection.json")
    if conf is None:
        conf = json.loads(sel.read_text(encoding="utf-8"))["general_threshold"] if sel.exists() else DEFAULT_CONF
    info = ModelInfo(weights.stem, "sha256:" + sha256(weights)[:12], "path", str(weights))
    if classes_file.exists():
        classes = read_classes(classes_file)
    else:
        from ultralytics import YOLO
        classes = {int(k): v for k, v in YOLO(str(weights)).names.items()}
    return Detector(weights, classes, conf=conf, info=info)


def load_from_env(env=None):
    """Load per the environment variables above; returns (detector, seconds to ready)."""
    env = os.environ if env is None else env
    t0 = time.perf_counter()
    path = env.get("MODEL_PATH")
    conf = float(env["MODEL_CONF"]) if env.get("MODEL_CONF") else None
    if env.get("MODEL_BUNDLE"):
        det = load_bundle(env["MODEL_BUNDLE"])
    elif env.get("MODEL_ALIAS"):
        try:
            det = load_from_registry(env["MODEL_ALIAS"], env.get("MODEL_NAME", DEFAULT_MODEL_NAME), env.get("MLFLOW_TRACKING_URI"))
        except ModelLoadError:
            raise
        except Exception as e:
            if not path:
                raise ModelLoadError(f"registry alias {env['MODEL_ALIAS']!r} could not be resolved: {e}") from e
            log.warning("registry alias %s not usable (%s); falling back to MODEL_PATH=%s", env["MODEL_ALIAS"], e, path)
            det = load_from_path(path, env.get("MODEL_CLASSES"), conf)
    else:
        det = load_from_path(path or REPO / "models/d1_yolov8n_best.pt", env.get("MODEL_CLASSES"), conf)
    if conf is not None:
        det.conf = conf
    if not det.self_check():
        raise ModelLoadError("readiness self-check failed")
    return det, time.perf_counter() - t0
