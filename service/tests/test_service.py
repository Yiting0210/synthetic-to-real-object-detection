"""Tests for the inference service (run from the repository root: pytest service/tests).

Uses the arm B seed-42 detector from its run folder (weights are not in git; tests skip if it is missing) and fixed
real photos from detection/real_multi (hand-drawn YOLO labels, class 6 = OBJ011 mug, class 2 = OBJ002 paper roll).
"""
import io
import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "detection/cluster/IE7615_project/runs/m2_20261108_054413_armB_seed42"
WEIGHTS = RUN / "models/d1_yolov8n_best.pt"
CLASSES = RUN / "data/classes.json"
IMAGES = REPO / "detection/real_multi/images"
LABELS = REPO / "detection/real_multi/labels"
FIXTURES = ["IMG_2291.jpg", "IMG_2296.jpg", "IMG_2300.jpg"]      # landscape and portrait photos
DB = REPO / "mlops/mlflow.db"

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="arm B seed-42 weights not available locally")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(os.environ.get("TEMP", "/tmp")) / "ultralytics_cfg"))

PATH_ENV = {"MODEL_PATH": str(WEIGHTS), "MODEL_CLASSES": str(CLASSES), "MODEL_CONF": "0.7"}
MODEL_ENV_KEYS = ("MODEL_BUNDLE", "MODEL_ALIAS", "MODEL_PATH", "MODEL_CLASSES", "MODEL_CONF", "MODEL_NAME")


def make_client(monkeypatch, env):
    for k in MODEL_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    from service.app import app
    return TestClient(app)


@pytest.fixture
def client(monkeypatch):
    with make_client(monkeypatch, PATH_ENV) as c:
        yield c


def post_image(client, path_or_bytes, name="x.jpg", **params):
    data = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes, (str, Path)) else path_or_bytes
    return client.post("/detect", files={"file": (name, data, "image/jpeg")}, params=params)


def gt_boxes(name, w, h):
    out = []
    for line in (LABELS / Path(name).with_suffix(".txt")).read_text().split("\n"):
        if line.strip():
            c, cx, cy, bw, bh = map(float, line.split())
            out.append((int(c), [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]))
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def jpeg(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    return buf.getvalue()


# ---------- health / startup ----------

def test_health_reports_ready_model(client):
    r = client.get("/health").json()
    assert r["status"] == "ready"
    assert r["model"]["source"] == "path"
    assert r["confidence_threshold"] == 0.7 and r["imgsz"] == 640 and r["n_classes"] == 20


def test_model_missing_at_startup_fails(monkeypatch, tmp_path):
    from service.model_loader import ModelLoadError
    with pytest.raises(ModelLoadError):
        with make_client(monkeypatch, {"MODEL_PATH": str(tmp_path / "missing.pt")}):
            pass


def test_class_map_mismatch_is_rejected(tmp_path):
    from service.model_loader import ModelLoadError, load_from_path
    m = json.loads(CLASSES.read_text(encoding="utf-8"))
    m["0"], m["1"] = m["1"], m["0"]
    bad = tmp_path / "classes.json"
    bad.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ModelLoadError, match="class names"):
        load_from_path(WEIGHTS, bad)


# ---------- /detect on fixed real photos ----------

@pytest.mark.parametrize("name", FIXTURES)
def test_detect_real_photo_matches_hand_labels(client, name):
    path = IMAGES / name
    w, h = Image.open(path).size
    r = post_image(client, path, name)
    assert r.status_code == 200
    body = r.json()
    assert body["image"] == {"width": w, "height": h}
    classes = {int(k): v for k, v in json.loads(CLASSES.read_text(encoding="utf-8")).items()}
    for d in body["detections"]:
        assert classes[d["class_id"]] == d["class_name"]          # index ↔ course-ID mapping
        assert 0.7 <= d["confidence"] <= 1
        x1, y1, x2, y2 = d["box_xyxy"]
        assert -1 <= x1 < x2 <= w + 1 and -1 <= y1 < y2 <= h + 1  # original-image pixels, not 640×640
    matched = [g for g in gt_boxes(name, w, h)
               if any(d["class_id"] == g[0] and iou(d["box_xyxy"], g[1]) >= 0.5 for d in body["detections"])]
    assert matched, f"no hand-labelled object found in {name}: {body['detections']}"


def test_boxes_follow_original_resolution(client):
    """The same photo at half size gives the same boxes at half the coordinates."""
    path = IMAGES / FIXTURES[0]
    full = post_image(client, path).json()["detections"]
    im = Image.open(path)
    half = post_image(client, jpeg(im.resize((im.width // 2, im.height // 2), Image.BILINEAR))).json()["detections"]
    top = max(full, key=lambda d: d["confidence"])
    best = max((d for d in half if d["class_id"] == top["class_id"]),
               key=lambda d: iou([v * 2 for v in d["box_xyxy"]], top["box_xyxy"]))
    assert iou([v * 2 for v in best["box_xyxy"]], top["box_xyxy"]) > 0.8


def test_exif_rotation_is_applied(client):
    """A photo stored sideways with EXIF orientation 6 is processed (and sized) as it is displayed."""
    path = IMAGES / FIXTURES[0]
    im = Image.open(path)
    exif = Image.Exif()
    exif[274] = 6                                     # display = rotate 90° clockwise
    buf = io.BytesIO()
    im.rotate(90, expand=True).save(buf, "JPEG", quality=95, exif=exif)
    body = post_image(client, buf.getvalue()).json()
    assert body["image"] == {"width": im.width, "height": im.height}
    assert body["n_detections"] >= 1


def test_empty_scene_gives_no_detections(client):
    body = post_image(client, jpeg(Image.new("RGB", (800, 600), (128, 128, 128)))).json()
    assert body["n_detections"] == 0 and body["detections"] == []


def test_conf_query_overrides_threshold(client):
    body = post_image(client, IMAGES / FIXTURES[0], conf=0.3).json()
    assert body["confidence_threshold"] == 0.3
    assert all(d["confidence"] >= 0.3 for d in body["detections"])
    assert post_image(client, IMAGES / FIXTURES[0], conf=1.5).status_code == 422


@pytest.mark.parametrize("payload", [b"", b"not an image at all", jpeg(Image.new("RGB", (64, 64)))[:200]])
def test_invalid_or_corrupt_image_is_400(client, payload):
    assert post_image(client, payload).status_code == 400


# ---------- bundle and registry loading ----------

def make_bundle(tmp_path):
    b = tmp_path / "lab2real-detector-v9"
    b.mkdir()
    shutil.copy(WEIGHTS, b / WEIGHTS.name)
    shutil.copy(CLASSES, b / "classes.json")
    from service.model_loader import sha256
    (b / "serving.json").write_text(json.dumps({"weights": WEIGHTS.name, "imgsz": 640, "conf_threshold": 0.7, "nms_iou": 0.7}))
    files = {p.name: sha256(p) for p in b.iterdir()}
    (b / "bundle.json").write_text(json.dumps({"model_name": "lab2real-detector", "version": 9, "run_id": "test",
                                               "files_sha256": files}))
    return b


def test_bundle_loads_and_reports_version(monkeypatch, tmp_path):
    with make_client(monkeypatch, {"MODEL_BUNDLE": str(make_bundle(tmp_path))}) as c:
        body = post_image(c, IMAGES / FIXTURES[0]).json()
        assert body["model"]["version"] == "9" and body["model"]["source"] == "bundle"


def test_tampered_bundle_is_refused(tmp_path):
    from service.model_loader import ModelLoadError, load_bundle
    b = make_bundle(tmp_path)
    (b / "serving.json").write_text(json.dumps({"weights": WEIGHTS.name, "imgsz": 640, "conf_threshold": 0.05, "nms_iou": 0.7}))
    with pytest.raises(ModelLoadError, match="checksum"):
        load_bundle(b)


@pytest.mark.skipif(not DB.exists(), reason="local MLflow store not created (run tools/import_runs.py)")
def test_registry_alias_resolves_to_registered_version(monkeypatch):
    from mlflow import MlflowClient
    uri = "sqlite:///" + DB.as_posix()
    mv = MlflowClient(uri, uri).get_model_version_by_alias("lab2real-detector", "champion")
    with make_client(monkeypatch, {"MODEL_ALIAS": "champion", "MLFLOW_TRACKING_URI": uri}) as c:
        m = c.get("/health").json()["model"]
        assert m["source"] == "registry" and m["version"] == str(mv.version) and m["run_id"] == mv.run_id


def test_unknown_alias_falls_back_to_path(monkeypatch, tmp_path):
    env = {"MODEL_ALIAS": "champion", "MODEL_NAME": "no-such-model", **PATH_ENV,
           "MLFLOW_TRACKING_URI": "sqlite:///" + (tmp_path / "empty.db").as_posix()}
    with make_client(monkeypatch, env) as c:
        assert c.get("/health").json()["model"]["source"] == "path"
