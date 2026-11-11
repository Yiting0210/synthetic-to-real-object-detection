"""The demo (demo/predict.py) and the service (service/model_loader.py) each wrap the detector.
Both must return the same objects for the same image and threshold. Uses models/d1_yolov8n_best.pt (not in git; skips without it).
"""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
WEIGHTS = MODELS / "d1_yolov8n_best.pt"
IMAGES = REPO / "detection/data/real_photos/images"
PHOTOS = ["IMG_2291.jpg", "IMG_2296.jpg", "IMG_2300.jpg"]      # landscape and portrait

pytestmark = pytest.mark.skipif(not WEIGHTS.exists(), reason="models/d1_yolov8n_best.pt not available locally")
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(os.environ.get("TEMP", "/tmp")) / "ultralytics_cfg"))


@pytest.fixture(scope="module")
def detectors():
    sys.path.insert(0, str(REPO / "demo"))
    try:
        import predict as demo
    finally:
        sys.path.pop(0)
    from service.model_loader import load_from_path
    return demo, demo.Detector(MODELS), load_from_path(WEIGHTS, MODELS / "classes.json")


def test_same_threshold(detectors):
    _, demo_det, service_det = detectors
    assert demo_det.conf == service_det.conf


@pytest.mark.parametrize("name", PHOTOS)
@pytest.mark.parametrize("conf", [None, 0.25])
def test_same_detections(detectors, name, conf):
    demo, demo_det, service_det = detectors
    image = demo.load_image(IMAGES / name)          # EXIF rotation applied, as in the service
    a = demo_det(image, conf)["objects"]
    b, _ = service_det(image, conf)
    assert len(a) > 0
    assert sorted(((o["object_id"], o["confidence"], o["box_xyxy"]) for o in a), key=str) == \
           sorted(((o["class_name"], o["confidence"], o["box_xyxy"]) for o in b), key=str)
