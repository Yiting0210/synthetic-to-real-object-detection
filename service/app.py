"""Minimal inference service for the Lab-to-Real detector (proposal §7.3).

    GET  /health   readiness, model name / version / source, time to ready
    POST /detect   multipart field `file` = image; optional query `conf`
                   → class IDs, course IDs (class names), confidences, boxes [x1, y1, x2, y2] in original-image pixels

Boxes are in the pixel frame of the uploaded image after its EXIF orientation is applied (the way a viewer shows it);
width/height in the response are that frame's size. The model is chosen by environment variables (see model_loader.py).

Run locally:  uvicorn service.app:app --port 8000      (from the repository root)
"""
import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, ImageOps

from .model_loader import load_from_env

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("service")
MAX_BYTES = 20 * 1024 * 1024
STATE = {}


@asynccontextmanager
async def lifespan(app):
    det, seconds = load_from_env()      # fails startup (no "ready" service) if the model cannot be loaded or checked
    STATE.update(detector=det, ready_seconds=round(seconds, 3))
    i = det.info
    log.info("model ready: %s v%s (source=%s, location=%s, run=%s) in %.2fs",
             i.name, i.version, i.source, i.location, i.run_id, seconds)
    yield
    STATE.clear()


app = FastAPI(title="Lab-to-Real Object ID", version="1.0", lifespan=lifespan)


def model_block():
    i = STATE["detector"].info
    return {"name": i.name, "version": i.version, "source": i.source, "run_id": i.run_id, "alias": i.alias, **i.extra}


@app.get("/health")
def health():
    det = STATE.get("detector")
    if det is None:
        raise HTTPException(503, "model not loaded")
    return {"status": "ready", "model": model_block(), "ready_seconds": STATE["ready_seconds"],
            "readiness": getattr(det, "readiness", None),
            "confidence_threshold": det.conf, "nms_iou": det.nms_iou, "imgsz": det.imgsz,
            "runtime": "pytorch-cpu (ultralytics)", "n_classes": len(det.classes)}


@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = Query(None, gt=0, lt=1)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 20 MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            image = ImageOps.exif_transpose(im).convert("RGB")
    except Exception as e:      # PIL raises UnidentifiedImageError, OSError, ValueError, ... on bad input
        raise HTTPException(400, f"not a readable image: {e.__class__.__name__}")
    det = STATE["detector"]
    dets, used_conf = det(image, conf)
    return {"model": model_block(), "image": {"width": image.width, "height": image.height},
            "confidence_threshold": used_conf, "n_detections": len(dets), "detections": dets}
