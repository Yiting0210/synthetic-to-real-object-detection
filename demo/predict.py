"""M3 model interface: single-object image → object ID; multi-object image → IDs and locations of all objects.

Models (copied from the M1 / M2 run folders on Drive into one folder, default ../models, the repository-level models/ folder):
    classes.json          class index ↔ course ID (from M1, shared by both models)
    mobilenetv2.keras     M1 classifier selected on validation (preprocessing is built into the model)
    d1_yolov8n_best.pt    M2 detector
    selection.json        M2 selection record; its general_threshold is the default confidence threshold

Usage:
    python predict.py single path/to/image.jpg
    python predict.py multi  path/to/composite.jpg --save annotated.jpg
    python predict.py multi  photo.jpg --conf 0.4 --models-dir path/to/models

Output is JSON on stdout. TensorFlow is only imported for `single`, Ultralytics only for `multi`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

CLASSIFIER_FILE = "mobilenetv2.keras"
DETECTOR_FILE = "d1_yolov8n_best.pt"
CLS_SIZE = 224          # M1 input size
DET_SIZE = 640          # M2 input size
NMS_IOU = 0.7           # same as M2
DEFAULT_CONF = 0.25     # used only if selection.json is missing


def load_classes(models_dir):
    m = json.loads((Path(models_dir) / "classes.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in m.items()}


def load_image(path):
    """RGB image with the EXIF rotation applied (phone photos)."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class Classifier:
    """M1 Keras classifier. Same preprocessing as M1: resize with padding to 224×224, RGB 0–255;
    the model-specific scaling is the model's own first layer."""

    def __init__(self, models_dir, file=CLASSIFIER_FILE):
        import keras

        @keras.saving.register_keras_serializable(package="m1")
        class CaffePreprocess(keras.layers.Layer):     # needed only if the ResNet-50 model from M1 is loaded
            def call(self, x):
                x = keras.ops.flip(x, axis=-1)
                return x - keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)

        import tensorflow as tf
        self.tf = tf
        self.model = keras.models.load_model(Path(models_dir) / file, compile=False)
        self.classes = load_classes(models_dir)
        self.version = file

    def __call__(self, image, top_k=3):
        tf = self.tf
        x = tf.image.resize_with_pad(tf.cast(np.asarray(image), tf.float32), CLS_SIZE, CLS_SIZE)
        x = tf.clip_by_value(tf.round(x), 0, 255)[None]
        p = self.model.predict(x, verbose=0)[0]
        top = np.argsort(p)[::-1][:top_k]
        return dict(object_id=self.classes[int(top[0])], confidence=round(float(p[top[0]]), 4),
                    top_k=[dict(object_id=self.classes[int(k)], probability=round(float(p[k]), 4)) for k in top],
                    model=self.version)


class Detector:
    """M2 YOLOv8n detector. Boxes are returned in the input image's pixel coordinates (Ultralytics undoes the letterbox)."""

    def __init__(self, models_dir, file=DETECTOR_FILE, conf=None):
        from ultralytics import YOLO
        self.model = YOLO(str(Path(models_dir) / file))
        self.classes = load_classes(models_dir)
        sel = Path(models_dir) / "selection.json"
        self.default_conf = json.loads(sel.read_text(encoding="utf-8"))["general_threshold"] if sel.exists() else DEFAULT_CONF
        self.conf = conf if conf is not None else self.default_conf
        names = {int(k): v for k, v in self.model.names.items()}
        if [names[i] for i in sorted(names)] != [self.classes[i] for i in sorted(self.classes)]:
            raise ValueError("Detector class names do not match classes.json; the model and the class map are from different runs.")
        self.version = file

    def __call__(self, image, conf=None):
        conf = self.conf if conf is None else conf
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])     # Ultralytics expects BGR arrays, like cv2.imread
        res = self.model.predict(bgr, conf=conf, iou=NMS_IOU, imgsz=DET_SIZE, device="cpu", verbose=False)[0]
        objs = [dict(object_id=self.classes[int(c)], confidence=round(float(s), 4), box_xyxy=[round(float(v), 1) for v in b])
                for c, s, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy)]
        objs.sort(key=lambda d: (d["box_xyxy"][1], d["box_xyxy"][0]))
        return dict(n_objects=len(objs), objects=objs, confidence_threshold=conf, model=self.version)


PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
           (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
           (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 225, 25)]


def draw(image, objects, classes):
    """Annotated copy: one color per class, label = course ID + confidence."""
    idx = {v: k for k, v in classes.items()}
    img = image.copy()
    d = ImageDraw.Draw(img)
    w = max(2, round(max(img.size) / 250))
    for o in objects:
        color = PALETTE[idx[o["object_id"]] % len(PALETTE)]
        x0, y0, x1, y1 = o["box_xyxy"]
        text = f"{o['object_id']} {o['confidence']:.2f}"
        d.rectangle([x0, y0, x1, y1], outline=color, width=w)
        tb = d.textbbox((x0, y0), text)
        th, tw = tb[3] - tb[1] + 4, tb[2] - tb[0] + 6
        ty = y0 - th if y0 - th >= 0 else y0                    # no room above the box → put the label inside it
        d.rectangle([x0, ty, x0 + tw, ty + th], fill=color)
        dark = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2] < 140
        d.text((x0 + 3, ty + 1), text, fill="white" if dark else "black")
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["single", "multi"])
    ap.add_argument("image", type=Path)
    ap.add_argument("--models-dir", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    ap.add_argument("--conf", type=float, default=None, help="detector confidence threshold (default: from selection.json)")
    ap.add_argument("--save", type=Path, default=None, help="multi mode: save the annotated image here")
    args = ap.parse_args(argv)

    image = load_image(args.image)
    if args.mode == "single":
        out = Classifier(args.models_dir)(image)
    else:
        det = Detector(args.models_dir, conf=args.conf)
        out = det(image)
        if args.save:
            draw(image, out["objects"], det.classes).save(args.save)
            out["annotated_image"] = str(args.save)
    out = dict(image=str(args.image), mode=args.mode, **out)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
