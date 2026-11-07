"""M3 model interface: single-object image → object ID; multi-object image → IDs and locations of all objects.

Models (copied from the M1 / M2 run folders on Drive into one folder, default ../models, the repository-level models/ folder):
    classes.json          class index ↔ course ID (from M1, shared by both models)
    mobilenetv2.keras     M1 classifier selected on validation (preprocessing is built into the model)
    d1_yolov8n_best.pt    M2 detector
    selection.json        M2 selection record; its general_threshold is the default confidence threshold
Optional (M1 extension with an OTHERS class, classification/m1_others.ipynb); used instead of mobilenetv2.keras when both are present:
    mobilenetv2_others.keras    21-class classifier (the 20 course IDs + OTHERS = not one of the 20 objects)
    classes_with_others.json    its class map (0–19 as in classes.json, 20 = OTHERS)
    others_threshold.json       confidence threshold T chosen on validation: a course ID is returned only if its
                                probability is at least T, otherwise OTHERS (classification/src/others_threshold.py)

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
OTHERS_CLASSIFIER_FILE = "mobilenetv2_others.keras"     # optional 21-class classifier
OTHERS_CLASSES_FILE = "classes_with_others.json"
OTHERS_THRESHOLD_FILE = "others_threshold.json"        # optional; used only with the 21-class classifier
OTHERS_ID = "OTHERS"
DETECTOR_FILE = "d1_yolov8n_best.pt"
CLS_SIZE = 224          # M1 input size
DET_SIZE = 640          # M2 input size
NMS_IOU = 0.7           # same as M2
DEFAULT_CONF = 0.25     # used only if selection.json is missing


def load_classes(models_dir, file="classes.json"):
    m = json.loads((Path(models_dir) / file).read_text(encoding="utf-8"))
    return {int(k): v for k, v in m.items()}


def classifier_available(models_dir):
    d = Path(models_dir)
    return (d / CLASSIFIER_FILE).exists() or ((d / OTHERS_CLASSIFIER_FILE).exists() and (d / OTHERS_CLASSES_FILE).exists())


def load_image(path):
    """RGB image with the EXIF rotation applied (phone photos)."""
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


class Classifier:
    """M1 Keras classifier. Same preprocessing as M1: resize with padding to 224×224, RGB 0–255;
    the model-specific scaling is the model's own first layer."""

    def __init__(self, models_dir, file=None):
        import keras

        @keras.saving.register_keras_serializable(package="m1")
        class CaffePreprocess(keras.layers.Layer):     # needed only if the ResNet-50 model from M1 is loaded
            def call(self, x):
                x = keras.ops.flip(x, axis=-1)
                return x - keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype=x.dtype)

        import tensorflow as tf
        self.tf = tf
        d = Path(models_dir)
        if file is None:     # prefer the 21-class model (with OTHERS) when it and its class map are present
            file = OTHERS_CLASSIFIER_FILE if (d / OTHERS_CLASSIFIER_FILE).exists() and (d / OTHERS_CLASSES_FILE).exists() else CLASSIFIER_FILE
        self.model = keras.models.load_model(d / file, compile=False)
        self.classes = load_classes(d, OTHERS_CLASSES_FILE if file == OTHERS_CLASSIFIER_FILE else "classes.json")
        n_out = self.model.output_shape[-1]
        if n_out != len(self.classes):
            raise ValueError(f"{file} has {n_out} outputs but its class map has {len(self.classes)} classes.")
        self.has_others = OTHERS_ID in self.classes.values()
        thr = d / OTHERS_THRESHOLD_FILE
        self.threshold = json.loads(thr.read_text(encoding="utf-8"))["T"] if self.has_others and thr.exists() else None
        self.version = file

    def __call__(self, image, top_k=3, threshold=None):
        """threshold: override of the validation threshold for this call (21-class model only; 0 = no threshold)."""
        tf = self.tf
        if self.has_others and image.width != image.height:
            # The 21-class model saw its OTHERS photos as central square crops (no padding); do the same here,
            # so a padded phone photo does not show black borders the OTHERS class never had. Square images are unchanged.
            s = min(image.size)
            left, top_ = (image.width - s) // 2, (image.height - s) // 2
            image = image.crop((left, top_, left + s, top_ + s))
        x = tf.image.resize_with_pad(tf.cast(np.asarray(image), tf.float32), CLS_SIZE, CLS_SIZE)
        x = tf.clip_by_value(tf.round(x), 0, 255)[None]
        p = self.model.predict(x, verbose=0)[0]
        top = np.argsort(p)[::-1][:top_k]
        best = self.classes[int(top[0])]
        # Threshold (21-class model only): a course ID below T is not trusted, the answer becomes OTHERS
        t = (self.threshold if threshold is None else threshold) if self.has_others else None
        below = t is not None and best != OTHERS_ID and p[top[0]] < t
        out = dict(object_id=OTHERS_ID if below else best, confidence=round(float(p[top[0]]), 4),
                   in_course_classes=not below and best != OTHERS_ID,   # False = not one of the 20 objects
                   top_k=[dict(object_id=self.classes[int(k)], probability=round(float(p[k]), 4)) for k in top],
                   model=self.version)
        if t is not None:
            out.update(threshold=t, rejected_by_threshold=bool(below))
        return out


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
