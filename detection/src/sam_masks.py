"""Run SAM locally on every `sam` row of mask_review.csv, the same way the M2 pipeline does,
so the new masks can be checked before training.

Run (from the detection folder):
    E:/CodeTools/anaconda3/envs/ie7615-demo/python.exe src/sam_masks.py

Writes masks/fixed/<image_id>_sam.png (the same file the pipeline writes) and
masks/fixed/sam_boxes.json (box used for each file). A rerun only redoes images whose box changed.
Then check them in the review tool:
    E:/CodeTools/anaconda3/python.exe src/review_masks.py --sam-only
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_masks import MANIFEST, REVIEW, PROJECT, find_raw  # noqa: E402

FIX_DIR = PROJECT / "masks" / "fixed"
WEIGHTS = Path(os.environ.get("SAM_WEIGHTS", "E:/CodeTools/weights/sam_b.pt"))


def main():
    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
        man = {r["image_id"]: r for r in csv.DictReader(f)}
    with open(REVIEW, newline="", encoding="utf-8-sig") as f:
        todo = {r["image_id"].strip(): r["box"].strip() for r in csv.DictReader(f)
                if r["action"].strip().lower() == "sam"}
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    log_path = FIX_DIR / "sam_boxes.json"
    done = json.loads(log_path.read_text()) if log_path.exists() else {}
    work = [i for i, b in sorted(todo.items()) if done.get(i) != b or not (FIX_DIR / f"{i}_sam.png").exists()]
    print(f"{len(todo)} sam rows, {len(work)} to run")
    if not work:
        return
    from ultralytics import SAM
    sam = SAM(str(WEIGHTS))
    t0 = time.time()
    for k, i in enumerate(work, 1):
        box = [float(v) for v in todo[i].split()]
        path = find_raw(man[i]["src_path"])
        res = sam(str(path), bboxes=[box], verbose=False)[0]          # same call as the pipeline
        m = res.masks.data[0].cpu().numpy() > 0.5
        w, h = Image.open(path).size
        if m.shape != (h, w):
            m = np.asarray(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST)) > 0
        Image.fromarray(m.astype(np.uint8) * 255).save(FIX_DIR / f"{i}_sam.png")
        done[i] = todo[i]
        if k % 20 == 0 or k == len(work):
            log_path.write_text(json.dumps(done, indent=0))
            print(f"{k}/{len(work)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
