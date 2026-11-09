"""Mask review tool: look at each image with its mask outline and write decisions to mask_review.csv.

Run (from the detection folder):
    E:/CodeTools/anaconda3/python.exe tools/mask_review_tool.py OBJ000 OBJ001
    (no class names = all classes)

Screen:
    red outline   current mask (rembg)
    blue box      box the training labels get from that mask
    cyan box      box already saved for `sam` in mask_review.csv
    green         SAM result for that box (after tools/sam_preview.py has run)
    yellow box    box you are drawing

Check the SAM results:  tools/mask_review_tool.py --sam-only
    green right → / Space;  green wrong → draw a new box + Enter (then rerun sam_preview.py) or x

Mouse: drag a box around the whole object (without hands, without background).
Keys:
    Enter       save `sam` with the yellow box, go to next image
    b           save `sam` with the blue box (box is right, outline is not)
    o           save `ok`
    x           save `exclude`
    Delete / d  remove this image's decision (Backspace works too)
    → / Space   next image (no change)      ← previous image
    m           hide/show the mask outline
    q / Esc     quit

Every decision is written to the CSV at once. The first write also keeps a copy as
mask_review_backup.csv next to it.
"""
import argparse
import csv
import io
import json
import shutil
import tkinter as tk
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageTk

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "cluster"                       # detection/cluster: same layout as the cluster
PROJECT = ROOT / "IE7615_project"
MANIFEST = PROJECT / "runs" / "m1_20261018_022316" / "data" / "split_manifest.csv"
MASK_ZIP = PROJECT / "masks" / "raw_u2net.zip"
REVIEW = PROJECT / "masks" / "mask_review.csv"
RAW = ROOT / "data" / "raw"
SIZE = 224          # masks and boxes are in pixels of the 224x224 image
SCALE = 3           # shown 3x larger
FIELDS = ["image_id", "action", "box", "note", "auto_flags"]


def find_raw(rel):
    """Source image path; a folder may have lost its trailing space on Windows."""
    folder, name = rel.split("/", 1)
    for cand in (RAW / folder / name, RAW / folder.strip() / name):
        if cand.exists():
            return cand
    raise FileNotFoundError(RAW / rel)


def load_review():
    rows = {}
    if REVIEW.exists():
        with open(REVIEW, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                rows[r["image_id"].strip()] = {k: (r.get(k) or "") for k in FIELDS}
    return rows


def save_review(rows, backed_up=[False]):
    if REVIEW.exists() and not backed_up[0]:
        shutil.copy2(REVIEW, REVIEW.with_name("mask_review_backup.csv"))
        backed_up[0] = True
    with open(REVIEW, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])


FIX_DIR = PROJECT / "masks" / "fixed"


def load_sam(iid, box):
    """SAM mask from tools/sam_preview.py, only if it was made with this exact box."""
    log, png = FIX_DIR / "sam_boxes.json", FIX_DIR / f"{iid}_sam.png"
    if not (log.exists() and png.exists()):
        return None
    try:
        if json.loads(log.read_text()).get(iid) != box:
            return None
    except ValueError:
        return None
    return np.asarray(Image.open(png).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127


class App:
    def __init__(self, root, ids, man, masks):
        self.root, self.ids, self.man, self.masks = root, ids, man, masks
        self.rows = load_review()
        self.i, self.show_mask, self.drag, self.box = 0, True, None, None
        root.title("Mask review")
        self.info = tk.Label(root, font=("Consolas", 12), anchor="w", justify="left")
        self.info.pack(fill="x")
        self.cv = tk.Canvas(root, width=SIZE * SCALE, height=SIZE * SCALE, cursor="crosshair")
        self.cv.pack()
        tk.Label(root, font=("Consolas", 10), fg="gray30", justify="left",
                 text="drag = draw box   Enter = sam   b = sam with blue box   o = ok   x = exclude   Delete/d = remove   "
                      "→/Space = next   ← = back   m = mask on/off   q = quit").pack(fill="x")
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<B1-Motion>", self.on_drag)
        self.cv.bind("<ButtonRelease-1>", self.on_release)
        for key, fn in [("<Return>", self.save_sam), ("b", self.save_sam_blue), ("o", lambda e: self.decide("ok")),
                        ("x", lambda e: self.decide("exclude")), ("<Delete>", self.remove),
                        ("<BackSpace>", self.remove), ("d", self.remove),
                        ("<Right>", lambda e: self.move(1)), ("<space>", lambda e: self.move(1)),
                        ("<Left>", lambda e: self.move(-1)), ("m", self.toggle_mask),
                        ("q", lambda e: root.destroy()), ("<Escape>", lambda e: root.destroy())]:
            root.bind(key, fn)
        self.render()

    # ---------- drawing ----------
    def render(self):
        iid = self.ids[self.i]
        img = Image.open(find_raw(self.man[iid]["src_path"])).convert("RGB").resize((SIZE, SIZE))
        big = img.resize((SIZE * SCALE, SIZE * SCALE), Image.BICUBIC)
        d = ImageDraw.Draw(big)
        m = self.masks.get(iid)
        if m is not None and self.show_mask:
            mb = Image.fromarray(m.astype(np.uint8) * 255).resize(big.size, Image.NEAREST)
            edge = np.asarray(mb) > 0
            edge = edge & ~(np.asarray(mb.filter(ImageFilter.MinFilter(5))) > 0)
            arr = np.asarray(big).copy()
            arr[edge] = (255, 0, 0)
            big = Image.fromarray(arr)
            d = ImageDraw.Draw(big)
            ys, xs = np.where(m)
            if len(xs):
                d.rectangle([xs.min() * SCALE, ys.min() * SCALE, (xs.max() + 1) * SCALE, (ys.max() + 1) * SCALE],
                            outline=(40, 120, 255), width=2)
        row = self.rows.get(iid, {})
        sam_note = ""
        if row.get("action", "").strip().lower() == "sam" and row.get("box", "").strip():
            x0, y0, x1, y1 = (float(v) * SCALE for v in row["box"].split())
            d.rectangle([x0, y0, x1, y1], outline=(0, 230, 230), width=2)
            sm = load_sam(iid, row["box"].strip())
            if sm is None:
                sam_note = "   (SAM not run for this box yet)"
            else:                                            # SAM result: green outline + green box
                sb = Image.fromarray(sm.astype(np.uint8) * 255).resize(big.size, Image.NEAREST)
                edge = (np.asarray(sb) > 0) & ~(np.asarray(sb.filter(ImageFilter.MinFilter(5))) > 0)
                arr = np.asarray(big).copy()
                arr[edge] = (0, 255, 0)
                big = Image.fromarray(arr)
                d = ImageDraw.Draw(big)
                ys, xs = np.where(sm)
                if len(xs):
                    d.rectangle([xs.min() * SCALE, ys.min() * SCALE, (xs.max() + 1) * SCALE, (ys.max() + 1) * SCALE],
                                outline=(0, 160, 0), width=2)
                sam_note = "   green = SAM result"
        self.photo = ImageTk.PhotoImage(big)
        self.cv.delete("all")
        self.cv.create_image(0, 0, anchor="nw", image=self.photo)
        self.box, self.rect = None, None
        act = row.get("action", "").strip() or "(none)"
        extra = f"  box={row['box']}" if row.get("box") else ""
        flags = f"  auto_flags={row['auto_flags']}" if row.get("auto_flags") else ""
        self.info.config(text=f"{iid}  [{self.man[iid]['split']}]   {self.i + 1}/{len(self.ids)}   "
                              f"decision: {act}{extra}{flags}{sam_note}")

    def on_press(self, e):
        self.drag = (e.x, e.y)
        if self.rect:
            self.cv.delete(self.rect)
        self.rect = self.cv.create_rectangle(e.x, e.y, e.x, e.y, outline="yellow", width=2)

    def on_drag(self, e):
        if self.drag:
            self.cv.coords(self.rect, self.drag[0], self.drag[1], e.x, e.y)

    def on_release(self, e):
        if not self.drag:
            return
        (ax, ay), self.drag = self.drag, None
        x0, x1 = sorted((ax, e.x))
        y0, y1 = sorted((ay, e.y))
        clip = lambda v: max(0, min(SIZE, round(v / SCALE)))
        b = [clip(x0), clip(y0), clip(x1), clip(y1)]
        if b[2] - b[0] < 4 or b[3] - b[1] < 4:       # a click, not a box
            self.cv.delete(self.rect)
            self.rect, self.box = None, None
            return
        self.box = b
        self.info.config(text=self.info.cget("text").split("   new box")[0] + f"   new box {b[0]} {b[1]} {b[2]} {b[3]} (Enter = save sam)")

    # ---------- decisions ----------
    def write(self, iid, action, box=""):
        old = self.rows.get(iid, {})
        self.rows[iid] = dict(image_id=iid, action=action, box=box,
                              note=old.get("note", "") or "review tool", auto_flags=old.get("auto_flags", ""))
        save_review(self.rows)

    def save_sam(self, e=None):
        if not self.box:
            self.info.config(text=self.info.cget("text") + "   ← draw a box first")
            return
        self.write(self.ids[self.i], "sam", " ".join(map(str, self.box)))
        self.move(1)

    def save_sam_blue(self, e=None):
        """`sam` with the blue box (the current mask's box), for masks whose box is right but whose outline is not."""
        m = self.masks.get(self.ids[self.i])
        if m is None or not m.any():
            self.info.config(text=self.info.cget("text") + "   ← no mask box here, draw a box")
            return
        ys, xs = np.where(m)
        self.write(self.ids[self.i], "sam", f"{xs.min()} {ys.min()} {xs.max() + 1} {ys.max() + 1}")
        self.move(1)

    def decide(self, action):
        self.write(self.ids[self.i], action)
        self.move(1)

    def remove(self, e=None):
        iid = self.ids[self.i]
        had = iid in self.rows
        if had:
            del self.rows[iid]
            save_review(self.rows)
        self.render()
        self.info.config(text=self.info.cget("text") + ("   ← decision removed" if had else "   ← nothing saved for this image"))

    def move(self, step):
        self.i = max(0, min(len(self.ids) - 1, self.i + step))
        self.render()

    def toggle_mask(self, e=None):
        self.show_mask = not self.show_mask
        self.render()


def main():
    ap = argparse.ArgumentParser(description="Draw SAM boxes and write mask_review.csv")
    ap.add_argument("classes", nargs="*", help="e.g. OBJ000 OBJ001 (default: all classes)")
    ap.add_argument("--start", default="", help="image id to start at, e.g. OBJ000_0030")
    ap.add_argument("--sam-only", action="store_true", help="show only the `sam` rows, with the SAM result in green")
    a = ap.parse_args()
    with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
        man = {r["image_id"]: r for r in csv.DictReader(f)}
    wanted = {c.upper() for c in a.classes}
    ids = sorted(i for i, r in man.items() if not wanted or r["class_id"] in wanted)
    if a.sam_only:
        sam_rows = {k for k, r in load_review().items() if r["action"].strip().lower() == "sam"}
        ids = [i for i in ids if i in sam_rows]
    if not ids:
        raise SystemExit(f"No images for {sorted(wanted)}")
    z = zipfile.ZipFile(MASK_ZIP)
    names = set(z.namelist())
    masks = {i: np.asarray(Image.open(io.BytesIO(z.read(f"{i}.png"))).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127
             for i in ids if f"{i}.png" in names}
    print(f"{len(ids)} images, {len(masks)} masks; writing to {REVIEW}")
    root = tk.Tk()
    app = App(root, ids, man, masks)
    if a.start in ids:
        app.i = ids.index(a.start)
        app.render()
    root.mainloop()


if __name__ == "__main__":
    main()
