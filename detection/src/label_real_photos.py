"""Draw ground-truth boxes on the real multi-object photos (detection/data/real_photos/images).

Run:
    E:/CodeTools/anaconda3/python.exe detection/src/label_real_photos.py      (from the repository root)

Mouse: drag a box around one object, then press its class key
    1 = OBJ011 (mug)      2 = OBJ002 (paper roll)
Other objects (bottles, jars, boxes ...) are not boxed: they are distractors.
Keys: z = undo last box, -> / Space = save + next, <- = save + back, q / Esc = save + quit.
Labels go to detection/data/real_photos/labels/<name>.txt in YOLO format (class x_center y_center w h, 0-1),
with the class index of data/classes.json (OBJ011 = 6, OBJ002 = 2). An empty file means "no target object".
"""
import sys
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent.parent / "data" / "real_photos"
IMG_DIR, LBL_DIR = ROOT / "images", ROOT / "labels"
KEYS = {"1": (6, "OBJ011", "#00c000"), "2": (2, "OBJ002", "#ff8000")}
COLOR = {v[0]: v[2] for v in KEYS.values()}
NAME = {v[0]: v[1] for v in KEYS.values()}
VIEW = 900


class App:
    def __init__(self):
        self.files = sorted(IMG_DIR.glob("*.jpg"))
        LBL_DIR.mkdir(exist_ok=True)
        self.i = next((k for k, f in enumerate(self.files) if not (LBL_DIR / f"{f.stem}.txt").exists()), 0)
        if len(sys.argv) > 1:          # open a given photo, e.g.  real_box_tool.py IMG_2318
            self.i = next(k for k, f in enumerate(self.files) if f.stem == sys.argv[1])
        self.root = tk.Tk()
        global VIEW
        VIEW = min(VIEW, self.root.winfo_screenheight() - 170)   # whole photo must fit on screen
        self.canvas = tk.Canvas(self.root, width=VIEW, height=VIEW, bg="gray20")
        self.canvas.pack()
        self.status = tk.Label(self.root, font=("Arial", 12), anchor="w")
        self.status.pack(fill="x")
        tk.Label(self.root, anchor="w", justify="left", font=("Arial", 11), fg="gray25",
                 text="Drag a box, then press 1 = mug (OBJ011) or 2 = paper roll (OBJ002). Do not box other objects.\n"
                      "z = undo last box    Right / Space = save + next    Left = save + back    q = save + quit").pack(fill="x")
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.drag)
        for k in KEYS:
            self.root.bind(k, self.set_class)
        self.root.bind("z", lambda e: (self.boxes and self.boxes.pop(), self.draw()))
        for k in ("<Right>", "<space>"):
            self.root.bind(k, lambda e: self.go(1))
        self.root.bind("<Left>", lambda e: self.go(-1))
        for k in ("q", "<Escape>"):
            self.root.bind(k, lambda e: (self.save(), self.root.destroy()))
        self.load()
        self.root.mainloop()

    def load(self):
        f = self.files[self.i]
        self.img = Image.open(f)
        W, H = self.img.size
        self.s = VIEW / max(W, H)
        self.tk_img = ImageTk.PhotoImage(self.img.resize((int(W * self.s), int(H * self.s))))
        self.boxes, self.pending = [], None
        lf = LBL_DIR / f"{f.stem}.txt"
        if lf.exists():
            for line in lf.read_text().split("\n"):
                if line.strip():
                    c, x, y, w, h = line.split()
                    x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                    self.boxes.append((int(c), x - w / 2, y - h / 2, x + w / 2, y + h / 2))
        self.draw()

    def draw(self):
        c = self.canvas
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=self.tk_img)
        for cls, x0, y0, x1, y1 in self.boxes:
            s = self.s
            c.create_rectangle(x0 * s, y0 * s, x1 * s, y1 * s, outline=COLOR[cls], width=3)
            c.create_text(x0 * s + 4, y0 * s + 4, anchor="nw", text=NAME[cls], fill=COLOR[cls], font=("Arial", 12, "bold"))
        if self.pending:
            c.create_rectangle(*self.pending, outline="yellow", width=2, dash=(4, 2))
        counts = {n: sum(b[0] == k for b in self.boxes) for k, n in NAME.items()}
        msg = "  press 1 (OBJ011 mug) or 2 (OBJ002 roll) for the yellow box" if self.pending else ""
        self.status.config(text=f"{self.i + 1}/{len(self.files)}  {self.files[self.i].name}  {counts}{msg}")

    def press(self, e):
        self.start = (e.x, e.y)
        self.pending = None

    def drag(self, e):
        x0, y0 = self.start
        self.pending = (min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y))
        self.draw()

    def set_class(self, e):
        if not self.pending:
            return
        x0, y0, x1, y1 = (v / self.s for v in self.pending)
        W, H = self.img.size
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if x1 - x0 > 5 and y1 - y0 > 5:
            self.boxes.append((KEYS[e.char][0], x0, y0, x1, y1))
        self.pending = None
        self.draw()

    def save(self):
        W, H = self.img.size
        lines = [f"{c} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}"
                 for c, x0, y0, x1, y1 in self.boxes]
        (LBL_DIR / f"{self.files[self.i].stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    def go(self, d):
        self.save()
        self.i = max(0, min(len(self.files) - 1, self.i + d))
        self.load()


if __name__ == "__main__":
    App()
