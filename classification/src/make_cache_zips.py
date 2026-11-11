"""Pack the data for the OTHERS training into two zips, so Colab reads two files from Drive instead of thousands.

    python classification/src/make_cache_zips.py          (from the repository root, on the laptop)

Writes to detection/cluster/IE7615_project/cache/ (not in git); upload both files to Drive  MyDrive/data/IE7615_project/cache/.
  course_images_m1.zip   the 2,022 course images of the M1 split, each checked against the md5 in split_manifest.csv
  others.zip             classification/others/images_OTHERS/ and unseen_test/
Run it again after adding or changing photos.
"""
import csv, hashlib, os, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("IE7615_RAW_DIR", ROOT / "detection" / "cluster" / "data" / "raw"))   # the 20 images_OBJxxx folders
MANIFEST = ROOT / "classification" / "results" / "m1" / "data" / "split_manifest.csv"      # the M1 split (in git)
OTHERS = ROOT / "classification" / "others"
OUT = ROOT / "detection" / "cluster" / "IE7615_project" / "cache"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff"}
OUT.mkdir(parents=True, exist_ok=True)

# 1) course images (JPEGs are already compressed: store only)
folders = {p.name.strip(): p for p in RAW.iterdir() if p.is_dir()}       # folder names may carry trailing spaces
rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
bad = []
with zipfile.ZipFile(OUT / "course_images_m1.zip", "w", zipfile.ZIP_STORED) as zf:
    for r in rows:
        folder, name = r["src_path"].split("/", 1)
        src = folders.get(folder.strip(), RAW / folder) / name
        if not src.exists() or hashlib.md5(src.read_bytes()).hexdigest() != r["md5"]:
            bad.append(r["src_path"])
            continue
        zf.write(src, arcname=f"{folder.strip()}/{name}")
print(f"course_images_m1.zip: {len(rows) - len(bad)} images, {(OUT / 'course_images_m1.zip').stat().st_size / 1e6:.1f} MB"
      + (f"; {len(bad)} missing or changed: {bad[:5]}" if bad else ""))

# 2) OTHERS photos and unseen-object photos
counts = {}
with zipfile.ZipFile(OUT / "others.zip", "w", zipfile.ZIP_STORED) as zf:
    for sub in ["images_OTHERS", "unseen_test"]:
        files = sorted(f for f in (OTHERS / sub).glob("*") if f.is_file() and f.suffix.lower() in IMG_EXT)
        for f in files:
            zf.write(f, arcname=f"{sub}/{f.name}")
        counts[sub] = len(files)
print(f"others.zip: images_OTHERS {counts['images_OTHERS']}, unseen_test {counts['unseen_test']}, "
      f"{(OUT / 'others.zip').stat().st_size / 1e6:.1f} MB")
if counts["unseen_test"] == 0:
    print("note: unseen_test is empty; add the unseen-object photos and run this again before uploading")
print("upload both files to Drive:  MyDrive/data/IE7615_project/cache/")
