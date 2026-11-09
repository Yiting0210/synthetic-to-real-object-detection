"""Make the mask-review upload for the cluster from the locally checked SAM masks.

Run (from the detection folder), after src/sam_masks.py and the --sam-only check:
    E:/CodeTools/anaconda3/python.exe src/export_masks_for_cluster.py

Writes cluster/upload_masks/ with
    mask_review.csv   the local file, every `sam` row turned into `fixed` (the box stays in its column)
    fixed/<id>.png    the SAM mask that was checked, under the name the pipeline reads for `fixed`
Upload both into ~/hpc_m2/IE7615_project/masks/ on the cluster. The run then uses exactly the
masks that were checked and needs no SAM weights.
"""
import csv
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from review_masks import FIELDS, FIX_DIR, ROOT, load_review, load_sam  # noqa: E402

OUT = ROOT / "upload_masks"


def main():
    rows = load_review()
    sam_ids = [k for k, r in rows.items() if r["action"].strip().lower() == "sam"]
    missing = [k for k in sam_ids if load_sam(k, rows[k]["box"].strip()) is None]
    if missing:
        raise SystemExit(f"{len(missing)} sam rows have no SAM mask for their current box "
                         f"(run src/sam_masks.py again): {', '.join(missing[:10])}")
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "fixed").mkdir(parents=True)
    for k in sam_ids:
        shutil.copy2(FIX_DIR / f"{k}_sam.png", OUT / "fixed" / f"{k}.png")
        rows[k] = dict(rows[k], action="fixed", note=(rows[k]["note"] + " (SAM, checked)").strip())
    with open(OUT / "mask_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    acts = {}
    for r in rows.values():
        a = r["action"].strip() or "(blank)"
        acts[a] = acts.get(a, 0) + 1
    print(f"Wrote {OUT}: {len(rows)} rows {acts}, {len(sam_ids)} masks in fixed/")


if __name__ == "__main__":
    main()
