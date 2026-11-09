# M2 on the university cluster (Slurm)

This folder runs the detection pipeline (`../src/m2_detection.py`, the same code as the M2 Colab notebook) as a batch job instead of in Colab. It was uploaded to the cluster as `~/hpc_m2`, the name used in the commands below. The folder layout copies the Drive layout, so no paths in the code change.

```
hpc_m2/                   (this folder, detection/cluster/ in the repository)
├── setup_env.sh          one-time setup on the login node
├── m2_job.sbatch         the Slurm job
├── code/m2_detection.py  copy of ../src/m2_detection.py (if missing, the job uses ../src)
├── run/yolov8n.pt        COCO-pretrained weights (job working directory)
├── data/raw/             ← you add: the 20 images_OBJxxx folders
├── IE7615_project/
│   ├── runs/m1_20261018_022316/data/   ← you add: classes.json, split_manifest.csv
│   └── masks/                          ← optional: raw_u2net.zip, mask_review.csv
├── logs/                 job logs (m2_<jobid>.out)
└── ycfg/                 Ultralytics settings and font
```

## 1. Fill in the data (on your laptop)

1. Drive → `data/student uploaded files` → download all `images_OBJxxx` folders → put the 20 folders directly in `data/raw/`. Drive may split the download into several zips; unzip all into the same place. A folder that loses its trailing space (`images_OBJ001 ` → `images_OBJ001`) is fine: the code finds it.
2. Copy `classes.json` and `split_manifest.csv` from `../../classification/results/m1/data/` (in git; the same files as on Drive `data/IE7615_project/runs/m1_20261018_022316/data/`) into `IE7615_project/runs/m1_20261018_022316/data/`. **Do not open them in Excel**: saving changes the checksum, and the job stops on purpose if they differ from M1. Copy `../data/masks/mask_review.csv` into `IE7615_project/masks/` (the review decisions of the reported runs).
3. Optional: if the Colab run already finished the masks, download `data/IE7615_project/masks/raw_u2net.zip` → put it in `IE7615_project/masks/`. The job then skips the mask step.
4. Copy `../src/m2_detection.py` to `code/`, then zip this folder as `hpc_m2.zip` with `hpc_m2/` as its top folder.

## 2. Upload and set up (once)

Upload the zip to your home or project directory (Open OnDemand file browser, or `scp`), then on the login node:

```bash
unzip hpc_m2.zip && cd hpc_m2
```

```bash
bash setup_env.sh
```

It ends with `setup OK`. If it says `conda not found`, run `module avail anaconda` and put the listed name into the `module load` line of `setup_env.sh` and `m2_job.sbatch`.

## 3. Submit and watch

```bash
sinfo -s
```

Pick a GPU partition from the list (often `gpu`; course accounts may have `courses-gpu`) and set `#SBATCH --partition=` in `m2_job.sbatch`. Then:

```bash
sbatch m2_job.sbatch
```

```bash
squeue -u $USER
```

```bash
tail -f logs/m2_<jobid>.out
```

Check in the log: `nvidia-smi` shows a GPU, and Ultralytics prints `CUDA:0 (...)` at the start of training.

Expected time: masks 15–30 min on 8 CPUs (seconds with the cache), generation 10–20 min, training 45–70 min.

## Experiments: mask fixes and training augmentation

Every run also writes two things used to improve the detector (both use train/val only, never test):

- **Single-object val photos** in `M2_results_for_report.md` and `selection.json`: the val split's single-object photos, where the object fills the frame. D1 has no object larger than about half the canvas, so this is the check for large objects.
- **Suspect partial masks**: `IE7615_project/masks/mask_review_suggested.csv` and `masks/review_sheets/u2net/suspect_partial_masks.jpg` (red: current mask, yellow: suggested SAM box). Look at the sheet, delete the rows you disagree with, and append the rest to `IE7615_project/masks/mask_review.csv`.

Order of runs (compare on D1-val and the single-object val photos only; D1-test is printed at the end of every run but is not used for choosing):

| Run | Command | Purpose |
|---|---|---|
| A | `sbatch m2_job.sbatch` | Current masks; produces the suspect list |
| B | `sbatch m2_job.sbatch` after updating `mask_review.csv` | New baseline with the fixed masks |
| C | `M2_TRAIN_AUG=large sbatch m2_job.sbatch` | Same data as B, training augmentation for large objects (`scale=0.9`, `mosaic=0.5`) |

The first `sam` decision downloads `sam_b.pt` (~375 MB) into `run/`.

## D1 vs D1+D3 comparison (D3 copy-paste)

`M2_ARM` switches the run to the comparison of proposal §7.2: the masks cut the objects out and paste them anywhere on the canvas (D3), and the two arms train on the same objects:

| Arm | Training data | Command |
|---|---|---|
| A | D1-train, 1,500 grid images | `M2_ARM=A sbatch m2_job.sbatch` |
| B | D1-train 750–1499 + 750 D3 images that reuse the objects of D1-train 0–749 | `M2_ARM=B sbatch m2_job.sbatch` |
| C | 1,500 D3 images that reuse the objects of all of D1-train (D3-only control) | `M2_ARM=C sbatch m2_job.sbatch` |

All arms: 60 epochs without early stopping; checkpoint and threshold on D1-val + hard-val (300 D3 composites from val photos); final test on D1-test and T-hard (300 harder D3 composites from test photos). For the three seeds per arm add `M2_TRAIN_SEED=1` and `M2_TRAIN_SEED=2` (the data stay the same; only training changes), e.g. `M2_ARM=B M2_TRAIN_SEED=1 sbatch m2_job.sbatch`.

Backgrounds: put indoor photos without any of the 20 objects into `IE7615_project/backgrounds/A/` (training and hard-val) and `IE7615_project/backgrounds/B/` (T-hard only). Without them the cut-outs are pasted on the gray canvas, and the run summary says so. Fix the masks first: D3 pastes exactly the mask pixels.

## 4. Get the results back

Results are in `IE7615_project/runs/m2_<time>/` on the cluster, same as on Drive. Download each run folder into `runs/` at the repository root (gitignored), for example:

```bash
ssh login.explorer.northeastern.edu "cd ~/hpc_m2/IE7615_project/runs && tar -cf - m2_<time>_armB_seed42" | tar -xf - -C runs
```

Then, from the repository root:

- `python detection/src/summarize_ablation.py` rebuilds `detection/results/ablation/` (comparison table and a copy of every run report).
- `detection/src/eval_real_photos.py` scores every comparison run on the real photos (`detection/results/real_photos/`).
- `mlops/tools/import_runs.py` adds the runs to MLflow; to change the model in use, register it and move the `champion` alias, then `mlops/tools/export_bundle.py --alias champion --to-models` refreshes `models/` for the demo.

**Updating the code on the cluster.** The job runs `~/hpc_m2/code/m2_detection.py`. After changing `detection/src/m2_detection.py`, copy it there and strip Windows line endings: `scp detection/src/m2_detection.py login.explorer.northeastern.edu:hpc_m2/code/` then `sed -i 's/$//' ~/hpc_m2/code/m2_detection.py` on the cluster.

**Mask review.** `detection/data/masks/mask_review.csv` keeps the `sam` rows with their boxes, so the job runs SAM itself (`sam_b.pt` is downloaded into `run/` on first use). The reported runs instead used masks made on the laptop: `detection/src/sam_masks.py` (same SAM call), checked with `review_masks.py --sam-only`, then `export_masks_for_cluster.py` turns the `sam` rows into `fixed` rows plus `fixed/<id>.png` in `detection/cluster/upload_masks/`, uploaded to `~/hpc_m2/IE7615_project/masks/`.
$//' ~/hpc_m2/code/m2_detection.py` on the cluster. The current mask review decisions are in `detection/data/masks/mask_review.csv`; upload them to `~/hpc_m2/IE7615_project/masks/` with the `fixed/` masks from `detection/src/export_masks_for_cluster.py`.

## If something fails

| Message in the log | Fix |
|---|---|
| `classes.json or split_manifest.csv differs from the M1 run` | Re-download the two files from Drive without opening them |
| `N source images are missing or changed since M1` | `data/raw/` must contain the 20 `images_OBJxxx` folders directly, not a `student uploaded files/` folder around them |
| `Invalid partition` / `QOSMaxGRESPerUser` | Change `--partition` to one listed by `sinfo -s` |
| `$'\r': command not found` | The script got Windows line endings; run `sed -i 's/\r$//' *.sh *.sbatch` |
| Job killed at the time limit | Raise `#SBATCH --time` |
