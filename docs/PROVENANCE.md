# Provenance of the reported detection runs

The nine runs of the D1 / D1+D3 / D3 comparison (`detection/results/ablation/`) record `git_commit = "not recorded"`:
the cluster job ran a copy of the script in `~/hpc_m2/code/`, a folder without `.git`, so `git_info()` in
`detection/src/m2_detection.py` could not read the commit. This file links each run to the exact code, environment and
inputs, from the files that are still on the cluster. Checked on 2026-11-10 over ssh (read only, nothing on the cluster was changed).

Hashes are sha256 of the files as they are. Line endings: the cluster copies have LF only, and the git versions below give
the same hash after converting CRLF to LF (`sed 's/\r$//' | sha256sum`).

## Runs

| Run folder | Arm | Seed | Slurm job | GPU | Code |
|---|---|---|---|---|---|
| `m2_20261108_054413_armA_seed42` | A (D1) | 42 | 11727585 | Tesla V100-SXM2-32GB | `f2460e6` (see below) |
| `m2_20261108_054413_armB_seed42` | B (D1+D3) | 42 | 11727586 | Tesla V100-SXM2-32GB | `f2460e6` |
| `m2_20261108_135032_armA_seed1` | A | 1 | 11734566 | Tesla V100-SXM2-32GB | `f2460e6` |
| `m2_20261108_135138_armB_seed1` | B | 1 | 11734567 | Tesla **P100**-PCIE-12GB | `f2460e6` |
| `m2_20261108_141110_armA_seed2` | A | 2 | 11734568 | Tesla V100-SXM2-32GB | `f2460e6` |
| `m2_20261108_143042_armB_seed2` | B | 2 | 11734569 | Tesla V100-SXM2-32GB | `f2460e6` |
| `m2_20261108_183715_armC_seed42` | C (D3) | 42 | 11739661 | Tesla V100-SXM2-32GB | `66ea52b` |
| `m2_20261108_183715_armC_seed1` | C | 1 | 11739662 | Tesla V100-SXM2-32GB | `66ea52b` |
| `m2_20261108_185747_armC_seed2` | C | 2 | 11739663 | Tesla V100-SXM2-32GB | `66ea52b` |

Job ↔ run: the first line of each `~/hpc_m2/logs/m2_<job>.out` (GPU check) and the run folder it writes.
Arm B seed 1 ran on a P100, all other runs on a V100; GPU type is one more source of run-to-run variation besides the seed.

## Code

| Cluster file | sha256 | Same as |
|---|---|---|
| `code/m2_detection.py` (uploaded 2026-11-08 18:35 EST) | `c93f6524cdb8705f8af56c63b80f767953ed35fbce6d6fd1349d79e0754050de` | `detection/src/m2_detection.py` at `66ea52b` (unchanged up to `68db2e4`) |
| `code/m2_detection_before_armC.py` (copy kept when arm C was uploaded) | `d6f6a7a8f2830eb644fe3ef32f63ec3a7e02e6f6617f9e7d655a908ee73cfbbf` | `detection/src/m2_detection.py` at `f2460e6` |
| `code/m2_detection_before_d3.py` (copy kept when D3 was uploaded, 2026-11-08 05:41) | `00c1e2244874d27955347143daaa5b207af9301bf234d8189641b60cb5782d2f` | `c7bd8aa` (= `8cc6839`, the code before the D3 work) |

- **Arm C (verified):** the three arm C jobs started after the 18:35 upload, and the file has not changed since; it equals `66ea52b`.
- **Arms A and B (inferred, not proven per job):** the D3 code was uploaded at 05:42, two minutes before the first two jobs;
  the copy kept at 18:35, just before arm C replaced it, equals `f2460e6`, and git has no other version between `f2460e6` and `66ea52b`.
  So the six A/B jobs most likely ran `f2460e6`. Nothing records the file at the moment each job started.
- `66ea52b` only adds arm C on top of `f2460e6` (arm label, `replaced` records, messages); arms A and B behave the same in both.

The job script on the cluster, `m2_job.sbatch`, is identical to the repository version (sha256 `aa4ab198a14fc0ee…`).
`setup_env.sh` differs only in comments.

## Environment

- Conda env `ie7615-m2`, Python 3.11.18; full package list: [`detection/cluster/env_ie7615-m2_freeze.txt`](../detection/cluster/env_ie7615-m2_freeze.txt).
- Main packages: ultralytics 8.4.202, torch 2.5.1+cu121, torchvision 0.20.1+cu121, rembg 2.0.86, onnxruntime 1.31.1,
  numpy 2.4.6, opencv-python 5.0.0.93, pillow 13.0.0, pandas 3.1.0.
- No package in the env was installed or changed after 2026-11-03 22:56 EST (newest `*.dist-info` in site-packages), before all nine runs.
  Every job log prints `Ultralytics 8.4.202`, the same version as `environment.ultralytics` in each run record.
- `detection/cluster/setup_env.sh` installs `ultralytics>=8.3,<9` and unpinned helpers, so a new env today can differ; install from the freeze file to match.

## Inputs

| Input | sha256 | Where |
|---|---|---|
| `IE7615_project/masks/mask_review.csv` (uploaded 2026-11-08 05:42, unchanged since) | `d2bb47b9fe45776629a65a9c52b3794e0ded2de39408bdb093d2b32c2c0939c7` | `detection/data/masks/mask_review.csv` at `a5408c1` (267 `fixed` rows) |
| `IE7615_project/masks/fixed/*.png` (267 SAM masks) | per-file hashes identical to the local `detection/cluster/upload_masks/fixed/` | not in git; `68db2e4` stores the same rows as `sam` + box, and M2 regenerates identical masks (IoU 1.0 for all 267, CPU check 2026-11-09) |
| `IE7615_project/masks/raw_u2net.zip` (rembg mask cache, 2026-11-03) | `b5dc8aa4fb167194e5430fe1948a88eadfe9c1a954864ae298a4d1aae6346ddf` | not in git; M2 regenerates it with rembg when missing |
| `run/yolov8n.pt` (COCO-pretrained start weights) | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | downloaded by Ultralytics (`setup_env.sh`) |
| M1 split `classes.json`, `split_manifest.csv` | checked by M2 against `511d30a1…` / `ce9172b7…` | `classification/results/m1/data/` |
| Course images (20 `images_OBJxxx` folders) | M2 checks every source image against the M1 manifest | not in git (course data) |

## For new runs

Run the job from a git checkout instead of a copied script, so `git_commit` and `git_dirty` are recorded
(`m2_job.sbatch` falls back to `../src/m2_detection.py` when `code/` is missing, and then reads the M1 split from git);
steps in [`detection/cluster/README.md`](../detection/cluster/README.md#recommended-for-new-runs-a-git-checkout).
`setup_env.sh` installs the versions of `env_ie7615-m2_freeze.txt`.
