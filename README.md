# Synthetic-to-Real Object Detection

**Lab-to-Real Object ID · IE 7615 Discriminative Deep Learning Project**

Can YOLOv8 trained on synthetic composites generalize to real photographs with clutter, occlusion, and scale changes?

This project compares grid concatenation with mask-based copy-paste training data, evaluates synthetic-to-real generalization, and adds ML-specific engineering: experiment tracking and a model registry with rollback. A separate Keras/TensorFlow benchmark supports single-object classification.

> **Status (November 11, 2026):** course milestones M1–M3 are done. Since then:
> - masks of the seven weakest classes reviewed;
> - the copy-paste (D3) comparison run with three seeds per arm (D1 only, D1+D3, D3 only), and a real multi-object photo test;
> - MLflow tracking with a re-evaluation check of every registered version, report tables exported from MLflow, a model registry with rollback, and a FastAPI/Docker service with a startup readiness check.
>
> Postponed: the CPU runtime comparison (ONNX / OpenVINO export) and the `/classify` endpoint.

## Repository contents

| Folder | What it contains |
| --- | --- |
| `classification/` | `notebooks/` (M1 classifiers, OTHERS class; executed on Colab T4), `src/` (OTHERS training, rejection methods), `results/` (`m1/`, `others/`, `openset/`) |
| `detection/src/` | `m2_detection.py` (masks, D1 grid and D3 copy-paste composites, YOLOv8n training, evaluation; Colab notebook or cluster script), mask review tools (`review_masks.py`, `sam_masks.py`, `export_masks_for_cluster.py`), real-photo tools (`label_real_photos.py`, `eval_real_photos.py`) and `summarize_ablation.py` |
| `detection/data/` | `masks/mask_review.csv` (mask review decisions; `sam` rows keep their boxes) and `real_photos/` (29 phone photos with hand-drawn boxes) |
| `detection/results/` | `ablation/` (arm A/B/C × 3 seeds, generated from the run reports), `real_photos/` (all nine models on the real photos), `archive/` (earlier single run) |
| `detection/cluster/` | Slurm job and environment setup for the Explorer cluster |
| `models/` | Class maps and the detector selection record exported from the registry `champion`; trained weights go here (not in git), see `models/README.md` |
| `demo/` | `predict.py` (single image → ID; multi-object image → IDs and boxes) and `demo_app.py` (local Gradio demo) with example images; see `demo/README.md` |
| `mlops/` | `tools/` (MLflow import, registration, bundle export, rollback demo), `logs/` (run-ID map, rollback log); the MLflow store `store/` is local only |
| `service/` | FastAPI inference service (`POST /detect`, `GET /health`), pytest tests and a CPU-only Dockerfile |
| `docs/` | `PROVENANCE.md`: commit, Slurm job, GPU, environment and input checksums of each reported detection run |
| `tools/` | `percent2ipynb.py`: turns a percent-format script (`# %%` cells) into a notebook for Colab |

Local only (gitignored): `runs/` (run folders downloaded from the cluster), `mlops/store/` (MLflow database and artifacts), `model_bundles/` (exported bundles). Results reach git as generated summaries; every run is in MLflow; the model in use is the registry alias `champion`.

Model versions are marked with git tags per component, named `<component>-v<major>.<minor>` (each tag is a snapshot of the whole repository):

- **minor + 1** (v1.0 → v1.1): same method, only the data or settings changed;
- **major + 1** (v1 → v2): the method changed (for detection: the way the training data is generated).

| Tag | What changed |
|---|---|
| [`classification-v1.0`](../../tree/classification-v1.0) | 20-class MobileNetV2 (M1, single-object identification) |
| [`classification-v1.1`](../../tree/classification-v1.1) | Same model and training recipe; data adds an `OTHERS` class (100 photos), setting adds a rejection threshold chosen on validation |
| [`detection-v1.0`](../../tree/detection-v1.0) | YOLOv8n on grid composites, as submitted for M2 (Colab run, D1-test mAP@0.5 0.908), with the earlier `m1/`, `m2/`, `m3/` layout |
| [`detection-v1.1`](../../tree/detection-v1.1) | Same grid composites; masks reviewed, retrained on the cluster (D1-test mAP@0.5 0.928) |
| [`detection-v2.0`](../../tree/detection-v2.0) | Method change: trained on D1 grid + D3 copy-paste composites (arm B of the A/B/C ablation, seed 42; registry `lab2real-detector` v2, alias `champion`). D1-test mAP@0.5 0.962, T-hard 0.940, real photos 0.906 (3-seed mean 0.864 ± 0.056). Includes the MLflow tracking, registry and FastAPI/Docker service |

The tag `detection-v1.0` also holds the executed Colab notebook, report and figures of the submitted M2 run. Data, run outputs and the mask cache are not in the repository; the notebooks read the course dataset from Google Drive. The trained weights used by the demo (`mobilenetv2.keras`, `d1_yolov8n_best.pt`) are not published yet; they will be released once the models are final.

## Project scope

- **Dataset:** 20 shared course object IDs, 2,022 single-object images (224×224, 100–114 per class), audited in M1. Source-image collection contributes OBJ011.
- **Classification:** implement and compare at least three distinct architectures using Keras/TensorFlow: custom CNN, MobileNetV2, ResNet-50 and EfficientNet-B0. Hyperparameter variants do not count as additional architectures. Use each model's required preprocessing and justify the best model using validation performance and accuracy/speed/size trade-offs.
- **Detection:** fine-tune Ultralytics YOLOv8n (PyTorch), generating at least 100 multi-object images by concatenating randomly selected single-object images.
- **Controlled experiment:** compare 1,500 grid training images against 750 of those grid images plus 750 copy-paste images, controlling object counts and training budget. Target three training seeds per arm; a single-seed result is reported as preliminary.
- **Real-photo evaluation:** evaluate OBJ011 in clutter using session-separated validation/test photographs, including no-target images. This is a one-object case study, not evidence of real-world performance across all 20 classes.

## Evaluation and reproducibility

1. Group near-duplicate source images before a fixed train/validation/test split; generate composites only from their corresponding source split.
2. Keep one shared class map between continuous model indices and course IDs such as `OBJ011`.
3. Review segmentation masks and derive visible-object boxes after scaling, rotation, clipping and occlusion.
4. Select architectures, checkpoints and thresholds on validation data; report frozen choices on held-out data.
5. Report classification accuracy/macro-F1 and confusion matrices; detection mAP@0.5 and mAP@0.5:0.95; and real-case recall, matched-box IoU and false positives per image.
6. Record configurations, seeds, code revisions, dataset checksums and model artifacts with MLflow (portable JSON/CSV records as a fallback). Store per-composite generation records for sample regeneration.

**Compute plan:** M1 and the submitted M2 ran on Google Colab (Tesla T4); the retrained M2 ran on the Explorer cluster (Tesla P100, Slurm). The multi-seed detection comparison (3 arms × 3 seeds) also ran there as Slurm jobs (`detection/cluster/`; Tesla V100 / P100). Persist run outputs on storage appropriate to each environment. CPU deployment benchmarks name the exact machine and compare runtimes under matched conditions.

## Engineering work

The engineering scope is limited to ML-specific practices; data pipelines, monitoring and load testing are not part of this project.

| Area | Implementation | Evidence | Status |
| --- | --- | --- | --- |
| Experiment tracking | MLflow with a local SQLite backend; Colab and HPC run records (JSON/CSV) imported by script | Every reported number traces to a run with its configuration, data checksums and model; registered versions re-evaluated (`mlops/logs/reevaluation.md`); report tables exported from MLflow | Done. Git commits of the training runs were not recorded and are not back-filled |
| Model registry and rollback | MLflow Model Registry; `champion` / `previous` aliases; exported model bundles with checksums | A recorded rollback to the previous version, confirmed through the service response (`mlops/logs/rollback_demo.log`) | Done |
| Minimal inference service | FastAPI `POST /detect`, Docker image with the model bundle mounted read-only, pytest | Tested API calls that report the model version; readiness check on a development image | Done; `POST /classify` postponed |
| Detector CPU inference | PyTorch FP32, ONNX Runtime FP32, OpenVINO FP32/INT8 | Accuracy, mean/p95 latency, memory, artifact size and load time under matched conditions | Postponed |

The four-runtime benchmark applies to the detector. The classifier's native correctness baseline is Keras/TensorFlow; any classifier export is checked against that baseline. No speed or accuracy number is published for an export that has not passed the class-map, preprocessing and coordinate checks.

## Delivery plan

| Phase | Focus |
| --- | --- |
| M1 · Oct 17 | At least three distinct classifiers, comparison and justified selection; PDF report within five pages including the front page |
| M2 · Oct 27 | Synthetic multi-object dataset, YOLOv8 training and ID/location outputs |
| M3 · Nov 3 | Completed detection model and live single/multi-object demonstration |
| M4 · Nov 10 | Final report and models |
| Nov 8–11 (done) | MLflow import, Model Registry with a rollback demonstration, re-evaluation check, tables exported from MLflow, FastAPI + Docker |
| Later (postponed) | Detector runtime comparison (PyTorch FP32 vs ONNX Runtime / OpenVINO, INT8) and `POST /classify` |
| December (optional) | Webcam demo and a single-object data-drift case study |

Course deliverables take priority. If dates move, document the revised schedule; do not present pending features as completed.

## Results and examples

Measured so far (single runs, seed 42):

| Task | Result |
| --- | --- |
| M1 classification (303 test images, 20 classes; Colab T4) | MobileNetV2 selected on validation: test accuracy 0.993, macro-F1 0.993 (2.28 M parameters); EfficientNet-B0 1.000, ResNet-50 0.997, custom CNN 0.891 |
| M2 detection, submitted (Colab T4; 1,939 usable source images) | YOLOv8n on 300 grid-composite test images: mAP@0.5 0.908, mAP@0.5:0.95 0.738, macro-F1 0.890 (threshold 0.50) |
| M2 detection, retrained (cluster P100; 1,987 usable source images after mask review) | YOLOv8n on 300 grid-composite test images (1,154 objects): mAP@0.5 0.928, mAP@0.5:0.95 0.743, macro-F1 0.900 (threshold 0.55) |

The two M2 test sets were generated from different usable source images, so the two rows are not on identical test images. The detection test set is built like the training data, so it measures in-distribution performance only.

### Copy-paste comparison (D1 vs D1+D3 vs D3, three seeds)

Mean over training seeds 42, 1 and 2; full table: [`detection/results/ablation/`](detection/results/ablation/README.md), real photos: [`detection/results/real_photos/`](detection/results/real_photos/README.md).

| Test | arm A (D1) | arm B (D1+D3) | arm C (D3) |
|---|---|---|---|
| D1-test mAP@0.5 (grid composites) | 0.954 | 0.962 | 0.623 |
| T-hard mAP@0.5 (copy-paste composites, same mechanism as D3) | 0.606 | 0.938 | 0.945 |
| Single-object test photos correct (of 303) | 201 | 240 | 169 |
| Real multi-object photos, mAP@0.5 (2 classes, 29 photos) | 0.817 | 0.864 | 0.713 |
| Real multi-object photos, precision | 0.555 | 0.551 | 0.279 |

- Mixing D1 and D3 (arm B) is the best arm on every test; D3 alone matches it only on the D3-style synthetic sets.
- T-hard shares the copy-paste mechanism with arms B and C, so its large gain overstates the real-photo effect.
- On real photos, B finds the occluded paper roll more often (recall 0.76 vs 0.58) but has as many false alarms as A, mostly distractors reported as look-alike course objects.
- Limits: real photos cover only OBJ011 and OBJ002; D3 uses a gray canvas; only 7 of 20 classes had their masks reviewed.

### M1 conclusions (single-object classification)

1. **Transfer learning is what matters; the three pretrained models are tied.**
   - From-scratch CNN: 0.891 test accuracy; ImageNet-pretrained models: 0.993–1.000.
   - On validation (303 images), ResNet-50, EfficientNet-B0 and MobileNetV2 lie within each other's 95% Wilson intervals: the difference is not measurable at this sample size.
   - MobileNetV2 was chosen by the rule fixed before testing (fewest parameters among the competitive models): 2.28 M parameters, 10× fewer than ResNet-50.
   - Test accuracy 0.993: 2 errors in 303 (OBJ123→OBJ777, OBJ004→OBJ007).
2. **The accuracy is in-distribution.**
   - Each class was photographed by one person in one setting; some classes carry framing cues (borders, plain backgrounds).
   - The grouped split removes near-duplicate leakage, but test images come from the same sessions as training.
   - So ~99% does not show that the model works on new photos of these objects.
3. **The 20-class model cannot say "none of these".**
   - It always returns one of the 20 IDs.
   - Adding an `OTHERS` class (100 photos of 31 everyday kinds, same recipe) keeps known accuracy (302/303 on test).
   - It rejects 12 of 15 OTHERS test photos, but only 15 of 36 photos of 18 kinds never seen in training.
   - A confidence threshold chosen on validation (T = 0.50) raises this to 21/36, with 1 more known image rejected; this is what the demo uses.
4. **The rejection rule matters less than the training data.**
   - Five rules compared, each threshold set on validation (at most 1% of known images rejected); see the table below.
   - All reject 15–25 of 36 unseen photos, AUROC 0.980–0.991, with overlapping intervals.
   - Objects that look like a course object (e.g. sunglasses) are accepted by all five: the next improvement would come from look-alike negative examples, not another rule.
   - The unseen set was used to compare the rules, so the best number is optimistic; an unbiased figure needs a fresh unseen set.

| Model | Rejection rule | Known test correct | Unseen rejected (of 36) | AUROC |
|---|---|---|---|---|
| 20 classes | Softmax threshold | 300/303 | 23 | 0.984 |
| 20 classes | Prototype (cosine) distance | 300/303 | 24 | 0.987 |
| 21 classes | `OTHERS` only | 302/303 | 15 | 0.980 |
| 21 classes | `OTHERS` + softmax threshold | 301/303 | 21 | 0.985 |
| 21 classes | `OTHERS` + prototype distance | 301/303 | 25 | 0.991 |

Details: `classification/results/openset/`.

## Data

The course images are not in the repository: they were taken by the course participants and are shared through the course Google Drive. What is in git is enough to check that the same images are used:

| What | Where | In git |
|---|---|---|
| 20 folders `images_OBJxxx`, 2,022 single-object photos | course Drive (`data/student uploaded files`); a local zip of them, if you have one, goes to `detection/cluster/IE7615_project/cache/course_images_m1.zip` (gitignored) | no |
| Class map and train/val/test split with the md5 of every source photo | `classification/results/m1/data/classes.json`, `split_manifest.csv` | yes |
| Mask review decisions (ok / sam + box / exclude) | `detection/data/masks/mask_review.csv` | yes |
| Real multi-object test photos (own photos, EXIF removed) and hand-drawn boxes | `detection/data/real_photos/` | yes |
| A few demo images | `demo/examples/` | yes |

- M2 checks the two M1 files against fixed checksums (`classes.json` `511d30a1…`, `split_manifest.csv` `ce9172b7…`) and every source photo against its md5 in the manifest; it stops if anything differs. Use the files from git rather than re-running M1, which may write a different file.
- Masks: M2 computes the rembg masks itself (cached in `IE7615_project/masks/raw_u2net.zip`). For the 267 `sam` rows it runs SAM (`sam_b.pt`, downloaded by Ultralytics) with the stored box, so no mask images need to be shared. The reported runs used the same SAM masks, computed on CPU with `detection/src/sam_masks.py` and uploaded as `fixed` masks. Check on Nov 9 (CPU): M2 regenerated all 267 SAM masks from the stored boxes, identical to the uploaded ones (IoU 1.0); on GPU they may differ by a few pixels.

## Reproduce

1. **Data** (needs course access): put the 20 `images_OBJxxx` folders into `detection/cluster/data/raw/`; for the zip upload, copy `classification/results/m1/data/classes.json` and `split_manifest.csv` into `detection/cluster/IE7615_project/runs/m1_20261018_022316/data/` (a git checkout on the cluster reads them from git, see [`detection/cluster/README.md`](detection/cluster/README.md)); copy `detection/data/masks/mask_review.csv` into `detection/cluster/IE7615_project/masks/`.
2. **Classification (M1, optional):** open `classification/notebooks/m1_classification.ipynb` in Colab with a T4 GPU and run all cells. Detection does not need it; it only reads the two files above.
3. **Detection, nine runs** on the Explorer cluster (setup in [`detection/cluster/README.md`](detection/cluster/README.md); about 10 min per run on a V100):

   ```bash
   for s in 42 1 2; do for a in A B C; do M2_ARM=$a M2_TRAIN_SEED=$s sbatch m2_job.sbatch; done; done
   ```

   The data are generated with a fixed seed, so all arms and seeds share the same source objects and the same test sets; `M2_TRAIN_SEED` changes only training. `M2_SMOKE=1` runs a 1-epoch check on a few images (also on a CPU).

   On Colab instead: `m2_detection.py` is the notebook source in percent format (`# %%` cells); convert it with `python tools/percent2ipynb.py detection/src/m2_detection.py m2_detection.ipynb` (needs `nbformat`) and open the notebook in Colab. The submitted M2 notebook itself is in the tag `detection-v1.0` (`m2/m2_detection.ipynb`).
4. **Summaries** (repository root, after downloading the run folders into `runs/`):

   ```bash
   python detection/src/summarize_ablation.py
   ```

   ```bash
   python detection/src/eval_real_photos.py
   ```

5. **Tracking, registry, service:** run `mlops/tools/import_runs.py`, `register_models.py` and `export_bundle.py --alias champion --to-models`, then `pytest` and the Docker commands (see [MLOps](#mlops-tracking-registry-and-inference-service)).
6. **Demo:** copy `mobilenetv2.keras` from Drive into `models/` (the detector is already there after step 5) and follow [`demo/README.md`](demo/README.md).

Environment: the cluster job installs its own packages (`detection/cluster/setup_env.sh`; the exact versions of the reported runs are in [`detection/cluster/env_ie7615-m2_freeze.txt`](detection/cluster/env_ie7615-m2_freeze.txt)); the local tools (evaluation, MLflow, service, demo) run in one Python 3.11 environment from [`requirements-local.txt`](requirements-local.txt). Seeds fix the data and the training order, but GPU kernels are not bit-exact: expect differences in the third decimal between runs on different GPUs (see the ± columns in the ablation table).

Code version of the reported runs: their records say `git_commit: not recorded` (the cluster ran a copied script outside git). [`docs/PROVENANCE.md`](docs/PROVENANCE.md) links each run to its commit, Slurm job, GPU, environment and input checksums, checked against the files kept on the cluster.

## MLOps: tracking, registry and inference service

All commands run from the repository root in a Python 3.11 environment with `ultralytics`, `mlflow`, `fastapi`, `uvicorn` and `pytest`. The run folders under `runs/` (weights, manifests) are local and not in git.

**Experiment tracking (MLflow, no retraining).** Tracking is done after the runs, not during them: the training code (Colab notebooks, the cluster job) does not call MLflow and needs no tracking server; each run writes JSON/CSV records and a report into its run folder. `mlops/tools/import_runs.py` reads those records and logs them into a SQLite-backed MLflow store in `mlops/store/` (`mlflow.db`, artifacts in `mlartifacts/`; gitignored):

- **Experiments:** `M1-classification` (4 classifiers), `openset-rejection` (1 parent run with 6 method runs) and `M2-detection` (arm A/B/C × seeds 42/1/2).
- **Each run logs:**
  - params: full config, seed, arm and Ultralytics train args
  - metrics: `val.*` (D1-val, hard-val), `test.*` (D1-test, T-hard, real photos), and per-epoch `train/*` curves
  - checksums: as recorded, and recomputed from the manifests, with a match tag
  - artifacts: reports, figures, records, and the model folder (weights, `classes.json`, `serving.json`, `MODEL_CARD.md`)
- **Run IDs:** the original run ID is kept as the tag `source_run_id`, and `mlops/logs/run_id_map.csv` maps it to the MLflow run ID. Re-running skips runs already imported (`--force` re-imports them).

```bash
python mlops/tools/import_runs.py
```

```bash
mlflow ui --backend-store-uri sqlite:///mlops/store/mlflow.db --port 5000
```

- **Re-evaluation check:** `mlops/tools/reevaluate.py` loads every registered version from the Model Registry and evaluates it again on this machine's CPU. It then compares the result with the metrics its training run logged:
  - detectors: D1-test mAP@0.5 and mAP@0.5:0.95 (tolerance 0.01), and real-photo mAP@0.5 (tolerance 0.005)
  - classifier: M1 test accuracy and macro-F1 (tolerance 0.004, about one image)
  - Each check is logged in the MLflow experiment `re-evaluation`, and the version gets a `reeval_passed` tag. The summary is in [`mlops/logs/reevaluation.md`](mlops/logs/reevaluation.md).
- **Report tables from MLflow:** `mlops/tools/export_tables.py` writes [`mlops/logs/tables_from_mlflow.md`](mlops/logs/tables_from_mlflow.md) from the store alone: M1 classifiers, open-set methods, the A/B/C ablation, real photos, and the registry. `--check` compares the ablation and real-photo tables cell by cell with the summaries generated from the run folders.

```bash
python mlops/tools/reevaluate.py
```

```bash
python mlops/tools/export_tables.py --check
```

**Model registry and rollback.**

- `mlops/tools/register_models.py` registers the seed-42 detector of each arm as versions of `lab2real-detector`: v1 = arm A, v2 = arm B, v3 = arm C (for comparison, no alias). It also registers the M1 classifier (MobileNetV2) as v1 of `lab2real-classifier`, which is not served yet. Each version is tagged with the class-map, weights and manifest checksums, preprocessing, threshold, NMS and runtime.
- **Readiness fixture:** each detector version's folder also holds `readiness/`, one D1-val (development) image plus `expected.json` with the detections that version gave on it, so every bundle carries it.
- Aliases: `champion` is the version to serve; `previous` is the last good one. On registration `champion` → v2 (arm B, the best arm of the comparison), `previous` → v1.
- `mlops/tools/export_bundle.py --alias champion --to-models` resolves the alias to a version number. It writes `model_bundles/lab2real-detector-v<N>/` with the files and a `bundle.json` holding the version, the source run and the SHA-256 of every file. `--to-models` also copies those weights to `models/` and writes `models/selection.json` from the source run, so the demo uses the same model.
- `mlops/tools/rollback_demo.py`:
  - promotes B and serves it, from the bundle and by alias;
  - moves `champion` back to A and restarts the service;
  - checks that `/health` and `/detect` now report version 1.
- The output is in [`mlops/logs/rollback_demo.log`](mlops/logs/rollback_demo.log) (recorded on Nov 8, before the folders were reorganized). The script ends with `champion` on version 1 (arm A). Afterwards `champion` was moved back to version 2 (arm B), the version served now; rerunning the script moves it to version 1 again.

```bash
python mlops/tools/register_models.py
```

```bash
python mlops/tools/rollback_demo.py
```

**Inference service.** `service/app.py` (FastAPI):

- `POST /detect` takes a multipart `file` (image) and an optional `conf` query. It returns class IDs, course IDs, confidences and boxes `[x1, y1, x2, y2]` in pixels of the uploaded image after EXIF rotation, plus the model version.
- `GET /health` reports readiness, the model version and its source.
- **Model selection** (first one set wins):
  - `MODEL_BUNDLE`: an exported bundle; checksums are verified at startup.
  - `MODEL_ALIAS`: resolved in the local registry at startup.
  - `MODEL_PATH`: plain weights; also the fallback when the registry cannot be read.
- **Readiness check:** before the service reports ready, it loads the model and runs it on the bundle's `readiness/` image. It then compares the output with `expected.json`: same classes, box IoU ≥ 0.9, confidence within 0.05. Detections within 0.1 of the threshold may appear or vanish. `/health` shows the result.
  - A model without `readiness/` (plain `MODEL_PATH`) gets a blank-image smoke test instead, and `/health` says so.

```bash
MODEL_ALIAS=champion uvicorn service.app:app --port 8000
```

```bash
curl -F "file=@detection/data/real_photos/images/IMG_2291.jpg" http://127.0.0.1:8000/detect
```

**Tests.** `pytest` (config in `pytest.ini`) runs `service/tests/`. Without weights (fresh clone) 15 tests run and the rest are skipped, with the reason printed:

- `test_no_weights.py`: the copies in `models/` equal their sources in `classification/results/`, the class maps and `selection.json` are valid, the loader rejects missing weights, non-bundles and tampered bundles, and the readiness comparison rule behaves as documented
- `test_demo_parity.py` (needs `models/d1_yolov8n_best.pt`): `demo/predict.py` and the service return the same objects for the same photo and threshold
- `test_service.py` (needs the arm B seed-42 run folder in `runs/`): the API on fixed photos from `detection/data/real_photos/images` against their hand-drawn labels:
  - original-coordinate boxes, EXIF rotation and empty scenes
  - invalid or corrupt images, class-map mismatch and a missing model
  - tampered bundles, alias resolution and the path fallback
  - the readiness check: passes on an exported bundle, and refuses a bundle whose recorded output the model does not reproduce

```bash
pytest
```

**Docker (CPU only).** The image holds the code only. The model bundle is mounted read-only at `/model` (default `MODEL_BUNDLE=/model`):

```bash
docker build -t lab2real-detector:1.0 service
```

```bash
docker run --rm -p 8000:8000 -v "$PWD/model_bundles/lab2real-detector-v2:/model:ro" lab2real-detector:1.0
```

- The image is about 1.6 GB (CPU-only PyTorch and Ultralytics).
- The last section of `mlops/logs/rollback_demo.log` records the Docker check: each bundle was mounted (v2, then v1) and `/detect` was called with curl. `/health` was ready within about 9 s of container start and reported the mounted version, and a non-image upload returned HTTP 400.
- `mlops/tools/rollback_demo.py` rewrites the log, so rerunning it removes that Docker section.

## References

- [MMDetection Simple Copy-Paste](https://github.com/open-mmlab/mmdetection/tree/main/configs/simple_copy_paste): mask-aware compositing and label updates.
- [Made With ML](https://github.com/GokuMohandas/Made-With-ML): experiment tracking and code/data/model testing practices.
- [Warehouse Box Detection](https://github.com/abhijeetgupta23/Box-Detection-in-Warehouse-using-Vision-Based-ML-Engineering): related training-to-API-to-Docker delivery.

Adapted code will retain required license and attribution notices. Reference-project results are not results of this project.