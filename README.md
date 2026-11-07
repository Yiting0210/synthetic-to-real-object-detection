# Synthetic-to-Real Object Detection

**Lab-to-Real Object ID · IE 7615 Discriminative Deep Learning Project**

Can YOLOv8 trained on synthetic composites generalize to real photographs with clutter, occlusion, and scale changes?

This project compares grid concatenation with mask-based copy-paste training data, evaluates synthetic-to-real generalization, and adds ML-specific engineering: experiment tracking and a model registry with rollback. A separate Keras/TensorFlow benchmark supports single-object classification.

> **Status (November 6, 2026):** course milestones M1–M3 are done — classifier comparison, grid-concatenated detection data with YOLOv8n, and a local demo. After the M2 submission, the detector was retrained on the university cluster with reviewed masks (same code). The copy-paste comparison, real-photo evaluation, runtime benchmarks and the engineering work below are still planned.

## Repository contents

| Folder | What it contains |
| --- | --- |
| `classification/` | `m1_classification.ipynb`: data checks, near-duplicate grouping, grouped split, four classifiers, validation-based selection, test results (executed on Colab T4) |
| `detection/src/` | `m2_detection.py`: object masks, grid-composite generation, YOLOv8n fine-tuning, threshold on validation, test results (run as a notebook on Colab or as a script on the cluster) |
| `detection/cluster/` | Slurm job and environment setup to run the detection pipeline on the Explorer cluster |
| `detection/results/` | Results of the current (retrained) detector run: `README.md` (tables), `figures/`, `records/`; `data/` holds the mask review, mask and composite manifests and the per-composite generation records |
| `models/` | Class map and detector threshold shared by both models; the trained weights go here (not in git), see `models/README.md` |
| `demo/` | `predict.py` (single image → ID; multi-object image → IDs and boxes) and `demo_app.py` (local Gradio demo) with example images; see `demo/README.md` |

Model versions are marked with git tags per component, named `<component>-v<major>.<minor>` (each tag is a snapshot of the whole repository):

- **minor + 1** (v1.0 → v1.1): same method, only the data or settings changed;
- **major + 1** (v1 → v2): the method changed (for detection: the way the training data is generated).

| Tag | What changed |
|---|---|
| [`classification-v1.0`](../../tree/classification-v1.0) | 20-class MobileNetV2 (M1, single-object identification) |
| [`classification-v1.1`](../../tree/classification-v1.1) | Same model and training recipe; data adds an `OTHERS` class (100 photos), setting adds a rejection threshold chosen on validation |
| [`detection-v1.0`](../../tree/detection-v1.0) | YOLOv8n on grid composites, as submitted for M2 (Colab run, D1-test mAP@0.5 0.908), with the earlier `m1/`, `m2/`, `m3/` layout |
| [`detection-v1.1`](../../tree/detection-v1.1) | Same grid composites; masks reviewed, retrained on the cluster (D1-test mAP@0.5 0.928) |
| [`demo`](../../tree/demo) | Local demo (M3): single-object ID with OTHERS, multi-object IDs and boxes |

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

**Compute plan:** M1 and the submitted M2 ran on Google Colab (Tesla T4); the retrained M2 ran on the Explorer cluster (Tesla P100, Slurm). The multi-seed detection comparison is planned for the university HPC cluster using Slurm (`detection/cluster/`), with Colab as fallback. Persist run outputs on storage appropriate to each environment. CPU deployment benchmarks name the exact machine and compare runtimes under matched conditions.

## Planned engineering work

The engineering scope is limited to ML-specific practices; data pipelines, monitoring and load testing are not part of this project.

| Area | Implementation | Evidence to publish |
| --- | --- | --- |
| Experiment tracking | MLflow with a local SQLite backend; Colab and HPC run records (JSON/CSV) imported by script | Every reported number traces to a run with its code commit, configuration, data checksums and model |
| Model registry and rollback | MLflow Model Registry; `champion` / `previous` aliases; exported model bundles with checksums | A recorded rollback to the previous version, confirmed through the service response |
| Minimal inference service | FastAPI `POST /detect` (then `/classify`), Docker image with the model bundle mounted read-only, pytest | Tested API calls that report the model version |
| Detector CPU inference | PyTorch FP32, ONNX Runtime FP32, OpenVINO FP32/INT8 | Accuracy, mean/p95 latency, memory, artifact size and load time under matched conditions |

The four-runtime benchmark applies to the detector. The classifier's native correctness baseline is Keras/TensorFlow; any classifier export is checked against that baseline. No speed or accuracy number is published for an export that has not passed the class-map, preprocessing and coordinate checks.

## Delivery plan

| Phase | Focus |
| --- | --- |
| M1 · Oct 17 | At least three distinct classifiers, comparison and justified selection; PDF report within five pages including the front page |
| M2 · Oct 27 | Synthetic multi-object dataset, YOLOv8 training and ID/location outputs |
| M3 · Nov 3 | Completed detection model and live single/multi-object demonstration |
| M4 · Nov 10 | Final report and models; then detector runtime comparison (at least PyTorch FP32 vs OpenVINO INT8) |
| Nov 11–24 | Remaining runtime measurements, FastAPI + Docker, MLflow import and Model Registry with a rollback demonstration, README results linked to runs |
| December (optional) | Webcam demo and a single-object data-drift case study |

Course deliverables take priority. If dates move, document the revised schedule; do not present pending features as completed.

## Results and examples

Measured so far (single runs, seed 42):

| Task | Result |
| --- | --- |
| M1 classification (303 test images, 20 classes; Colab T4) | MobileNetV2 selected on validation: test accuracy 0.993, macro-F1 0.993 (2.28 M parameters); EfficientNet-B0 1.000, ResNet-50 0.997, custom CNN 0.891 |
| M2 detection, submitted (Colab T4; 1,939 usable source images) | YOLOv8n on 300 grid-composite test images: mAP@0.5 0.908, mAP@0.5:0.95 0.738, macro-F1 0.890 (threshold 0.50) |
| M2 detection, retrained (cluster P100; 1,987 usable source images after mask review) | YOLOv8n on 300 grid-composite test images (1,154 objects): mAP@0.5 0.928, mAP@0.5:0.95 0.743, macro-F1 0.900 (threshold 0.55) |

The two M2 test sets were generated from different usable source images, so the two rows are not on identical test images. The detection test set is built like the training data, so it measures in-distribution performance only. Still planned:

- Grid-only vs grid + copy-paste results on held-out grid and hard synthetic sets.
- OBJ011 real-photo detections and failures, with case-study limits stated.
- Detector FP32/INT8 accuracy, latency and resource tables with measurement conditions.
- MLflow-tracked runs and a Model Registry rollback demonstration.

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

Details: `classification/results_openset/`.

## Running the project

- **Classification (M1):** open `classification/m1_classification.ipynb` in Colab with a T4 GPU, add a Drive shortcut to the course data folder, and run all cells; outputs are written to `IE7615_project/runs/` on Drive.
- **Detection (M2):** on the cluster, follow `detection/cluster/README.md` (Slurm job running `detection/src/m2_detection.py`); on Colab, convert the script with `detection/src/percent2ipynb.py` or use the notebook under the tag `detection-v1.0`.
- **Demo (M3, local CPU):** put the two weight files into `models/` and follow `demo/README.md`; the demo runs at `http://127.0.0.1:7860` on the local machine.

Docker and API instructions will accompany those implementations.

## References

- [MMDetection Simple Copy-Paste](https://github.com/open-mmlab/mmdetection/tree/main/configs/simple_copy_paste): mask-aware compositing and label updates.
- [Made With ML](https://github.com/GokuMohandas/Made-With-ML): experiment tracking and code/data/model testing practices.
- [Warehouse Box Detection](https://github.com/abhijeetgupta23/Box-Detection-in-Warehouse-using-Vision-Based-ML-Engineering): related training-to-API-to-Docker delivery.

Adapted code will retain required license and attribution notices. Reference-project results are not results of this project.