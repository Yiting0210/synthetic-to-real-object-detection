# Synthetic-to-Real Object Detection

**Lab-to-Real Object ID · IE 7615 Discriminative Deep Learning Project**

Can YOLOv8 trained on synthetic composites generalize to real photographs with clutter, occlusion, and scale changes?

This project compares grid concatenation with mask-based copy-paste training data, evaluates synthetic-to-real generalization, and plans a reproducible CPU inference service. A separate Keras/TensorFlow benchmark supports single-object classification.

> **Status (November 3, 2026):** course milestones M1–M3 are implemented — classifier comparison, grid-concatenated detection data with YOLOv8n, and a local demo. The copy-paste comparison, real-photo evaluation, runtime benchmarks, service and CI/CD below are still planned.

## Repository contents

| Folder | What it contains |
| --- | --- |
| `m1/` | `m1_classification.ipynb`: data checks, near-duplicate grouping, grouped split, four classifiers, validation-based selection, test results (executed on Colab T4) |
| `m2/` | `m2_detection.ipynb`: object masks, grid-composite generation, YOLOv8n fine-tuning, threshold on validation, test results (executed); `src/` holds the notebook source; `M2_report.md` and `figures/` |
| `m3/` | `predict.py` (single image → ID; multi-object image → IDs and boxes) and `demo_app.py` (local Gradio demo); see `m3/README.md` |
| `hpc_m2/` | Slurm job and environment setup to run M2 on a cluster |

Data, trained weights and run outputs are not in the repository; the notebooks read the course dataset from Google Drive.

## Project scope

- **Dataset:** 20 shared course object IDs, approximately 100 single-object images per class; the full data audit is pending. Source-image collection contributes OBJ011.
- **Classification:** implement and compare at least three distinct architectures using Keras/TensorFlow: custom CNN, MobileNetV2 and ResNet-50; EfficientNet-B0 is the fourth candidate if time permits. Hyperparameter variants do not count as additional architectures. Use each model's required preprocessing and justify the best model using validation performance and accuracy/speed/size trade-offs.
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

**Compute plan:** M1 classification runs on Google Colab. From M2, multi-run detection experiments are planned for the university HPC cluster using Slurm, with Colab as fallback. Persist run outputs on storage appropriate to each environment. CPU deployment benchmarks name the exact machine and compare runtimes under matched conditions.

## Planned engineering work

| Area | Implementation | Evidence to publish |
| --- | --- | --- |
| Experiment and data tracking | MLflow, dataset manifests, composite-generation records | Traceable results and regeneration examples |
| Detector CPU inference | PyTorch FP32, ONNX Runtime FP32, OpenVINO FP32/INT8 | Accuracy, mean/p95 latency, memory, artifact size and startup comparisons |
| Serving | FastAPI classification/detection endpoints, SQLite prediction history | API examples and model/runtime version information |
| Testing | pytest, real-model smoke/behavioral tests, coordinate and class-map checks | Reproducible test commands and CI results |
| Load testing | Fixed workload at concurrency 1, 2 and 4 | Successful requests/s, client p95, errors/timeouts and peak memory |
| Model releases | Validated immutable bundles and restart-based rollback | Code/model checksums, readiness checks and recovery logs |
| Continuous delivery | GitHub Actions checks, Docker build/container verification, GHCR publication | Versioned image digests, release manifests and clean pull/run verification |

The four-runtime benchmark applies to the detector. The classifier's native correctness baseline is Keras/TensorFlow; any classifier export is checked against that baseline.

### Planned CI/CD flow

`Pull request → CI checks → merge to main → checks for that commit → Docker build → real-model container smoke test → publish tested image to GHCR`

Failed checks block publication. Image releases identify the exact code commit and model bundle; run and rollback commands pin the image digest. The registry package remains private while this repository is private. A previous image/bundle is retained for a verified restart-based recovery, with SQLite history persisted across restarts.

This is **continuous delivery** of a runnable image, followed by manual deployment. Automated deployment to a hosted service is not currently in scope. CI/CD is a planned capability, not an implemented feature.

## Delivery plan

| Phase | Focus |
| --- | --- |
| M1 · Oct 17 | At least three distinct classifiers, comparison and justified selection; PDF report within five pages including the front page |
| M2 · Oct 27 | Synthetic multi-object dataset, YOLOv8 training and ID/location outputs |
| M3 · Nov 3 | Completed detection model and live single/multi-object demonstration |
| M4 · Nov 10 | Final deliverables subject to instructor details; prioritize report and core results, then detector runtime comparison |
| Nov 11–24 | API, Docker, tests, remaining runtime measurements, model bundles/rollback and GHCR delivery |
| Nov 25–Dec 8 | Load tests, published-image pull/run and recovery verification, reproduction walkthrough and portfolio documentation |

Course deliverables take priority. If dates move, document the revised schedule; do not present pending features as completed.

## Results and examples

Measured so far (single runs, seed 42, Colab Tesla T4):

| Task | Result |
| --- | --- |
| M1 classification (303 test images, 20 classes) | MobileNetV2 selected on validation: test accuracy 0.993, macro-F1 0.993 (2.28 M parameters); EfficientNet-B0 1.000, ResNet-50 0.997, custom CNN 0.891 |
| M2 detection (300 grid-composite test images, 1,154 objects) | YOLOv8n: mAP@0.5 0.908, mAP@0.5:0.95 0.738 |

The detection test set is built like the training data, so it measures in-distribution performance only. Still planned:

- Classification results across at least three architectures, with the selection rationale.
- Grid-only vs grid + copy-paste results on held-out grid and hard synthetic sets.
- OBJ011 real-photo detections and failures, with case-study limits stated.
- Detector FP32/INT8 accuracy, latency and resource tables with measurement conditions.
- API concurrency results and a versioned-image rollback demonstration.

## Running the project

- **M1 / M2:** open the notebook in Colab with a T4 GPU, add a Drive shortcut to the course data folder, and run all cells; outputs are written to `IE7615_project/runs/` on Drive.
- **M3 demo (local CPU):** put the trained models in `m3/models/` and follow `m3/README.md`; the demo runs at `http://127.0.0.1:7860` on the local machine.

Docker and API instructions will accompany those implementations. Final evaluation images will not be used as routine CI fixtures.

## References

- [MMDetection Simple Copy-Paste](https://github.com/open-mmlab/mmdetection/tree/main/configs/simple_copy_paste): mask-aware compositing and label updates.
- [Made With ML](https://github.com/GokuMohandas/Made-With-ML): experiment tracking and code/data/model testing practices.
- [Warehouse Box Detection](https://github.com/abhijeetgupta23/Box-Detection-in-Warehouse-using-Vision-Based-ML-Engineering): related training-to-API-to-Docker delivery.

Adapted code will retain required license and attribution notices. Reference-project results are not results of this project.