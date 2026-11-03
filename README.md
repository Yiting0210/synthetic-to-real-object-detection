# Synthetic-to-Real Object Detection

**Lab-to-Real Object ID · IE 7615 Discriminative Deep Learning Project**

Can YOLOv8 trained on synthetic composites generalize to real photographs with clutter, occlusion, and scale changes?

This project compares grid concatenation with mask-based copy-paste training data, evaluates synthetic-to-real generalization, and adds ML-specific engineering: experiment tracking, a model registry with rollback, and CI that checks the real exported model. A separate Keras/TensorFlow benchmark supports single-object classification.

> **Status (November 3, 2026):** course milestones M1–M3 are implemented — classifier comparison, grid-concatenated detection data with YOLOv8n, and a local demo. The copy-paste comparison, real-photo evaluation, runtime benchmarks and the engineering work below are still planned.

## Repository contents

| Folder | What it contains |
| --- | --- |
| `m1/` | `m1_classification.ipynb`: data checks, near-duplicate grouping, grouped split, four classifiers, validation-based selection, test results (executed on Colab T4) |
| `m2/` | `m2_detection.ipynb`: object masks, grid-composite generation, YOLOv8n fine-tuning, threshold on validation, test results (executed); `src/` holds the notebook source; `M2_report.md` and `figures/` |
| `m3/` | `predict.py` (single image → ID; multi-object image → IDs and boxes) and `demo_app.py` (local Gradio demo); see `m3/README.md` |
| `hpc_m2/` | Slurm job and environment setup to run M2 on a cluster |

Data, trained weights and run outputs are not in the repository; the notebooks read the course dataset from Google Drive.

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

**Compute plan:** M1 and M2 ran on Google Colab (Tesla T4). The multi-seed detection comparison is planned for the university HPC cluster using Slurm (`hpc_m2/`), with Colab as fallback. Persist run outputs on storage appropriate to each environment. CPU deployment benchmarks name the exact machine and compare runtimes under matched conditions.

## Planned engineering work

The engineering scope is limited to ML-specific practices; general software CI/CD, data pipelines, monitoring and load testing are not part of this project.

| Area | Implementation | Evidence to publish |
| --- | --- | --- |
| Experiment tracking | MLflow with a local SQLite backend; Colab and HPC run records (JSON/CSV) imported by script | Every reported number traces to a run with its code commit, configuration, data checksums and model |
| Model registry and rollback | MLflow Model Registry; `champion` / `previous` aliases; exported model bundles with checksums | A recorded rollback to the previous version, confirmed through the service response |
| Model-aware CI | GitHub Actions running checks on a pinned, exported detector: class map, letterbox coordinates, ONNX vs PyTorch outputs, known-object detection | CI run history on every push |
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
| Nov 11–24 | Remaining runtime measurements, FastAPI + Docker, model-aware CI, MLflow import and Model Registry with a rollback demonstration, README results linked to runs |
| December (optional) | Open-set rejection, webcam demo, and a single-object data-drift case study |

Course deliverables take priority. If dates move, document the revised schedule; do not present pending features as completed.

## Results and examples

Measured so far (single runs, seed 42, Colab Tesla T4):

| Task | Result |
| --- | --- |
| M1 classification (303 test images, 20 classes) | MobileNetV2 selected on validation: test accuracy 0.993, macro-F1 0.993 (2.28 M parameters); EfficientNet-B0 1.000, ResNet-50 0.997, custom CNN 0.891 |
| M2 detection (300 grid-composite test images, 1,154 objects) | YOLOv8n: mAP@0.5 0.908, mAP@0.5:0.95 0.738 |

The detection test set is built like the training data, so it measures in-distribution performance only. Still planned:

- Grid-only vs grid + copy-paste results on held-out grid and hard synthetic sets.
- OBJ011 real-photo detections and failures, with case-study limits stated.
- Detector FP32/INT8 accuracy, latency and resource tables with measurement conditions.
- MLflow-tracked runs, a Model Registry rollback demonstration, and CI history of the real-model checks.

## Running the project

- **M1 / M2:** open the notebook in Colab with a T4 GPU, add a Drive shortcut to the course data folder, and run all cells; outputs are written to `IE7615_project/runs/` on Drive.
- **M3 demo (local CPU):** put the trained models in `m3/models/` and follow `m3/README.md`; the demo runs at `http://127.0.0.1:7860` on the local machine.

Docker and API instructions will accompany those implementations. Final evaluation images will not be used as routine CI fixtures.

## References

- [MMDetection Simple Copy-Paste](https://github.com/open-mmlab/mmdetection/tree/main/configs/simple_copy_paste): mask-aware compositing and label updates.
- [Made With ML](https://github.com/GokuMohandas/Made-With-ML): experiment tracking and code/data/model testing practices.
- [Warehouse Box Detection](https://github.com/abhijeetgupta23/Box-Detection-in-Warehouse-using-Vision-Based-ML-Engineering): related training-to-API-to-Docker delivery.

Adapted code will retain required license and attribution notices. Reference-project results are not results of this project.