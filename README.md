# Synthetic-to-Real Object Detection

**Lab-to-Real Object ID · IE 7615 Discriminative Deep Learning Project**

Can an object detector trained on synthetic image composites generalize to real photographs with clutter, occlusion, and scale changes?

This project studies that question using grid-concatenated and mask-based copy-paste training images, YOLOv8, and a controlled evaluation protocol. A separate classification benchmark supports single-object identification.

> **Status: planning and setup.** The repository has been initialized. Training results, detection examples, inference benchmarks, and a runnable service have not been published. Features below are planned unless explicitly marked complete.

## Scope

- **Shared dataset:** 20 course object IDs, approximately 100 single-object images per class; a complete data audit is pending.
- **Classification:** compare at least three distinct image architectures on identical source splits. Planned candidates are a custom CNN, MobileNetV2, ResNet-50, and optionally EfficientNet-B0. Hyperparameter variants count as one architecture.
- **Detection:** fine-tune YOLOv8n on randomly concatenated single-object images, generating at least 100 multi-object images for the course requirement.
- **Synthetic-data experiment:** compare grid-only training against a fixed-size mixture of grid and copy-paste images, controlling data volume and training budget.
- **Real-world evaluation:** an OBJ011-only case study using separately collected real photographs. This does **not** establish real-world performance across all 20 classes.
- **Ownership:** instructor-approved solo completion of the group assignment, with no teammates. The shared class dataset is used for training; the student's own source-image contribution is OBJ011.

## Evaluation design

1. Group near-duplicate source images before splitting into train, validation, and test sets.
2. Build each synthetic split only from its corresponding source split. Keep real validation and test photographs separated by shooting session.
3. Select models, checkpoints, and confidence thresholds using validation data. Reserve test data for reporting frozen choices.
4. Use reviewed masks to derive visible-object boxes and update labels after geometric transformations and occlusion.
5. Report classification accuracy and macro-F1; detection mAP@0.5 and mAP@0.5:0.95; and real-case recall, matched-box IoU, and false positives per image.
6. Track configurations, seeds, code revisions, data manifests, generation records, and model artifacts. Report repeated-seed variation when compute permits.

## Planned engineering work

| Area | Planned implementation | Evidence to publish |
| --- | --- | --- |
| Reproducibility | MLflow tracking and per-composite generation records | Run-linked results and regeneration examples |
| CPU inference | PyTorch FP32, ONNX Runtime FP32, OpenVINO FP32/INT8 | Accuracy, mean/p95 latency, memory, artifact size, and startup comparisons |
| Serving | FastAPI classification/detection endpoints and SQLite prediction history | API examples and versioned responses |
| Packaging and tests | Docker, pytest, real-model behavioral checks, GitHub Actions CI | Runnable instructions and test results |
| Service performance | Fixed-workload load tests at concurrency 1, 2, and 4 | Throughput, client p95 latency, errors, and peak memory |
| Model releases | Validated immutable model bundles and restart-based rollback | Bundle metadata and a recorded rollback exercise |

Course deliverables take priority. API packaging, load testing, and complete release/rollback workflows are scheduled for the post-course portfolio phase.

## Results and examples

No measured results are available yet. This section will contain:

- Classification comparison across at least three architectures, with the best model justified.
- Grid-only versus grid + copy-paste detection results on held-out synthetic sets.
- OBJ011 real-photo case-study results, sample detections, and failure examples.
- FP32/INT8 runtime and resource comparisons with hardware and measurement conditions.

Targets and planned features are not reported as achieved results.

## Running the project

Executable training and serving code has not been published yet. Setup, reproduction, and Docker/API commands will be added alongside verified implementations.

## References

- [Simple Copy-Paste implementation in MMDetection](https://github.com/open-mmlab/mmdetection/tree/main/configs/simple_copy_paste): mask-aware compositing and label updates.
- [Made With ML](https://github.com/GokuMohandas/Made-With-ML): experiment tracking and code/data/model testing practices.
- [Warehouse Box Detection](https://github.com/abhijeetgupta23/Box-Detection-in-Warehouse-using-Vision-Based-ML-Engineering): a related training-to-API-to-Docker project.

Adapted code will retain its required license and attribution notices. Reference-project results are not results of this project.