# Models

One folder for both models, read by `demo/predict.py`, `demo/demo_app.py` and the service fallback. The weight files are not in git. The detector files are exported from the MLflow registry (`python mlops/tools/export_bundle.py --alias champion --to-models`); the classifier weights come from Drive.

| File | What it is | In git |
|---|---|---|
| `classes.json` | Class index ↔ course ID (e.g. `OBJ011`), from M1, shared by both models. sha256 `511d30a1…` | yes |
| `selection.json` | Selection record of the registry `champion` (now v2: arm B, seed 42, run `m2_20261108_054413_armB_seed42`): threshold 0.70 chosen on D1-val + hard-val, plus a `registry` block (version, run ID, bundle) | yes |
| `mobilenetv2.keras` | M1 classifier (MobileNetV2), selected on validation; Drive `IE7615_project/runs/m1_20261018_022316/models/` | no |
| `d1_yolov8n_best.pt` | Detector (YOLOv8n) of the registry `champion`: arm B (D1 + D3), seed 42, best epoch 54 of 60; its sha256 is `weights_sha256` in `selection.json` | no |
| `mobilenetv2_others.keras` | Optional 21-class classifier (20 course IDs + `OTHERS`), Colab run `m1o_20261107_113839`. sha256 `9deb2081…`. When present, the demo uses it instead of `mobilenetv2.keras` | no |
| `classes_with_others.json` | Class map of the 21-class model (0–19 as in `classes.json`, 20 = `OTHERS`). sha256 `83d20a48…` | yes |
| `others_threshold.json` | Confidence threshold for the 21-class model, chosen on validation (T = 0.50): a course ID below T becomes `OTHERS` | yes |

## 21-class classifier with OTHERS

Trained with `../classification/notebooks/m1_others.ipynb` (same recipe as the M1 MobileNetV2, plus 100 OTHERS photos). On the 303 M1 test images:
known accuracy 0.9967 (20-class model 0.9934), 1 image rejected as OTHERS; 12 of 15 OTHERS test photos and 15 of 36 unseen-kind photos rejected.
With the threshold T = 0.50 (`../classification/src/others_threshold.py`): known accuracy 0.9934, 2 rejected; 13 of 15 OTHERS test photos and 21 of 36 unseen-kind photos rejected.
Results: `../classification/results/others/`. Remove the three files above to go back to the 20-class demo.

## Current detector (arm B)

Chosen by the D1 vs D1+D3 vs D3 comparison (three seeds per arm, [`../detection/results/ablation/`](../detection/results/ablation/README.md)): arm B mixes 750 grid composites with 750 copy-paste composites built from the same objects. Seed 42: D1-test mAP@0.5 0.962, T-hard 0.940; real photos in [`../detection/results/real_photos/`](../detection/results/real_photos/README.md). Masks: 711 review decisions (`../detection/data/masks/mask_review.csv`).

## Earlier detector (after the first mask review)

From the cluster run `m2_20261103_225708` (Explorer, Tesla P100): same code and settings as the submitted M2 run,
but 48 of the 83 flagged masks were reviewed and kept (`../detection/results/archive/m2_20261103_225708/data/mask_review.csv`), so 1,987 instead of 1,939 source images were usable.

D1-test: mAP@0.5 0.928, mAP@0.5:0.95 0.743, macro-F1 0.900 (submitted run: 0.908 / 0.738 / 0.890). OBJ001 AP@0.5 0.658 (was 0.438).
D1-test was regenerated because the usable source images changed, so the comparison is not on identical test images.
Full tables: `../detection/results/archive/m2_20261103_225708/README.md`. The submitted run (Colab `m2_20261027_123730`): its executed notebook, report and figures are in git under the tag `detection-v1.0`; its weights stay on Drive.
