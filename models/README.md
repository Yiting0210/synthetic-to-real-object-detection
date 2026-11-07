# Models

One folder for both models, read by `demo/predict.py` and `demo/demo_app.py`. The weight files are not in git; copy them here from Drive.

| File | What it is | In git |
|---|---|---|
| `classes.json` | Class index ↔ course ID (e.g. `OBJ011`), from M1, shared by both models. sha256 `511d30a1…` | yes |
| `selection.json` | Detector threshold chosen on D1-val: 0.55 | yes |
| `mobilenetv2.keras` | M1 classifier (MobileNetV2), selected on validation; Drive `IE7615_project/runs/m1_20261018_022316/models/` | no |
| `d1_yolov8n_best.pt` | Retrained detector (YOLOv8n), best epoch 22 of 42 (16 min), cluster run `m2_20261103_225708`. sha256 `68dc75dcbb11…` | no |
| `mobilenetv2_others.keras` | Optional 21-class classifier (20 course IDs + `OTHERS`), Colab run `m1o_20261107_113839`. sha256 `9deb2081…`. When present, the demo uses it instead of `mobilenetv2.keras` | no |
| `classes_with_others.json` | Class map of the 21-class model (0–19 as in `classes.json`, 20 = `OTHERS`). sha256 `83d20a48…` | yes |
| `others_threshold.json` | Confidence threshold for the 21-class model, chosen on validation (T = 0.50): a course ID below T becomes `OTHERS` | yes |

## 21-class classifier with OTHERS

Trained with `../classification/m1_others.ipynb` (same recipe as the M1 MobileNetV2, plus 100 OTHERS photos). On the 303 M1 test images:
known accuracy 0.9967 (20-class model 0.9934), 1 image rejected as OTHERS; 12 of 15 OTHERS test photos and 15 of 36 unseen-kind photos rejected.
With the threshold T = 0.50 (`../classification/src/others_threshold.py`): known accuracy 0.9934, 2 rejected; 13 of 15 OTHERS test photos and 21 of 36 unseen-kind photos rejected.
Results: `../classification/results_others/`. Remove the three files above to go back to the 20-class demo.

## Retrained detector (after mask review)

From the cluster run `m2_20261103_225708` (Explorer, Tesla P100): same code and settings as the submitted M2 run,
but 48 of the 83 flagged masks were reviewed and kept (`../detection/results/data/mask_review.csv`), so 1,987 instead of 1,939 source images were usable.

D1-test: mAP@0.5 0.928, mAP@0.5:0.95 0.743, macro-F1 0.900 (submitted run: 0.908 / 0.738 / 0.890). OBJ001 AP@0.5 0.658 (was 0.438).
D1-test was regenerated because the usable source images changed, so the comparison is not on identical test images.
Full tables: `../detection/results/README.md`. The submitted run (Colab `m2_20261027_123730`): its executed notebook, report and figures are in git under the tag `m2-submitted`; its weights stay on Drive.
