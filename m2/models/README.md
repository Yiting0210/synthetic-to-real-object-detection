# Retrained M2 detector (after mask review)

From the cluster run `m2_20261103_225708` (Explorer, Tesla P100): same code and settings as the submitted M2 run,
but 48 of the 83 flagged masks were reviewed and kept (`../data/mask_review.csv`), so 1,987 instead of 1,939 source images were usable.

| File | What it is |
|---|---|
| `d1_yolov8n_best.pt` | YOLOv8n, best epoch 22 of 42 (16 min). sha256 `68dc75dcbb11…`. Not in git: download it from the GitHub release `m3-demo` |
| `selection.json` | Threshold chosen on D1-val: 0.55 |
| `classes.json` | Class map from M1 (`511d30a1…`) |

D1-test: mAP@0.5 0.928, mAP@0.5:0.95 0.743, macro-F1 0.900 (submitted run: 0.908 / 0.738 / 0.890). OBJ001 AP@0.5 0.658 (was 0.438).
D1-test was regenerated because the usable source images changed, so the comparison is not on identical test images.
Full tables: `../M2_results_retrained.md`. The submitted run (Colab `m2_20261027_123730`): its executed notebook, report and figures are in git under the tag `m2-submitted`; its weights stay on Drive.
