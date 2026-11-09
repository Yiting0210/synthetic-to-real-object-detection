# M2 results (m2_20261103_225708)

> Archived: the single run of Nov 3, 2026 (grid composites only, 48 mask review decisions). Current results: [`../../ablation/`](../../ablation/README.md) and [`../../real_photos/`](../../real_photos/README.md).

## Masks

Usable source images: 1987 of 2022; flagged by the automatic checks: 83; manual review decisions: 48.

| class | train img | train groups | val img | val groups | test img | test groups |
|---|---|---|---|---|---|---|
| OBJ000 | 70 | 67 | 15 | 14 | 15 | 14 |
| OBJ001 | 66 | 66 | 14 | 14 | 15 | 15 |
| OBJ002 | 55 | 49 | 12 | 10 | 14 | 12 |
| OBJ004 | 66 | 66 | 14 | 14 | 14 | 14 |
| OBJ006 | 69 | 68 | 13 | 12 | 15 | 15 |
| OBJ007 | 70 | 69 | 15 | 14 | 15 | 14 |
| OBJ011 | 70 | 70 | 15 | 15 | 15 | 15 |
| OBJ070 | 70 | 70 | 15 | 15 | 15 | 15 |
| OBJ108 | 70 | 70 | 15 | 15 | 15 | 15 |
| OBJ111 | 70 | 70 | 15 | 15 | 15 | 15 |
| OBJ123 | 70 | 65 | 15 | 14 | 15 | 14 |
| OBJ125 | 69 | 68 | 15 | 15 | 15 | 15 |
| OBJ201 | 70 | 70 | 15 | 15 | 15 | 15 |
| OBJ222 | 70 | 70 | 15 | 15 | 15 | 15 |
| OBJ523 | 70 | 69 | 15 | 14 | 15 | 15 |
| OBJ529 | 80 | 79 | 17 | 16 | 17 | 16 |
| OBJ608 | 76 | 75 | 16 | 15 | 16 | 16 |
| OBJ666 | 70 | 70 | 15 | 15 | 15 | 15 |
| OBJ777 | 70 | 70 | 15 | 15 | 15 | 15 |
| OBJ916 | 69 | 68 | 15 | 14 | 15 | 15 |

## Generated data

| set | source split | images | objects | objects / image | skipped placements |
|---|---|---|---|---|---|
| D1-train | train | 1500 | 5990 | 3.99 | 6 |
| D1-val | val | 300 | 1208 | 4.03 | 0 |
| D1-test | test | 300 | 1154 | 3.85 | 5 |

### Instances per class

| class | D1-train | D1-val | D1-test |
|---|---|---|---|
| OBJ000 | 300 | 61 | 58 |
| OBJ001 | 300 | 61 | 58 |
| OBJ002 | 300 | 61 | 58 |
| OBJ004 | 300 | 61 | 58 |
| OBJ006 | 300 | 61 | 58 |
| OBJ007 | 300 | 61 | 58 |
| OBJ011 | 300 | 61 | 58 |
| OBJ070 | 300 | 61 | 58 |
| OBJ108 | 300 | 60 | 58 |
| OBJ111 | 300 | 60 | 58 |
| OBJ123 | 299 | 60 | 58 |
| OBJ125 | 299 | 60 | 58 |
| OBJ201 | 299 | 60 | 58 |
| OBJ222 | 299 | 60 | 58 |
| OBJ523 | 299 | 60 | 57 |
| OBJ529 | 299 | 60 | 57 |
| OBJ608 | 299 | 60 | 57 |
| OBJ666 | 299 | 60 | 57 |
| OBJ777 | 299 | 60 | 57 |
| OBJ916 | 299 | 60 | 57 |
| **total** | 5990 | 1208 | 1154 |

Regeneration spot check: 9/9 identical.

## Training

YOLOv8n (COCO-pretrained), 42 epochs (best 22), 16.0 min on Tesla P100-PCIE-12GB.

## Detection results

| set | mAP@0.5 | mAP@0.5:0.95 | precision @0.55 | recall @0.55 | macro-F1 @0.55 |
|---|---|---|---|---|---|
| D1-val | 0.9548 | 0.8112 | 0.9457 | 0.9230 | 0.9330 |
| **D1-test** | 0.9279 | 0.7431 | 0.9070 | 0.8960 | 0.9001 |

General threshold 0.55 chosen on D1-val.

### Per-class (D1-test)

| class | AP@0.5 | AP@0.5:0.95 | F1 @0.55 |
|---|---|---|---|
| OBJ000 | 0.966 | 0.698 | 0.966 |
| OBJ001 | 0.658 | 0.481 | 0.691 |
| OBJ002 | 0.831 | 0.602 | 0.757 |
| OBJ004 | 0.771 | 0.553 | 0.703 |
| OBJ006 | 0.903 | 0.697 | 0.862 |
| OBJ007 | 0.995 | 0.902 | 1.000 |
| OBJ011 | 0.993 | 0.888 | 0.967 |
| OBJ070 | 0.983 | 0.671 | 0.920 |
| OBJ108 | 0.958 | 0.768 | 0.893 |
| OBJ111 | 0.838 | 0.479 | 0.713 |
| OBJ123 | 0.995 | 0.924 | 0.991 |
| OBJ125 | 0.995 | 0.907 | 1.000 |
| OBJ201 | 0.897 | 0.737 | 0.923 |
| OBJ222 | 0.993 | 0.765 | 0.966 |
| OBJ523 | 0.987 | 0.890 | 0.973 |
| OBJ529 | 0.889 | 0.677 | 0.855 |
| OBJ608 | 0.995 | 0.903 | 1.000 |
| OBJ666 | 0.993 | 0.774 | 0.974 |
| OBJ777 | 0.965 | 0.814 | 0.947 |
| OBJ916 | 0.954 | 0.732 | 0.901 |

### Error types (D1-test)

| error type | count |
|---|---|
| missed | 120 |
| wrong_class | 6 |
| wrong_location | 77 |
| duplicate | 7 |
| background | 16 |
| true positives | 1034 |
| ground-truth objects | 1154 |

Wrong-class pairs: [['OBJ222', 'OBJ011', 3], ['OBJ070', 'OBJ111', 1], ['OBJ916', 'OBJ123', 1], ['OBJ916', 'OBJ006', 1]]

Single-object test images through the detector: 119/303 correct, 161 without detection.

## Issues

- 35 source images excluded from detection (bad mask): OBJ001_0006, OBJ001_0030, OBJ001_0040, OBJ001_0043, OBJ001_0089, OBJ002_0001, OBJ002_0020, OBJ002_0021, OBJ002_0022, OBJ002_0027, OBJ002_0028, OBJ002_0041, OBJ002_0045, OBJ002_0047, OBJ002_0048, OBJ002_0054, OBJ002_0065, OBJ002_0070, OBJ002_0072, OBJ002_0075, OBJ002_0077, OBJ002_0091, OBJ002_0095, OBJ002_0097, OBJ004_0000, OBJ004_0012, OBJ004_0016, OBJ004_0054, OBJ004_0056, OBJ004_0064 …

Git commit: not recorded
