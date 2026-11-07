# M1 data report

## Per-class statistics

| class | images | most common size | distinct sizes | formats | converted |
|---|---|---|---|---|---|
| OBJ000 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ001 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ002 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ004 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ006 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ007 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ011 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ070 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ108 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ111 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ123 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ125 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ201 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ222 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ523 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ529 | 114 | 224×224 | 1 | JPEG | 0 |
| OBJ608 | 108 | 224×224 | 1 | JPEG | 0 |
| OBJ666 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ777 | 100 | 224×224 | 1 | JPEG | 0 |
| OBJ916 | 100 | 224×224 | 1 | JPEG | 0 |

## Split

| class | train | val | test | total |
|---|---|---|---|---|
| OBJ000 | 70 | 15 | 15 | 100 |
| OBJ001 | 70 | 15 | 15 | 100 |
| OBJ002 | 70 | 15 | 15 | 100 |
| OBJ004 | 70 | 15 | 15 | 100 |
| OBJ006 | 70 | 15 | 15 | 100 |
| OBJ007 | 70 | 15 | 15 | 100 |
| OBJ011 | 70 | 15 | 15 | 100 |
| OBJ070 | 70 | 15 | 15 | 100 |
| OBJ108 | 70 | 15 | 15 | 100 |
| OBJ111 | 70 | 15 | 15 | 100 |
| OBJ123 | 70 | 15 | 15 | 100 |
| OBJ125 | 70 | 15 | 15 | 100 |
| OBJ201 | 70 | 15 | 15 | 100 |
| OBJ222 | 70 | 15 | 15 | 100 |
| OBJ523 | 70 | 15 | 15 | 100 |
| OBJ529 | 80 | 17 | 17 | 114 |
| OBJ608 | 76 | 16 | 16 | 108 |
| OBJ666 | 70 | 15 | 15 | 100 |
| OBJ777 | 70 | 15 | 15 | 100 |
| OBJ916 | 70 | 15 | 15 | 100 |
| **total** | 1416 | 303 | 303 | 2022 |

## Actual split ratios (target 70/15/15)

| class | train | val | test |
|---|---|---|---|
| OBJ000 | 70% | 15% | 15% |
| OBJ001 | 70% | 15% | 15% |
| OBJ002 | 70% | 15% | 15% |
| OBJ004 | 70% | 15% | 15% |
| OBJ006 | 70% | 15% | 15% |
| OBJ007 | 70% | 15% | 15% |
| OBJ011 | 70% | 15% | 15% |
| OBJ070 | 70% | 15% | 15% |
| OBJ108 | 70% | 15% | 15% |
| OBJ111 | 70% | 15% | 15% |
| OBJ123 | 70% | 15% | 15% |
| OBJ125 | 70% | 15% | 15% |
| OBJ201 | 70% | 15% | 15% |
| OBJ222 | 70% | 15% | 15% |
| OBJ523 | 70% | 15% | 15% |
| OBJ529 | 70% | 15% | 15% |
| OBJ608 | 70% | 15% | 15% |
| OBJ666 | 70% | 15% | 15% |
| OBJ777 | 70% | 15% | 15% |
| OBJ916 | 70% | 15% | 15% |
| **total** | 70.0% | 15.0% | 15.0% |

Near-duplicate threshold: pHash Hamming ≤ 4. Groups: 1983 (39 with ≥2 images).

## Issues

- Folder name had leading/trailing spaces, removed: 'images_OBJ001 ' → OBJ001
- 2 identical files, within one class: ['OBJ007_0012', 'OBJ007_0042']
- 2 identical files, within one class: ['OBJ007_0013', 'OBJ007_0066']
- 2 identical files, within one class: ['OBJ007_0038', 'OBJ007_0088']
- 2 identical files, within one class: ['OBJ125_0006', 'OBJ125_0099']
