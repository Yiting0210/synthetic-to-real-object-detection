# Re-evaluation of registered model versions

Written by `mlops/tools/reevaluate.py` on 2026-11-11 08:34 UTC (CPU). Each version was loaded from the MLflow Model Registry and evaluated again; reported = the metric its training run logged. Tolerances: D1-test mAP 0.01, real photos 0.005, classifier 0.004.

| model | version | aliases | metric | re-evaluated | reported | difference | within tolerance |
|---|---|---|---|---|---|---|---|
| lab2real-detector | 1 | previous | test.d1.map50 | 0.9570 | 0.9570 | +0.0000 | yes |
| lab2real-detector | 1 | previous | test.d1.map50_95 | 0.8207 | 0.8207 | +0.0000 | yes |
| lab2real-detector | 1 | previous | test.real.map50_2cls | 0.8113 | 0.8113 | +0.0000 | yes |
| lab2real-detector | 2 | champion | test.d1.map50 | 0.9618 | 0.9618 | +0.0000 | yes |
| lab2real-detector | 2 | champion | test.d1.map50_95 | 0.8194 | 0.8194 | +0.0000 | yes |
| lab2real-detector | 2 | champion | test.real.map50_2cls | 0.9063 | 0.9063 | +0.0000 | yes |
| lab2real-detector | 3 | - | test.d1.map50 | 0.6181 | 0.6181 | +0.0000 | yes |
| lab2real-detector | 3 | - | test.d1.map50_95 | 0.4230 | 0.4230 | +0.0000 | yes |
| lab2real-detector | 3 | - | test.real.map50_2cls | 0.7078 | 0.7078 | +0.0000 | yes |
| lab2real-classifier | 1 | champion | test.accuracy | 0.9934 | 0.9934 | +0.0000 | yes |
| lab2real-classifier | 1 | champion | test.macro_f1 | 0.9933 | 0.9933 | +0.0000 | yes |

| model | version | MLflow re-evaluation run | details |
|---|---|---|---|
| lab2real-detector | 1 | `557f74ec1d394e4c9c02537c1e22ad5d` | data: runs/m2_20261108_054413_armA_seed42/data/d1_dataset.zip (images/test, 300) + detection/data/real_photos (29); real_ap50: {"OBJ011": 0.8912, "OBJ002": 0.7314} |
| lab2real-detector | 2 | `a50a427b195c4ef2bfd862235ca498d3` | data: runs/m2_20261108_054413_armB_seed42/data/d1_dataset.zip (images/test, 300) + detection/data/real_photos (29); real_ap50: {"OBJ011": 0.9653, "OBJ002": 0.8474} |
| lab2real-detector | 3 | `85e0d867e6c44ee6bfa839d57dfd66b6` | data: runs/m2_20261108_183715_armC_seed42/data/d1_dataset.zip (images/test, 300) + detection/data/real_photos (29); real_ap50: {"OBJ011": 0.7366, "OBJ002": 0.6791} |
| lab2real-classifier | 1 | `6a1eaa1081b044e8b0b1177a2108dc39` | data: M1 test split, 303 images (by MD5 from split_manifest.csv); correct: 301/303 |
