# M1 results (m1_20261018_022316)

## Model comparison (best in bold)

| model | params | val acc | val acc 95% CI | val macro-F1 | test acc | test acc 95% CI | test macro-F1 | train min | stage used |
|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 23.63 M | 1.0000 | 0.987–1.000 | 1.0000 | 0.9967 | 0.982–0.999 | 0.9967 | 7.1 | finetune |
| efficientnetb0 | 4.08 M | 1.0000 | 0.987–1.000 | 1.0000 | 1.0000 | 0.987–1.000 | 1.0000 | 7.4 | finetune |
| **mobilenetv2** | 2.28 M | 0.9967 | 0.982–0.999 | 0.9968 | 0.9934 | 0.976–0.998 | 0.9933 | 7.2 | finetune |
| custom_cnn | 1.18 M | 0.9175 | 0.881–0.943 | 0.9164 | 0.8911 | 0.851–0.921 | 0.8899 | 18.0 | scratch |

Training times measured on Tesla T4.

## Justification

Best model: mobilenetv2. Selection rule: Assignment 1 rule: models whose val accuracy is within the 95% Wilson CI of the best are competitive; among them choose the fewest parameters, then the shortest training time; decided on validation before test evaluation.
- Validation accuracy: mobilenetv2 0.9967 (95% CI 0.9815–0.9994, n=303) vs runner-up resnet50 1.0000; competitive within the top model's CI: ['resnet50', 'efficientnetb0', 'mobilenetv2']; chosen by fewest parameters, then training time. Validation macro-F1: 0.9968 vs 1.0000.
- Test macro-F1: 0.9933 vs 0.9967; test accuracy 0.9934 vs 0.9967.
- Size: 2.28 M vs 23.63 M params; training time 7.2 vs 7.1 min (Tesla T4).
- From-scratch baseline custom_cnn test macro-F1: 0.8899 (value of transfer learning).
- Most-confused pairs for mobilenetv2: OBJ123→OBJ777 (1); OBJ004→OBJ007 (1).

## Per-class test F1

| class | custom_cnn | mobilenetv2 | resnet50 | efficientnetb0 |
|---|---|---|---|---|
| OBJ000 | 0.839 | 1.000 | 1.000 | 1.000 |
| OBJ001 | 0.857 | 1.000 | 1.000 | 1.000 |
| OBJ002 | 0.867 | 1.000 | 1.000 | 1.000 |
| OBJ004 | 0.938 | 0.966 | 0.966 | 1.000 |
| OBJ006 | 0.815 | 1.000 | 1.000 | 1.000 |
| OBJ007 | 0.857 | 0.968 | 0.968 | 1.000 |
| OBJ011 | 0.750 | 1.000 | 1.000 | 1.000 |
| OBJ070 | 0.933 | 1.000 | 1.000 | 1.000 |
| OBJ108 | 0.933 | 1.000 | 1.000 | 1.000 |
| OBJ111 | 0.828 | 1.000 | 1.000 | 1.000 |
| OBJ123 | 0.966 | 0.966 | 1.000 | 1.000 |
| OBJ125 | 0.846 | 1.000 | 1.000 | 1.000 |
| OBJ201 | 0.938 | 1.000 | 1.000 | 1.000 |
| OBJ222 | 0.929 | 1.000 | 1.000 | 1.000 |
| OBJ523 | 0.933 | 1.000 | 1.000 | 1.000 |
| OBJ529 | 0.970 | 1.000 | 1.000 | 1.000 |
| OBJ608 | 0.857 | 1.000 | 1.000 | 1.000 |
| OBJ666 | 0.966 | 1.000 | 1.000 | 1.000 |
| OBJ777 | 0.812 | 0.968 | 1.000 | 1.000 |
| OBJ916 | 0.966 | 1.000 | 1.000 | 1.000 |

## Data

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

Git commit: not recorded
