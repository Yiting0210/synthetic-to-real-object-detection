# M1 extension: MobileNetV2 + OTHERS (m1o_20261107_113839)

OTHERS photos: 100 in 31 kinds (train / val / test = 70 / 15 / 15); unseen-object photos: 36 in 18 kinds.

## Validation

| metric | value |
|---|---|
| accuracy_21 | 0.9874 (314/318; CI 0.968–0.995) |
| known_accuracy | 0.9934 (301/303; CI 0.976–0.998) |
| false_rejection | 0.0000 (0/303; CI 0.000–0.013) |
| others_recall | 0.8667 (13/15; CI 0.621–0.963) |
| others_precision | 1.0000 (13/13; CI 0.772–1.000) |

## Test

| metric | value |
|---|---|
| accuracy_21 | 0.9874 (314/318; CI 0.968–0.995) |
| known_accuracy | 0.9967 (302/303; CI 0.982–0.999) |
| false_rejection | 0.0033 (1/303; CI 0.001–0.018) |
| others_recall | 0.8000 (12/15; CI 0.548–0.930) |
| others_precision | 0.9231 (12/13; CI 0.667–0.986) |
| unseen rejected as OTHERS | 0.4167 (15/36; CI 0.271–0.578) |

M1 20-class MobileNetV2: validation 0.9967, test 0.9934 (same 303 test images).

## Data issues

none
