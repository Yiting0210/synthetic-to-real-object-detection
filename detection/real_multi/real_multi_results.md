# Real multi-object photos: arm A (D1) vs arm B (D1+D3) vs arm C (D3)

29 phone photos (one mug OBJ011 + one paper roll OBJ002 each, 58 boxes drawn by hand), with distractor objects (bottles, jars, boxes, coaster, pencil case) that are not in the 20 classes. Desk, carpet and chair backgrounds, partial occlusion. Mean ± sample std over seeds 42 / 1 / 2; thresholds are each run's own, chosen on validation.

| metric | arm A (D1) | arm B (D1+D3) | arm C (D3) |
|---|---|---|---|
| AP50 OBJ011 | 0.877 ± 0.028 | 0.937 ± 0.049 | 0.746 ± 0.047 |
| AP50 OBJ002 | 0.757 ± 0.043 | 0.790 ± 0.065 | 0.679 ± 0.005 |
| mAP50 (2 classes) | 0.817 ± 0.008 | 0.864 ± 0.056 | 0.713 ± 0.021 |
| recall OBJ011 | 0.851 ± 0.020 | 0.885 ± 0.072 | 0.736 ± 0.080 |
| recall OBJ002 | 0.575 ± 0.072 | 0.759 ± 0.069 | 0.816 ± 0.040 |
| precision | 0.555 ± 0.039 | 0.551 ± 0.049 | 0.279 ± 0.026 |
| false alarms (other classes) | 24.0 ± 5.3 | 22.3 ± 2.3 | 49.0 ± 7.0 |
| other FP (bad box / duplicate) | 9.3 ± 3.8 | 17.3 ± 8.7 | 67.7 ± 9.6 |

## Per run

| run | thr | AP50 OBJ011 | AP50 OBJ002 | mAP50 (2 classes) | recall OBJ011 | recall OBJ002 | precision | false alarms (other classes) | other FP (bad box / duplicate) | false-alarm classes |
|---|---|---|---|---|---|---|---|---|---|---|
| A seed 1 | 0.25 | 0.894 | 0.732 | 0.813 | 0.828 | 0.655 | 0.589 | 18 | 12 | {'OBJ001': 2, 'OBJ125': 1, 'OBJ201': 1, 'OBJ529': 7, 'OBJ004': 2, 'OBJ608': 3, 'OBJ108': 2} |
| A seed 2 | 0.15 | 0.845 | 0.807 | 0.826 | 0.862 | 0.552 | 0.512 | 28 | 11 | {'OBJ201': 2, 'OBJ608': 12, 'OBJ001': 1, 'OBJ108': 1, 'OBJ004': 4, 'OBJ529': 6, 'OBJ125': 1, 'OBJ523': 1} |
| A seed 42 | 0.15 | 0.891 | 0.731 | 0.811 | 0.862 | 0.517 | 0.563 | 26 | 5 | {'OBJ201': 3, 'OBJ608': 3, 'OBJ001': 1, 'OBJ777': 1, 'OBJ523': 5, 'OBJ222': 2, 'OBJ125': 1, 'OBJ004': 3, 'OBJ529': 7} |
| B seed 1 | 0.55 | 0.966 | 0.803 | 0.885 | 0.966 | 0.828 | 0.500 | 25 | 27 | {'OBJ201': 2, 'OBJ001': 1, 'OBJ529': 9, 'OBJ222': 1, 'OBJ108': 2, 'OBJ608': 4, 'OBJ777': 3, 'OBJ523': 2, 'OBJ123': 1} |
| B seed 2 | 0.6 | 0.881 | 0.719 | 0.800 | 0.862 | 0.690 | 0.556 | 21 | 15 | {'OBJ001': 3, 'OBJ529': 10, 'OBJ201': 1, 'OBJ222': 1, 'OBJ608': 3, 'OBJ523': 3} |
| B seed 42 | 0.7 | 0.965 | 0.847 | 0.906 | 0.828 | 0.759 | 0.597 | 21 | 10 | {'OBJ001': 3, 'OBJ108': 1, 'OBJ529': 4, 'OBJ608': 9, 'OBJ222': 1, 'OBJ523': 3} |
| C seed 1 | 0.55 | 0.704 | 0.685 | 0.694 | 0.690 | 0.793 | 0.261 | 44 | 78 | {'OBJ001': 3, 'OBJ529': 17, 'OBJ201': 3, 'OBJ007': 6, 'OBJ608': 4, 'OBJ108': 4, 'OBJ777': 6, 'OBJ123': 1} |
| C seed 2 | 0.5 | 0.797 | 0.674 | 0.736 | 0.828 | 0.793 | 0.309 | 46 | 59 | {'OBJ529': 15, 'OBJ007': 4, 'OBJ001': 4, 'OBJ608': 13, 'OBJ777': 3, 'OBJ108': 1, 'OBJ111': 1, 'OBJ004': 4, 'OBJ523': 1} |
| C seed 42 | 0.5 | 0.737 | 0.679 | 0.708 | 0.690 | 0.862 | 0.268 | 57 | 66 | {'OBJ007': 11, 'OBJ529': 17, 'OBJ201': 5, 'OBJ001': 2, 'OBJ777': 5, 'OBJ608': 6, 'OBJ222': 1, 'OBJ108': 3, 'OBJ123': 5, 'OBJ523': 2} |

Caveat: only two of the 20 classes can be tested (the other objects belong to classmates); the paper roll is a different roll from the one in training (same category, different instance).
