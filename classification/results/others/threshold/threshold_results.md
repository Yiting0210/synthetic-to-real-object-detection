# Confidence threshold on the 21-class model

Rule (fixed before test): course ID only if its probability >= T, else OTHERS; T = largest value in 0.50..0.99 with validation false rejection <= 1%.

Chosen on validation: **T = 0.5** (validation false rejection 0.0099, validation OTHERS recall 0.867).

## Test and unseen (T applied once)

| | test known accuracy | test false rejection | test OTHERS recall | unseen rejected as OTHERS |
|---|---|---|---|---|
| no threshold | 0.9967 (302/303; CI 0.982–0.999) | 0.0033 (1/303; CI 0.001–0.018) | 0.8000 (12/15; CI 0.548–0.930) | 0.4167 (15/36; CI 0.271–0.578) |
| T = 0.5 | 0.9934 (301/303; CI 0.976–0.998) | 0.0066 (2/303; CI 0.002–0.024) | 0.8667 (13/15; CI 0.621–0.963) | 0.5833 (21/36; CI 0.422–0.729) |

## Unseen kinds at T = 0.5

| unseen kind | rejected | mistaken for |
|---|---|---|
| candle | 1/2 | OBJ222 |
| desk_lamp | 2/2 | – |
| dice | 1/2 | OBJ108 |
| electric_kettle | 1/2 | OBJ001 |
| glove | 1/2 | OBJ777 |
| hat | 1/2 | OBJ001 |
| headphones | 2/2 | – |
| pen | 1/2 | OBJ608 |
| power_outlet | 2/2 | – |
| sock | 2/2 | – |
| stapler | 1/2 | OBJ777 |
| sunglasses | 0/2 | OBJ201 |
| tissue_box | 1/2 | OBJ125 |
| toaster | 2/2 | – |
| toothbrush | 2/2 | – |
| towel | 1/2 | OBJ001 |
| toy_car | 0/2 | OBJ007 |
| wallet | 0/2 | OBJ001 |
