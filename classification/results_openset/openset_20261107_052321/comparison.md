# Rejection methods compared (openset_20261107_052321)

threshold per method = largest value rejecting at most 1% of the val course images; applied once to test. Rates with 95% Wilson intervals; AUROC with a bootstrap 95% interval.
OTHERS test photos are kinds V2 saw in training (unknown only for V1).

| | method | threshold (val) | test known accuracy | test false rejection | OTHERS test rejected | unseen rejected | AUROC (test vs unseen) |
|---|---|---|---|---|---|---|---|
| A | V1 + softmax threshold | 0.759 | 300/303 = 0.990 (0.971–0.997) | 2/303 = 0.007 (0.002–0.024) | 12/15 = 0.800 (0.548–0.930) | 23/36 = 0.639 (0.476–0.775) | 0.984 (0.971–0.995) |
| B | V1 + prototype distance | 0.7066 | 300/303 = 0.990 (0.971–0.997) | 1/303 = 0.003 (0.001–0.018) | 15/15 = 1.000 (0.796–1.000) | 24/36 = 0.667 (0.503–0.798) | 0.987 (0.975–0.997) |
| C | V2 OTHERS class only | – | 302/303 = 0.997 (0.982–0.999) | 1/303 = 0.003 (0.001–0.018) | 12/15 = 0.800 (0.548–0.930) | 15/36 = 0.417 (0.271–0.578) | 0.980 (0.955–0.995) |
| D | V2 OTHERS + softmax threshold | 0.508 | 301/303 = 0.993 (0.976–0.998) | 2/303 = 0.007 (0.002–0.024) | 13/15 = 0.867 (0.621–0.963) | 21/36 = 0.583 (0.422–0.729) | 0.985 (0.974–0.995) |
| E | V2 OTHERS + prototype distance | 0.6946 | 301/303 = 0.993 (0.976–0.998) | 2/303 = 0.007 (0.002–0.024) | 14/15 = 0.933 (0.702–0.988) | 25/36 = 0.694 (0.531–0.820) | 0.991 (0.981–0.998) |
| D* | V2 OTHERS + softmax threshold (deployed T = 0.50) | 0.5 | 301/303 = 0.993 (0.976–0.998) | 2/303 = 0.007 (0.002–0.024) | 13/15 = 0.867 (0.621–0.963) | 21/36 = 0.583 (0.422–0.729) | 0.985 (0.974–0.995) |

## Unseen photos rejected, per kind

| unseen kind | A | B | C | D | E | D* |
|---|---|---|---|---|---|---|
| candle | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| desk_lamp | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| dice | 2/2 | 1/2 | 1/2 | 1/2 | 1/2 | 1/2 |
| electric_kettle | 2/2 | 1/2 | 0/2 | 1/2 | 1/2 | 1/2 |
| glove | 1/2 | 2/2 | 1/2 | 1/2 | 2/2 | 1/2 |
| hat | 2/2 | 1/2 | 0/2 | 1/2 | 0/2 | 1/2 |
| headphones | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| pen | 1/2 | 1/2 | 0/2 | 1/2 | 1/2 | 1/2 |
| power_outlet | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| sock | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| stapler | 1/2 | 2/2 | 1/2 | 1/2 | 2/2 | 1/2 |
| sunglasses | 0/2 | 1/2 | 0/2 | 0/2 | 0/2 | 0/2 |
| tissue_box | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 | 1/2 |
| toaster | 2/2 | 1/2 | 1/2 | 2/2 | 1/2 | 2/2 |
| toothbrush | 0/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| towel | 1/2 | 1/2 | 0/2 | 1/2 | 2/2 | 1/2 |
| toy_car | 2/2 | 1/2 | 0/2 | 0/2 | 1/2 | 0/2 |
| wallet | 1/2 | 2/2 | 0/2 | 0/2 | 2/2 | 0/2 |
