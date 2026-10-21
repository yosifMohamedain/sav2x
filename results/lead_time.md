# Early-warning recall at matched false-alert rates

Share of conflict vehicles alerted at least L seconds before they enter the crosswalk, at the best threshold whose false-alert rate stays under the cap. From `scripts/lead_time_analysis.py`; calibration fit on the validation runs.

## test

| Model | L >= 0 s, <= 30/h | <= 60/h | L >= 3 s, <= 60/h | L >= 5 s, <= 60/h | L >= 5 s, <= 120/h | L >= 8 s, <= 120/h |
| --- | --- | --- | --- | --- | --- | --- |
| main | 0.79 | 0.85 | 0.67 | 0.52 | 0.76 | 0.56 |
| likely | 0.76 | 0.89 | 0.76 | 0.62 | 0.74 | 0.56 |
| no_spat | 0.77 | 0.87 | 0.75 | 0.61 | 0.77 | 0.62 |
| no_presence | 0.66 | 0.76 | 0.58 | 0.51 | 0.68 | 0.58 |
| no_turn_signals | 0.80 | 0.86 | 0.62 | 0.53 | 0.55 | 0.36 |

## heldout

| Model | L >= 0 s, <= 30/h | <= 60/h | L >= 3 s, <= 60/h | L >= 5 s, <= 60/h | L >= 5 s, <= 120/h | L >= 8 s, <= 120/h |
| --- | --- | --- | --- | --- | --- | --- |
| main | 0.94 | 0.95 | 0.88 | 0.79 | 0.83 | 0.73 |
| likely | 0.81 | 0.95 | 0.87 | 0.74 | 0.82 | 0.75 |
| no_spat | 0.89 | 0.94 | 0.87 | 0.80 | 0.87 | 0.82 |
| no_presence | 0.77 | 0.94 | 0.85 | 0.68 | 0.86 | 0.73 |
| no_turn_signals | 0.85 | 0.91 | 0.71 | 0.64 | 0.65 | 0.60 |
