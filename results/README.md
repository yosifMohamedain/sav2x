# Results on the SUMO scenario dataset

Emulated data: 39 runs of 20 minutes (see `configs/scenarios.json`). Models are trained on seeds 1-3
of seven scenarios, calibrated and thresholded on seed 4, and tested on seed 5 (in distribution) and on
two held-out scenarios (slow walkers; three-lane geometry), seeds 5 and 6.

| File | Contents |
| --- | --- |
| `summary.md` | Every headline metric for the baselines and the six models, test and held-out, plus per-scenario alerting |
| `lead_time.md`, `lead_time.json` | Recall at fixed warning leads and matched false-alert rates |
| `test/results.json`, `heldout/results.json` | All metrics from `evaluate.py`, including operating curves and reliability bins |

Models: `main` (all inputs, interval SPaT encoding), `likely` (state and likelyTime only), `no_spat`,
`no_presence` (no SDSM), `no_turn_signals`, `no_leader` (no other vehicles' BSMs). Baselines: `cv`
(constant velocity + pedestrian-window rule) and `rule` (SPaT kinematic path rule + pedestrian-window rule).
