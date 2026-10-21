## test (183,741 decisions, 2.3 h, runs: base_s5, peak_s5, offpeak_s5, ped_heavy_s5, turn_heavy_s5, aggressive_s5, short_green_s5)

| Metric | cv | rule | main | likely | no_spat | no_presence | no_turn_signals | no_leader |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minADE, 6 modes, 8 s (m) | 6.86 | 7.60 | 1.19 | 1.20 | 1.39 | 1.18 | 1.19 | 1.20 |
| minFDE, 6 modes, 8 s (m) | 17.64 | 18.41 | 2.26 | 2.28 | 2.79 | 2.26 | 2.28 | 2.29 |
| Most-likely-mode FDE, 8 s (m) | 17.64 | 18.41 | 9.35 | 10.00 | 11.46 | 9.20 | 9.53 | 9.87 |
|   approaching on red, minFDE 8 s (m) | 13.16 | 9.80 | 1.01 | 1.02 | 1.79 | 1.01 | 1.00 | 1.03 |
|   approaching on green, minFDE 8 s (m) | 21.18 | 32.59 | 3.70 | 3.77 | 4.01 | 3.70 | 3.77 | 3.77 |
| ISS score: top-3 weighted ADE, 5 s (m) | 3.30 | 3.87 | 1.76 | 1.81 | 2.18 | 1.74 | 1.81 | 1.80 |
| Miss rate at 2 m, 8 s | 0.592 | 0.691 | 0.314 | 0.320 | 0.373 | 0.315 | 0.316 | 0.319 |
| Maneuver accuracy (before stop line) |  | 0.898 | 0.908 | 0.908 | 0.912 | 0.901 | 0.646 | 0.910 |
|   turning vehicles |  | 0.783 | 0.783 | 0.781 | 0.788 | 0.778 | 0.287 | 0.783 |
| Crossing vehicles predicted to enter (P >= 0.5) | 0.458 | 0.916 | 0.861 | 0.860 | 0.771 | 0.867 | 0.735 | 0.864 |
| Crossing vehicles: crosswalk entry-time MAE (s) | 6.34 | 3.46 | 2.33 | 2.35 | 3.37 | 2.34 | 3.15 | 2.32 |
|   approaching on red (s) | 7.69 | 3.12 | 1.81 | 1.84 | 4.37 | 1.75 | 2.25 | 1.81 |
|   approaching on green (s) | 5.43 | 3.65 | 2.71 | 2.72 | 2.96 | 2.75 | 3.96 | 2.70 |
|   80% interval coverage | 0.257 | 0.558 | 0.889 | 0.891 | 0.863 | 0.891 | 0.889 | 0.886 |
| All vehicles: Brier, enters within 5 s | 0.065 | 0.066 | 0.015 | 0.016 | 0.021 | 0.016 | 0.019 | 0.015 |
| Conflict AUROC | 0.578 | 0.795 | 0.975 | 0.975 | 0.959 | 0.973 | 0.963 | 0.976 |
| Conflict AUPRC | 0.030 | 0.078 | 0.594 | 0.582 | 0.579 | 0.518 | 0.527 | 0.598 |
| Calibration error, raw | 0.149 | 0.171 | 0.008 | 0.008 | 0.009 | 0.009 | 0.007 | 0.008 |
| Calibration error, calibrated on val | 0.005 | 0.006 | 0.004 | 0.004 | 0.002 | 0.004 | 0.004 | 0.005 |
| Conflict vehicles | 224 | 224 | 224 | 224 | 224 | 224 | 224 | 224 |
| Alert recall | 0.94 | 0.87 | 0.85 | 0.86 | 0.77 | 0.85 | 0.86 | 0.91 |
| Timely recall (lead >= TTA_min) | 0.63 | 0.82 | 0.72 | 0.73 | 0.58 | 0.75 | 0.64 | 0.77 |
| Alert precision | 0.06 | 0.08 | 0.66 | 0.65 | 0.83 | 0.55 | 0.61 | 0.62 |
| Alert F2 | 0.25 | 0.29 | 0.81 | 0.81 | 0.78 | 0.77 | 0.79 | 0.83 |
| Median lead time (s) | 8.4 | 24.7 | 7.0 | 6.2 | 5.3 | 9.2 | 6.2 | 6.9 |
| False-alert vehicles per hour | 1,313 | 967 | 42 | 44 | 15 | 67 | 54 | 54 |
| Share of non-conflict vehicles alerted | 0.630 | 0.464 | 0.020 | 0.021 | 0.007 | 0.032 | 0.026 | 0.026 |
| Threshold p* (from val) | 0.02 | 0.10 | 0.50 | 0.60 | 0.70 | 0.40 | 0.50 | 0.50 |

| Model | Ped occupancy AUROC 0-5 / 5-10 / 10-20 s | Ped occupancy Brier 0-5 s | Conflict AUPRC, direct head | Conflict AUPRC, composed from arrival x occupancy |
| --- | --- | --- | --- | --- |
| main | 0.980 / 0.956 / 0.934 | 0.050 | 0.594 | 0.509 |
| likely | 0.977 / 0.953 / 0.927 | 0.052 | 0.582 | 0.513 |
| no_spat | 0.944 / 0.820 / 0.717 | 0.065 | 0.579 | 0.399 |
| no_presence | 0.941 / 0.939 / 0.929 | 0.081 | 0.518 | 0.443 |
| no_turn_signals | 0.980 / 0.957 / 0.935 | 0.050 | 0.527 | 0.453 |
| no_leader | 0.979 / 0.956 / 0.935 | 0.052 | 0.598 | 0.512 |

| Scenario | cv: timely recall, false alerts/h | rule: timely recall, false alerts/h | main: timely recall, false alerts/h | likely: timely recall, false alerts/h | no_spat: timely recall, false alerts/h | no_presence: timely recall, false alerts/h | no_turn_signals: timely recall, false alerts/h | no_leader: timely recall, false alerts/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive | 0.67, 1,287 (36 events) | 0.75, 927 (36 events) | 0.58, 39 (36 events) | 0.58, 42 (36 events) | 0.47, 18 (36 events) | 0.69, 60 (36 events) | 0.56, 51 (36 events) | 0.69, 60 (36 events) |
| base | 0.73, 1,320 (30 events) | 0.77, 993 (30 events) | 0.73, 48 (30 events) | 0.73, 48 (30 events) | 0.53, 12 (30 events) | 0.67, 75 (30 events) | 0.60, 69 (30 events) | 0.70, 45 (30 events) |
| offpeak | 0.53, 738 (19 events) | 0.79, 540 (19 events) | 0.89, 9 (19 events) | 0.89, 21 (19 events) | 0.74, 3 (19 events) | 0.84, 27 (19 events) | 0.89, 27 (19 events) | 0.89, 15 (19 events) |
| peak | 0.57, 1,944 (30 events) | 0.87, 1,506 (30 events) | 0.60, 69 (30 events) | 0.63, 75 (30 events) | 0.43, 21 (30 events) | 0.77, 108 (30 events) | 0.53, 75 (30 events) | 0.73, 99 (30 events) |
| ped_heavy | 0.71, 1,332 (24 events) | 0.79, 1,032 (24 events) | 0.67, 39 (24 events) | 0.67, 39 (24 events) | 0.58, 9 (24 events) | 0.75, 57 (24 events) | 0.67, 51 (24 events) | 0.67, 54 (24 events) |
| short_green | 0.59, 1,290 (29 events) | 0.83, 924 (29 events) | 0.76, 33 (29 events) | 0.76, 33 (29 events) | 0.66, 15 (29 events) | 0.79, 51 (29 events) | 0.69, 42 (29 events) | 0.83, 45 (29 events) |
| turn_heavy | 0.61, 1,281 (56 events) | 0.89, 849 (56 events) | 0.80, 57 (56 events) | 0.84, 48 (56 events) | 0.66, 27 (56 events) | 0.77, 93 (56 events) | 0.66, 60 (56 events) | 0.84, 57 (56 events) |

## heldout (89,662 decisions, 1.3 h, runs: slow_walkers_s5, slow_walkers_s6, wide_3lane_s5, wide_3lane_s6)

| Metric | cv | rule | main | likely | no_spat | no_presence | no_turn_signals | no_leader |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minADE, 6 modes, 8 s (m) | 7.01 | 6.95 | 1.20 | 1.20 | 1.41 | 1.19 | 1.19 | 1.21 |
| minFDE, 6 modes, 8 s (m) | 18.16 | 16.81 | 2.27 | 2.28 | 2.86 | 2.26 | 2.28 | 2.30 |
| Most-likely-mode FDE, 8 s (m) | 18.16 | 16.81 | 9.49 | 10.18 | 11.99 | 9.29 | 9.67 | 10.18 |
|   approaching on red, minFDE 8 s (m) | 14.14 | 9.43 | 1.05 | 1.08 | 1.99 | 1.07 | 1.06 | 1.08 |
|   approaching on green, minFDE 8 s (m) | 22.07 | 30.62 | 3.79 | 3.81 | 4.00 | 3.74 | 3.86 | 3.84 |
| ISS score: top-3 weighted ADE, 5 s (m) | 3.34 | 3.58 | 1.77 | 1.81 | 2.29 | 1.76 | 1.83 | 1.83 |
| Miss rate at 2 m, 8 s | 0.581 | 0.683 | 0.321 | 0.324 | 0.386 | 0.320 | 0.323 | 0.326 |
| Maneuver accuracy (before stop line) |  | 0.925 | 0.932 | 0.932 | 0.934 | 0.928 | 0.699 | 0.933 |
|   turning vehicles |  | 0.806 | 0.810 | 0.809 | 0.812 | 0.807 | 0.254 | 0.810 |
| Crossing vehicles predicted to enter (P >= 0.5) | 0.456 | 0.939 | 0.909 | 0.907 | 0.799 | 0.920 | 0.811 | 0.908 |
| Crossing vehicles: crosswalk entry-time MAE (s) | 6.21 | 3.07 | 2.03 | 2.03 | 3.33 | 2.02 | 2.56 | 1.99 |
|   approaching on red (s) | 7.91 | 2.62 | 1.54 | 1.47 | 4.41 | 1.44 | 1.83 | 1.49 |
|   approaching on green (s) | 4.87 | 3.33 | 2.38 | 2.41 | 2.75 | 2.43 | 3.25 | 2.37 |
|   80% interval coverage | 0.273 | 0.620 | 0.881 | 0.893 | 0.852 | 0.883 | 0.882 | 0.886 |
| All vehicles: Brier, enters within 5 s | 0.070 | 0.074 | 0.016 | 0.016 | 0.023 | 0.017 | 0.018 | 0.016 |
| Conflict AUROC | 0.546 | 0.814 | 0.985 | 0.986 | 0.978 | 0.983 | 0.974 | 0.985 |
| Conflict AUPRC | 0.030 | 0.089 | 0.690 | 0.681 | 0.647 | 0.629 | 0.609 | 0.691 |
| Calibration error, raw | 0.180 | 0.198 | 0.006 | 0.008 | 0.011 | 0.006 | 0.006 | 0.007 |
| Calibration error, calibrated on val | 0.002 | 0.005 | 0.002 | 0.002 | 0.004 | 0.003 | 0.003 | 0.002 |
| Conflict vehicles | 124 | 124 | 124 | 124 | 124 | 124 | 124 | 124 |
| Alert recall | 0.97 | 0.88 | 0.94 | 0.90 | 0.80 | 0.87 | 0.88 | 0.93 |
| Timely recall (lead >= TTA_min) | 0.65 | 0.85 | 0.80 | 0.77 | 0.60 | 0.78 | 0.68 | 0.85 |
| Alert precision | 0.07 | 0.08 | 0.74 | 0.70 | 0.85 | 0.68 | 0.71 | 0.71 |
| Alert F2 | 0.27 | 0.29 | 0.89 | 0.85 | 0.81 | 0.82 | 0.84 | 0.87 |
| Median lead time (s) | 9.9 | 33.9 | 8.2 | 7.5 | 5.1 | 11.4 | 8.7 | 9.7 |
| False-alert vehicles per hour | 1,203 | 947 | 31 | 36 | 14 | 38 | 33 | 35 |
| Share of non-conflict vehicles alerted | 0.618 | 0.487 | 0.016 | 0.018 | 0.007 | 0.020 | 0.017 | 0.018 |
| Threshold p* (from val) | 0.02 | 0.10 | 0.50 | 0.60 | 0.70 | 0.40 | 0.50 | 0.50 |

| Model | Ped occupancy AUROC 0-5 / 5-10 / 10-20 s | Ped occupancy Brier 0-5 s | Conflict AUPRC, direct head | Conflict AUPRC, composed from arrival x occupancy |
| --- | --- | --- | --- | --- |
| main | 0.984 / 0.966 / 0.953 | 0.046 | 0.690 | 0.620 |
| likely | 0.985 / 0.968 / 0.952 | 0.044 | 0.681 | 0.640 |
| no_spat | 0.952 / 0.865 / 0.792 | 0.070 | 0.647 | 0.534 |
| no_presence | 0.954 / 0.958 / 0.954 | 0.091 | 0.629 | 0.514 |
| no_turn_signals | 0.984 / 0.965 / 0.952 | 0.046 | 0.609 | 0.552 |
| no_leader | 0.984 / 0.967 / 0.953 | 0.047 | 0.691 | 0.627 |

| Scenario | cv: timely recall, false alerts/h | rule: timely recall, false alerts/h | main: timely recall, false alerts/h | likely: timely recall, false alerts/h | no_spat: timely recall, false alerts/h | no_presence: timely recall, false alerts/h | no_turn_signals: timely recall, false alerts/h | no_leader: timely recall, false alerts/h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| slow_walkers | 0.61, 1,239 (59 events) | 0.85, 960 (59 events) | 0.85, 28 (59 events) | 0.81, 30 (59 events) | 0.64, 9 (59 events) | 0.83, 36 (59 events) | 0.80, 28 (59 events) | 0.90, 28 (59 events) |
| wide_3lane | 0.68, 1,167 (65 events) | 0.86, 934 (65 events) | 0.75, 33 (65 events) | 0.74, 42 (65 events) | 0.55, 18 (65 events) | 0.74, 40 (65 events) | 0.57, 38 (65 events) | 0.80, 42 (65 events) |

