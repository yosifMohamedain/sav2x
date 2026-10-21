| Metric | cv | cv_spat | infra_none | infra_likely | infra_interval |
| --- | --- | --- | --- | --- | --- |
| Vehicle before stop line, minADE6 @5 s (m) | 2.945 | 4.381 | 0.678 | 0.624 | 0.615 |
| Vehicle before stop line, minFDE6 @8 s (m) | 16.455 | 23.697 | 2.980 | 2.523 | 2.482 |
| Vehicle before stop line, top-1 FDE @8 s (m) | 16.455 | 23.697 | 9.176 | 8.163 | 8.171 |
|   approaching on red, minFDE6 @8 s (m) | 12.561 | 18.526 | 1.125 | 0.868 | 0.847 |
|   approaching on green, minFDE6 @8 s (m) | 22.570 | 31.817 | 5.894 | 5.122 | 5.048 |
| Pedestrian minADE6 @5 s (m) | 0.851 | 0.942 | 0.261 | 0.233 | 0.235 |
| All agents, ISS top-3 weighted ADE @5 s (m) | 2.613 | 3.420 | 1.311 | 1.270 | 1.266 |
| All agents, miss rate @8 s, 2 m | 0.639 | 0.665 | 0.244 | 0.219 | 0.214 |
| Maneuver accuracy |  |  | 0.714 | 0.728 | 0.718 |
| Maneuver accuracy, turning vehicles |  |  | 0.168 | 0.220 | 0.193 |
| Maneuver accuracy, 0-20 m to stop line |  |  | 0.703 | 0.728 | 0.712 |
| Maneuver accuracy, 50+ m to stop line |  |  | 0.728 | 0.733 | 0.725 |
| Crossing vehicles, share predicted to enter (P >= 0.5) | 0.460 | 0.816 | 0.722 | 0.797 | 0.783 |
| Crossing vehicles, entry-time MAE, all (s) | 5.902 | 4.095 | 3.666 | 2.878 | 3.001 |
| Crossing vehicles, entry-time MAE, detected only (s) | 2.782 | 3.039 | 2.196 | 1.711 | 1.651 |
|   approaching on red, MAE all (s) | 6.960 | 3.952 | 3.622 | 2.030 | 1.952 |
|   approaching on green, MAE all (s) | 5.281 | 3.865 | 3.899 | 3.452 | 3.667 |
| Crossing vehicles, 80% interval coverage | 0.275 | 0.498 | 0.637 | 0.713 | 0.706 |
| All agents, Brier for entry within 5 s | 0.062 | 0.075 | 0.018 | 0.013 | 0.013 |
| Pedestrian entry-time MAE, all (s) | 8.782 | 4.705 | 3.975 | 0.872 | 0.670 |
| Conflict AUROC | 0.582 | 0.573 | 0.903 | 0.965 | 0.962 |
| Conflict AUPRC | 0.023 | 0.024 | 0.278 | 0.506 | 0.515 |
| Calibration error, raw | 0.139 | 0.154 | 0.029 | 0.009 | 0.009 |
| Calibration error, calibrated | 0.002 | 0.002 | 0.004 | 0.003 | 0.002 |
| Conflict events (vehicles) | 50 | 50 | 50 | 50 | 50 |
| Alert recall | 0.920 | 0.800 | 0.780 | 0.780 | 0.840 |
| Timely recall (lead >= TTA_min) | 0.720 | 0.540 | 0.520 | 0.540 | 0.640 |
| Alert precision | 0.054 | 0.058 | 0.260 | 0.639 | 0.583 |
| Alert F2 | 0.220 | 0.224 | 0.557 | 0.747 | 0.772 |
| Median lead time (s) | 17.500 | 7.800 | 2.800 | 5.500 | 9.150 |
| False-alert vehicles per hour | 1601.559 | 1306.904 | 222.494 | 44.098 | 60.134 |
| Alert threshold p* | 0.020 | 0.020 | 0.400 | 0.500 | 0.400 |
