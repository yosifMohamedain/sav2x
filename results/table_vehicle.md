| Metric | cv | cv_spat | infra_interval | vehicle_interval |
| --- | --- | --- | --- | --- |
| Vehicle before stop line, minADE6 @5 s (m) | 3.186 | 5.103 | 0.925 | 0.751 |
| Vehicle before stop line, minFDE6 @8 s (m) | 17.554 | 26.699 | 2.635 | 2.344 |
| Vehicle before stop line, top-1 FDE @8 s (m) | 17.554 | 26.699 | 7.694 | 7.976 |
|   approaching on red, minFDE6 @8 s (m) | 12.502 | 23.315 | 1.052 | 0.955 |
|   approaching on green, minFDE6 @8 s (m) | 24.055 | 31.053 | 4.672 | 4.132 |
| All agents, ISS top-3 weighted ADE @5 s (m) | 3.284 | 4.667 | 1.768 | 1.621 |
| All agents, miss rate @8 s, 2 m | 0.626 | 0.739 | 0.295 | 0.284 |
| Maneuver accuracy |  |  | 0.952 | 0.971 |
| Maneuver accuracy, turning vehicles |  |  | 0.904 | 0.962 |
| Maneuver accuracy, 0-20 m to stop line |  |  | 0.964 | 0.990 |
| Maneuver accuracy, 50+ m to stop line |  |  | 0.881 | 0.897 |
| Crossing vehicles, share predicted to enter (P >= 0.5) | 0.472 | 0.899 | 0.877 | 0.920 |
| Crossing vehicles, entry-time MAE, all (s) | 6.001 | 4.244 | 2.784 | 2.528 |
| Crossing vehicles, entry-time MAE, detected only (s) | 2.562 | 3.548 | 2.322 | 2.267 |
|   approaching on red, MAE all (s) | 7.137 | 4.128 | 2.388 | 2.072 |
|   approaching on green, MAE all (s) | 5.313 | 3.625 | 2.998 | 2.731 |
| Crossing vehicles, 80% interval coverage | 0.264 | 0.479 | 0.752 | 0.761 |
| All agents, Brier for entry within 5 s | 0.049 | 0.070 | 0.019 | 0.018 |
| Conflict AUROC | 0.591 | 0.853 | 0.977 | 0.978 |
| Conflict AUPRC | 0.047 | 0.154 | 0.527 | 0.569 |
| Calibration error, raw | 0.116 | 0.140 | 0.014 | 0.013 |
| Calibration error, calibrated | 0.008 | 0.009 | 0.016 | 0.019 |
| Conflict events (vehicles) | 7 | 7 | 7 | 7 |
| Alert recall | 0.429 | 1.000 | 0.857 | 0.571 |
| Timely recall (lead >= TTA_min) | 0.429 | 1.000 | 0.714 | 0.571 |
| Alert precision | 0.070 | 0.175 | 0.600 | 0.667 |
| Alert F2 | 0.211 | 0.515 | 0.789 | 0.588 |
| Median lead time (s) | 12.000 | 34.900 | 7.450 | 8.900 |
| False-alert vehicles per hour | 39.956 | 32.963 | 3.996 | 1.998 |
| Alert threshold p* | 0.050 | 0.050 | 0.700 | 0.600 |
