# SA-V2X-Ego: vehicle-side, signal-aware trajectory and crosswalk-conflict prediction

A connected vehicle predicts its own trajectory and whether it will be in a crosswalk
at the same time as a pedestrian, using only V2X inputs:

| Input | Used for | Source message |
| --- | --- | --- |
| Ego kinematics, brake, turn signals, yaw rate, path radius | Trajectory, maneuver | The vehicle's own BSM data (10 Hz) |
| Pedestrian presence, counted per zone on and next to the crosswalk | Pedestrian side of the conflict | SDSM VRU objects (presence only, no tracks) |
| Phase state and timing per movement and for the pedestrian signal | Stop/go timing, WALK timing | SPaT (minEnd, maxEnd, likelyTime, confidence) |
| Lanes, connections, signal groups, crosswalk zones | Candidate routes, zones, signal binding | MAP |
| Nearest connected vehicle ahead (optional) | Queues | Other vehicles' BSMs |

The repository covers the data contract (JSON Schema), a SUMO scenario generator with a
V2X emulator that produces training data from simulated kinematics and pedestrians, the
model, training, baselines and evaluation.

## Quick start

```bash
pip install -r requirements.txt
bash tests/smoke_test.sh                                   # 3-minute end-to-end check
python scripts/make_dataset.py --out data_out --workers 2   # 39 scenario runs, ~45 min on 2 cores
bash scripts/run_experiments.sh                             # 6 models + 2 baselines, ~2 h on 2 cores
```

## Layout

| Path | What it does |
| --- | --- |
| `schemas/sample.vehicle.schema.json` | The training sample: ego BSM history, candidate routes, SPaT, SDSM pedestrian presence, leader BSM, labels |
| `schemas/extracted/*.schema.json` | Decoded BSM, SDSM, SPaT and MAP records |
| `schemas/common.defs.schema.json` | Shared types |
| `configs/scenarios.json` | Scenario catalog and train / val / test / held-out splits |
| `sim/scenario.py` | Network per geometry, signal program generated from the network, demand per scenario and seed |
| `sim/run_sim.py` | SUMO via TraCI; logs vehicles, pedestrians and controller timing at 10 Hz |
| `sim/export_map.py` | MAP export with crosswalk halves and pedestrian presence zones |
| `sim/emulate_v2x.py` | BSM for every vehicle, SDSM (VRU objects), SPaT with min/max/likely timing |
| `sav2x/samples.py` | Every simulated vehicle as an ego: received inputs and labels, written as schema JSON |
| `sav2x/features.py` | Sample JSON to arrays in the ego frame |
| `sav2x/model.py`, `sav2x/losses.py` | The network and its objectives |
| `sav2x/baselines.py` | Constant velocity, a SPaT kinematic rule, and the pedestrian-window alert rule |
| `sav2x/metrics.py`, `evaluate.py` | Trajectory, maneuver, arrival, occupancy, conflict and alert metrics |
| `train.py` | Training with input ablation flags |
| `scripts/` | Dataset build, experiment runner, schema validation, result tables, lead-time analysis |
| `runs/` | Trained checkpoints for the main model and the five ablations |
| `results/` | Metrics on the scenario dataset (see Results below) |
| `example_data/` | One MAP per geometry, the first records of each stream, 40 samples, one conflict sample pretty-printed |

## Data contract

One JSON object per decision time per vehicle (`.jsonl.gz`). Rules:

* Times are UTC epoch ms (`*_ms`) or seconds relative to t0 (`*_s`); positions are meters in the
  MAP refPoint frame; headings are J2735 (clockwise from north).
* Every input is what the vehicle had received by t0: SPaT, SDSM and other BSMs are sampled with
  distance-dependent loss and 20-80 ms latency. The ego's own BSM history has no loss.
* SDSM enters only as pedestrian counts in six zones (`wait_in`, `cw_1`..`cw_4`, `wait_out`) per
  history step. No pedestrian positions, tracks or IDs reach the model.
* `labels`, `signal_truth` and every `truth_id` are training targets or analysis fields, never inputs.
* Split by run (scenario and seed), never by sample.

Labels: the ego's future centers (8 s at 10 Hz), its maneuver, its entry into and exit from the
target crosswalk (20 s, censoring-aware), pedestrian occupancy of each crosswalk half per 0.5 s bin,
and a conflict flag: the ego and a pedestrian occupy the same half within theta = 1 s.

## Simulation scenarios

`configs/scenarios.json` defines nine scenarios on one signalized four-leg intersection
(two actuated phases, permissive turns, right on red, pedestrian clearance sized at crossing
length / 1.07 m/s). The target is the north crosswalk; vehicles from all four approaches are egos.

| Scenario | What changes |
| --- | --- |
| base | 450 veh/h per approach (+-30%), 150 ped/h per crossing, 8% aggressive drivers |
| peak | 650 veh/h |
| offpeak | 250 veh/h, faster drivers |
| ped_heavy | 400 ped/h |
| turn_heavy | 30% left, 30% right |
| aggressive | 30% aggressive drivers (ignore pedestrians with p = 0.3, run late yellows) |
| short_green | green 5-25 s instead of 7-40 s |
| slow_walkers (held out) | walking speed mean 0.95 m/s instead of 1.30 m/s |
| wide_3lane (held out) | three lanes per direction, 19.2 m crossing, 18 s clearance |

Seeds 1-3 train, 4 validates (calibration and alert threshold), 5 tests; the two held-out
scenarios (seeds 5 and 6) are never seen in training.

## Model

Everything is expressed in the ego frame at t0. The ego history and the presence history
each pass through a step MLP and a GRU. Each candidate route (left / straight / right from the
ego's approach, or the movements consistent with its position inside the box) is encoded from
its MAP path and bound to its signal group's SPaT token. A fusion MLP per route combines the
ego context, the route and the pedestrian signal. Heads:

* maneuver: one logit per legal route;
* trajectory: two modes per route, decoded along the route as progress from non-negative speeds
  (a stop at a red light is a plateau) plus a lateral offset, with Laplace scales;
* arrival: discrete-time hazards for entering and leaving the target crosswalk, per route;
* occupancy: pedestrian presence on each crosswalk half per 0.5 s bin;
* conflict: probability of a conflict, per route.

Per-route outputs are mixed with P(route). The conflict probability is calibrated with isotonic
regression on the validation runs; an alert fires when it exceeds a threshold chosen on validation
and the predicted crosswalk entry is within 12 s. An alert is timely if its lead time is at least
TTA_min(v) = 0.3 s + 1.5 s + v / (2 x 3.4 m/s^2).

Training conditions the trajectory loss on the true route (winner-takes-all among that route's modes);
P(route) is learned from the maneuver loss alone. Without this, stopped vehicles, for which every route
predicts the same 8 s trajectory, pushed P(route) toward arbitrary routes.

Ablations (`train.py`): `--signals likely` (state and likelyTime only, the I2XTraj / V2X-RECT
encodings), `--signals none`, `--no-presence`, `--no-turn-signals`, `--no-leader`.

## Results on the scenario dataset

Emulated data (SUMO plus the V2X emulator), so these numbers show the pipeline works and which inputs
matter; they are not field performance. In-distribution test: 7 scenarios x 20 min, 183,741 decisions,
224 vehicles with a conflict. Held-out: slow walkers and the 3-lane geometry, 4 runs, 124 conflict
vehicles. With about 200 events, a recall carries roughly +-0.06 of 95% uncertainty. Full tables:
`results/summary.md`, `results/lead_time.md`.

| Metric (in-distribution test) | Constant velocity | SPaT kinematic rule | SA-V2X-Ego |
| --- | --- | --- | --- |
| minFDE at 8 s, 6 modes (m) | 17.6 | 18.4 | 2.3 |
| approaching on red (m) | 13.2 | 9.8 | 1.0 |
| Maneuver accuracy before the stop line | | 89.8% | 90.8% |
| Crosswalk entry-time error, crossing vehicles (s) | 6.3 | 3.5 | 2.3 |
| 80% interval coverage of entry time | 26% | 56% | 89% |
| Conflict AUPRC (AUROC) | 0.03 (0.58) | 0.08 (0.80) | 0.59 (0.975) |
| Calibration error of raw conflict score | 0.15 | 0.17 | 0.008 |
| Alerts at the validation threshold: recall / in time / precision | 0.94 / 0.63 / 0.06 | 0.87 / 0.82 / 0.08 | 0.85 / 0.72 / 0.66 |
| False-alert vehicles per hour (share of non-conflict vehicles) | 1,313 (63%) | 967 (46%) | 42 (2.0%) |

On the held-out scenarios the model keeps its edge: conflict AUPRC 0.69, 80% of conflict vehicles
warned in time at 31 false-alert vehicles per hour, entry-time error 2.0 s, raw calibration error 0.006.

Ablations, in-distribution test:

| Model | minFDE 8 s on red (m) | Entry error on red (s) | Ped occupancy AUROC 10-20 s | Conflict AUPRC | In time at <= 60 false alerts/h | Lead >= 5 s at <= 120/h |
| --- | --- | --- | --- | --- | --- | --- |
| all inputs | 1.01 | 1.81 | 0.93 | 0.59 | 0.72 | 0.76 |
| SPaT state + likelyTime only | 1.02 | 1.84 | 0.93 | 0.58 | 0.79 | 0.74 |
| no SPaT | 1.79 | 4.37 | 0.72 | 0.58 | 0.79 | 0.77 |
| no SDSM presence | 1.01 | 1.75 | 0.93 | 0.52 | 0.63 | 0.68 |
| no turn signals | 1.00 | 2.25 | 0.93 | 0.53 | 0.65 | 0.55 |
| no leader BSM | 1.03 | 1.81 | 0.93 | 0.60 | 0.77 | not run |

What the numbers say:

* SPaT shapes the trajectory and the timing: without it, the 8 s error approaching on red nearly doubles
  and the crosswalk entry-time error grows from 1.8 to 4.4 s; pedestrian occupancy beyond 10 s becomes
  much harder to forecast. In these scenarios it does not improve alerting at matched false-alert rates,
  even for leads of 5-8 s (differences are within the +-0.06 uncertainty): pedestrian presence and the ego's own kinematics already tell the model most of
  what it needs once the vehicle is near the crosswalk.
* The interval SPaT encoding and likelyTime alone perform the same here, as expected when likelyTime is
  accurate; the test that matters is degraded SPaT (`--spat-profile min_max` or `min_only` in the emulator).
* Turn signals carry the maneuver: without them accuracy falls from 91% to 65% (turning vehicles 78% to 29%)
  and early warnings (lead >= 5 s) from 0.76 to 0.55.
* SDSM presence matters at low false-alert rates: without it AUPRC drops from 0.59 to 0.52.
* Other vehicles' BSMs (15% connected) add nothing measurable.
* The learned conflict head beats composing arrival and occupancy forecasts (AUPRC 0.59 against 0.51).
* Weak spot: the single most likely mode is often wrong about stop/go timing (8 s error 9.4 m against
  2.3 m for the best of 6 modes).

Limitations of the simulation: SUMO pedestrians never cross against the signal, drivers yield the same way
every time, and the emulated SPaT is always consistent with the controller. Field data will be harder.

## Moving to the MLK twin and field data

* Twin: run `sim/run_sim.py` on the MLK network (set `--tls-id`), and `sim/export_map.py --junction <id>
  --target-leg <leg>`. If the twin replays logged SPaT, write it in the SPaT schema instead of emulating it.
* Field: decode OBU logs (own BSM data) and received SDSM / SPaT into the extracted schemas; build MAP
  from the RSU's J2735 MAP plus derived connector paths and zones; future positions come from the
  vehicle's own later BSMs, and crosswalk occupancy and conflicts from camera annotation.
