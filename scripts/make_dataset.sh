#!/usr/bin/env bash
# Build the emulated dataset: SUMO runs -> V2X streams -> samples for both cases.
# Runs are split by "day": runs 1-4 train, 5 val, 6 test.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-sim_out}
VEH=(0 350 450 550 400 500 600)   # vehicles per hour per approach, by run

python3 sim/build_net.py --out "$OUT/net"
python3 sim/export_map.py --net "$OUT/net/x.net.xml" --tls "$OUT/net/tls.add.xml" --out "$OUT/map_C.json"

one_run() {
  i=$1
  python3 sim/build_net.py --out "$OUT/net" --demand-only --seed "$i" --veh-per-hour "${VEH[$i]}"
  python3 sim/run_sim.py --net "$OUT/net/x.net.xml" --tls "$OUT/net/tls.add.xml" \
      --routes "$OUT/net/demand_s$i.rou.xml" --seed "$i" --end 2100 --warmup 300 --out "$OUT/runs/run$i"
  python3 sim/emulate_v2x.py --run "$OUT/runs/run$i" --map "$OUT/map_C.json" --day "$((i - 1))" --seed "$i"
  python3 data/build_samples.py --run "$OUT/runs/run$i" --map "$OUT/map_C.json" --case infra \
      --assoc estimated --stride 1.0 --out "$OUT/samples/run${i}_infra.jsonl.gz"
  python3 data/build_samples.py --run "$OUT/runs/run$i" --map "$OUT/map_C.json" --case vehicle \
      --assoc estimated --stride 2.0 --out "$OUT/samples/run${i}_vehicle.jsonl.gz"
}

for i in 1 2 3 4 5 6; do one_run "$i" > "$OUT/log_run$i.txt" 2>&1 &
  if (( i % 2 == 0 )); then wait; fi
done
wait
# oracle-association copy of the test run, for the association-error study
python3 data/build_samples.py --run "$OUT/runs/run6" --map "$OUT/map_C.json" --case infra \
    --assoc ground_truth --stride 1.0 --out "$OUT/samples/run6_infra_gtassoc.jsonl.gz"
echo done
