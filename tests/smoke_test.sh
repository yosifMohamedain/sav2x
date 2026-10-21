#!/usr/bin/env bash
# End-to-end smoke test on a 3-minute simulation: scenario -> SUMO -> MAP -> V2X streams ->
# samples -> schema validation -> 1 training epoch -> evaluation. About 3 minutes on 2 cores.
set -euo pipefail
cd "$(dirname "$0")/.."
O=${1:-smoke_out}
rm -rf "$O" && mkdir -p "$O/runs" "$O/samples"
for s in 1 2 3; do
  R=$O/runs/base_s$s
  python sim/scenario.py --scenario base --seed $s --out $R --net-cache $O/nets --set warmup_s=120 logged_s=180
  python sim/run_sim.py --run $R
  python sim/export_map.py --run $R
  python sim/emulate_v2x.py --run $R --bsm-jsonl 50
  python -m sav2x.samples --run $R --out $O/samples/base_s$s.jsonl.gz
done
python scripts/validate.py --map $O/runs/base_s1/map.json \
  --stream $O/runs/base_s1/sdsm.jsonl.gz:sdsm --stream $O/runs/base_s1/spat.jsonl.gz:spat \
  --stream $O/runs/base_s1/bsm_sample.jsonl.gz:bsm --sample $O/samples/base_s1.jsonl.gz:sample.vehicle --limit 200
python - "$O" <<'PY'
import json, sys
o = sys.argv[1]
runs = [{"scenario": "base", "seed": s, "split": sp, "samples": f"{o}/samples/base_s{s}.jsonl.gz", "n": 1}
        for s, sp in ((1, "train"), (2, "val"), (3, "test"))]
json.dump({"runs": runs}, open(f"{o}/manifest.json", "w"))
PY
python train.py --manifest $O/manifest.json --epochs 1 --cache $O/cache --out $O/model
python evaluate.py --manifest $O/manifest.json --split test --ckpt $O/model/best.pt --baselines cv rule \
  --cache $O/cache --logged-s 180 --out $O/results
python scripts/summarize.py $O/results/test/results.json > $O/results/summary.md
sed -n '1,14p' $O/results/summary.md
echo "smoke test passed"
