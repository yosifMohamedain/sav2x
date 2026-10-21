#!/usr/bin/env bash
# Train the main model and the input ablations with one schedule, then evaluate all of
# them and the baselines on the in-distribution test runs and the held-out scenarios.
# Two trainings run at a time (one thread each).
set -euo pipefail
cd "$(dirname "$0")/.."
M=${MANIFEST:-data_out/manifest.json}
COMMON="--manifest $M --epochs ${EPOCHS:-10} --subsample ${SUBSAMPLE:-0.5} --threads 1"
mkdir -p runs results

train() { python train.py $COMMON --out "runs/$1" "${@:2}" > "runs/$1.log" 2>&1; }

# warm the feature cache once, so parallel jobs do not featurize the same files
python -c "
from concurrent.futures import ProcessPoolExecutor
from sav2x.data import _cache_one, manifest_paths
ps = manifest_paths('$M', ['train', 'val', 'test', 'heldout'])
with ProcessPoolExecutor(2) as ex:
    list(ex.map(_cache_one, [(p, 'data_out/cache', False) for p in ps]))
"

train main &               train likely --signals likely &        wait
train no_spat --signals none & train no_presence --no-presence &  wait
train no_turn_signals --no-turn-signals & train no_leader --no-leader & wait

CK="runs/main/best.pt runs/likely/best.pt runs/no_spat/best.pt runs/no_presence/best.pt runs/no_turn_signals/best.pt runs/no_leader/best.pt"
python evaluate.py --manifest $M --split test heldout --ckpt $CK --baselines cv rule --out results
python scripts/summarize.py results/test/results.json results/heldout/results.json > results/summary.md
echo done
