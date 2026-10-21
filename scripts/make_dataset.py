"""Build the scenario dataset: for every (scenario, seed) in the catalog's splits,
prepare SUMO inputs, simulate, export MAP, emulate V2X streams, and build samples.

    python scripts/make_dataset.py --catalog configs/scenarios.json --out data_out --workers 2

Writes data_out/runs/<scenario>_s<seed>/ (simulation and streams),
data_out/samples/<scenario>_s<seed>.jsonl.gz (training samples), and
data_out/manifest.json (split of every run, sample counts). Finished runs are skipped,
so the script can be re-run after an interruption.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs_from_catalog(cat):
    sp = cat["splits"]
    out = []
    for sc in sp["train_scenarios"]:
        for s in sp["train_seeds"]:
            out.append((sc, s, "train"))
        for s in sp["val_seeds"]:
            out.append((sc, s, "val"))
        for s in sp["test_seeds"]:
            out.append((sc, s, "test"))
    for sc in sp["heldout_scenarios"]:
        for s in sp["heldout_seeds"]:
            out.append((sc, s, "heldout"))
    return out


def one(catalog, out, sc, seed, extra):
    run = os.path.join(out, "runs", f"{sc}_s{seed}")
    smp = os.path.join(out, "samples", f"{sc}_s{seed}.jsonl.gz")
    if os.path.exists(smp) and os.path.getsize(smp) > 0:
        return sc, seed, "skipped"
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=ROOT)
    t = time.time()
    steps = [[py, "sim/scenario.py", "--catalog", catalog, "--scenario", sc, "--seed", str(seed), "--out", run,
              "--net-cache", os.path.join(out, "nets")],
             [py, "sim/run_sim.py", "--run", run],
             [py, "sim/export_map.py", "--run", run],
             [py, "sim/emulate_v2x.py", "--run", run, "--bsm-jsonl", "100"],
             [py, "-m", "sav2x.samples", "--run", run, "--out", smp + ".tmp"] + extra]
    log = open(os.path.join(out, "logs", f"{sc}_s{seed}.log"), "w")
    for cmd in steps:
        r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            return sc, seed, f"failed at {cmd[1]}"
    os.replace(smp + ".tmp", smp)
    return sc, seed, f"ok in {time.time() - t:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="configs/scenarios.json")
    ap.add_argument("--out", default="data_out")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these scenario names")
    ap.add_argument("--samples-args", default="", help="extra arguments for sav2x.samples, quoted")
    a = ap.parse_args()
    cat = json.load(open(os.path.join(ROOT, a.catalog)))
    runs = runs_from_catalog(cat)
    if a.only:
        runs = [r for r in runs if r[0] in a.only]
    for d in ("runs", "samples", "logs", "nets"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    extra = a.samples_args.split() if a.samples_args else []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, a.catalog, a.out, sc, s, extra): (sc, s, split) for sc, s, split in runs}
        for f in as_completed(futs):
            sc, s, msg = f.result()
            print(f"{sc} s{s}: {msg}", flush=True)
    manifest = []
    for sc, s, split in runs:
        smp = os.path.join(a.out, "samples", f"{sc}_s{s}.jsonl.gz")
        n = None
        if os.path.exists(smp):
            import gzip
            with gzip.open(smp, "rt") as f:
                n = sum(1 for _ in f)
        manifest.append({"scenario": sc, "seed": s, "split": split, "samples": smp, "n": n})
    json.dump({"catalog": a.catalog, "runs": manifest}, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("manifest:", {sp: sum(r["n"] or 0 for r in manifest if r["split"] == sp) for sp in ("train", "val", "test", "heldout")})


if __name__ == "__main__":
    main()
