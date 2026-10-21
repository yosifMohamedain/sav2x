"""Validate message streams, MAP and samples against the JSON schemas.

Usage:
    python scripts/validate.py --schemas schemas --map sim_out/map_C.json \
        --stream sim_out/runs/run1/sdsm.jsonl.gz:sdsm --sample sim_out/samples/run1_infra.jsonl.gz:sample.infra
"""
import argparse
import gzip
import itertools
import json
import os

from jsonschema import Draft202012Validator
from referencing import Registry, Resource


def load_registry(root):
    reg = Registry()
    schemas = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".schema.json"):
                s = json.load(open(os.path.join(dirpath, f)))
                reg = reg.with_resource(s["$id"], Resource.from_contents(s))
                schemas[f.replace(".schema.json", "")] = s
    return reg, schemas


def check(validator, records, label):
    n = bad = 0
    first = None
    for r in records:
        n += 1
        errs = sorted(validator.iter_errors(r), key=lambda e: e.path)
        if errs:
            bad += 1
            if first is None:
                e = errs[0]
                first = f"{list(e.absolute_path)}: {e.message[:200]}"
    print(f"{label}: {n - bad}/{n} valid" + (f"  first error -> {first}" if first else ""))
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", default="schemas")
    ap.add_argument("--map")
    ap.add_argument("--stream", action="append", default=[], help="path:schema_name")
    ap.add_argument("--sample", action="append", default=[], help="path:schema_name")
    ap.add_argument("--limit", type=int, default=500)
    a = ap.parse_args()
    reg, schemas = load_registry(a.schemas)
    ok = True
    for name, s in schemas.items():
        Draft202012Validator.check_schema(s)
    print(f"{len(schemas)} schemas are valid JSON Schema 2020-12")
    if a.map:
        v = Draft202012Validator(schemas["map"], registry=reg)
        ok &= check(v, [json.load(open(a.map))], f"map {a.map}")
    for spec in a.stream + a.sample:
        path, name = spec.rsplit(":", 1)
        v = Draft202012Validator(schemas[name], registry=reg)
        with gzip.open(path, "rt") as f:
            recs = (json.loads(l) for l in itertools.islice(f, a.limit))
            ok &= check(v, recs, f"{name} {path}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
