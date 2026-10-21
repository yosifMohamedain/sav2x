"""Feature caching, input masking for ablations, and batching."""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from sav2x.features import ARRAY_KEYS, INTERVAL_COLS, META_KEYS, featurize_file

TENSOR_KEYS = ["hist", "pres", "ctx", "path", "rfeat", "rmask", "crosses", "sig", "smask", "fut", "fmask", "man",
               "surv", "inside", "occ", "omask", "conf", "conf_known"]


def _cache_one(args):
    path, cache_dir, oracle = args
    os.makedirs(cache_dir, exist_ok=True)
    name = os.path.basename(path).replace(".jsonl.gz", "") + (".oracle" if oracle else "") + ".npz"
    out = os.path.join(cache_dir, name)
    if not (os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path)):
        d = featurize_file(path, oracle)
        np.savez(out, **{k: (v.astype(str) if k in META_KEYS else v) for k, v in d.items()})
    return out


def load(paths, cache_dir="data_out/cache", workers=2, oracle=False):
    """Featurize (cached) and concatenate several sample files."""
    os.makedirs(cache_dir, exist_ok=True)
    with ProcessPoolExecutor(workers) as ex:
        files = list(ex.map(_cache_one, [(p, cache_dir, oracle) for p in paths]))
    parts = [dict(np.load(f)) for f in files]
    out = {}
    for k in list(parts[0]):                 # concatenate key by key, freeing parts as we go
        out[k] = np.concatenate([p[k] for p in parts])
        for p in parts:
            del p[k]
    return out


class InputMask:
    """Switch off input groups at batch time.
    signals: 'interval' (all), 'likely' (state + likelyTime only), 'none'."""

    def __init__(self, signals="interval", presence=True, turn_signals=True, leader=True):
        self.signals, self.presence, self.turn_signals, self.leader = signals, presence, turn_signals, leader

    def apply(self, b):
        if self.signals == "none":
            b["sig"] = torch.zeros_like(b["sig"])
            b["smask"] = torch.zeros_like(b["smask"])
        elif self.signals == "likely":
            b["sig"][..., INTERVAL_COLS] = 0
        if not self.presence:
            b["pres"] = torch.zeros_like(b["pres"])
            b["ctx"][:, 8:12] = 0
        if not self.turn_signals:
            b["hist"][..., 10:12] = 0
        if not self.leader:
            b["ctx"][:, 12:17] = 0
        return b

    def to_dict(self):
        return dict(signals=self.signals, presence=self.presence, turn_signals=self.turn_signals, leader=self.leader)


def batch(d, idx, mask: InputMask):
    b = {}
    for k in TENSOR_KEYS:
        v = d[k][idx]
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        b[k] = torch.from_numpy(np.ascontiguousarray(v))
    return mask.apply(b)


def batches(d, bs, mask, shuffle=False, seed=0, idx=None):
    idx = np.arange(len(d["man"])) if idx is None else np.asarray(idx).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield batch(d, idx[i:i + bs], mask)


def manifest_paths(manifest, splits, scenarios=None):
    m = json.load(open(manifest))
    return [r["samples"] for r in m["runs"] if r["split"] in splits and (scenarios is None or r["scenario"] in scenarios)
            and r["n"]]
