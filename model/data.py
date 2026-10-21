"""Dataset, caching and collation for SA-V2X samples."""
import gzip
import hashlib
import json
import os
import pickle

import numpy as np
import torch

from data.geom import IntersectionMap
from model.features import FeatureConfig, build_map_tokens, featurize


def map_tensors(map_path, device="cpu"):
    imap = IntersectionMap(map_path)
    mt = build_map_tokens(imap)
    t = {k: torch.as_tensor(v, device=device) for k, v in mt.items()
         if isinstance(v, np.ndarray)}
    t["target_tok"] = int(mt["target_tok"])
    t["target_sg_idx"] = int(mt["target_sg_idx"])
    t["sg_order"] = mt["sg_order"]
    return imap, mt, t


def load_samples(paths, mt, cfg: FeatureConfig, cache_dir=None, limit=None):
    key = hashlib.md5(json.dumps([sorted(paths), cfg.__dict__, limit]).encode()).hexdigest()[:12]
    cache = os.path.join(cache_dir, f"feat_{key}.pkl") if cache_dir else None
    if cache and os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    out = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                s = json.loads(line)
                if s["case"] == "vehicle" and s.get("signals") is None:
                    s["signals"] = None
                out.append(featurize(s, mt, cfg))
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        pickle.dump(out, open(cache, "wb"))
    return out


ARR_KEYS = ["hist", "bsm", "attr", "pose", "fut", "fmask", "man", "surv", "crosses", "appr", "weight",
            "valid", "is_vru"]


def collate(items):
    B = len(items)
    A = max(len(it["ids"]) for it in items)
    b = {}
    for k in ARR_KEYS:
        x0 = items[0][k]
        shape = (B, A) + x0.shape[1:]
        fill = -1 if k in ("man", "appr") else 0
        arr = np.full(shape, fill, dtype=x0.dtype)
        for i, it in enumerate(items):
            arr[i, :len(it[k])] = it[k]
        b[k] = torch.as_tensor(arr)
    b["amask"] = b["valid"].bool()
    b["sig"] = torch.as_tensor(np.stack([it["sig"] for it in items]))
    b["sig_mask"] = torch.as_tensor(np.stack([it["sig_mask"] for it in items]))
    b["meta"] = [{"ids": it["ids"], "sample_id": it["sample_id"], "t0_ms": it["t0_ms"],
                  "conflicts": it["conflicts"], "case": it["case"]} for it in items]
    return b


class Batches:
    def __init__(self, items, batch_size, shuffle, seed=0):
        self.items, self.bs, self.shuffle = items, batch_size, shuffle
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = np.arange(len(self.items))
        if self.shuffle:
            self.rng.shuffle(idx)
        for i in range(0, len(idx), self.bs):
            yield collate([self.items[j] for j in idx[i:i + self.bs]])

    def __len__(self):
        return (len(self.items) + self.bs - 1) // self.bs
