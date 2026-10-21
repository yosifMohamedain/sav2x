"""Train SA-V2X-Ego on vehicle-side samples.

    python train.py --manifest data_out/manifest.json --out runs/main
    python train.py --manifest data_out/manifest.json --signals likely --out runs/likely
    python train.py --manifest data_out/manifest.json --no-presence --out runs/no_presence
    python train.py --train a.jsonl.gz b.jsonl.gz --val c.jsonl.gz --out runs/x     # explicit files

Inputs are switched off with --signals {interval,likely,none}, --no-presence,
--no-turn-signals, --no-leader; the same featurized cache serves every variant.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from sav2x.data import InputMask, batches, load, manifest_paths
from sav2x.losses import total_loss
from sav2x.model import EgoNet


def run_epoch(model, d, mask, bs, opt=None, sched=None, seed=0, idx=None):
    train = opt is not None
    model.train(train)
    tot, n = {}, 0
    for b in batches(d, bs, mask, shuffle=train, seed=seed, idx=idx):
        with torch.set_grad_enabled(train):
            out = model(b)
            loss, parts = total_loss(out, b)
        if train:
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        for k, v in {"loss": loss, **parts}.items():
            tot[k] = tot.get(k, 0.0) + float(v.detach())
        n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--train", nargs="*", default=None)
    ap.add_argument("--val", nargs="*", default=None)
    ap.add_argument("--signals", default="interval", choices=["interval", "likely", "none"])
    ap.add_argument("--no-presence", action="store_true")
    ap.add_argument("--no-turn-signals", action="store_true")
    ap.add_argument("--no-leader", action="store_true")
    ap.add_argument("--oracle", action="store_true", help="signal features from the realized timing (sim upper bound)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--cache", default="data_out/cache")
    ap.add_argument("--subsample", type=float, default=1.0, help="fraction of training samples per epoch")
    ap.add_argument("--val-max", type=int, default=40000, help="validation samples used for checkpoint selection")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    os.makedirs(a.out, exist_ok=True)
    tr_paths = a.train or manifest_paths(a.manifest, ["train"])
    va_paths = a.val or manifest_paths(a.manifest, ["val"])
    t0 = time.time()
    tr = load(tr_paths, a.cache, oracle=a.oracle)
    va = load(va_paths, a.cache, oracle=a.oracle)
    if len(va["man"]) > a.val_max:                 # fixed random subset for checkpoint selection
        keep = np.sort(np.random.default_rng(1).choice(len(va["man"]), a.val_max, replace=False))
        va = {k: v[keep] for k, v in va.items()}
    print(f"{len(tr['man'])} train / {len(va['man'])} val samples (selection subset) from {len(tr_paths)} / {len(va_paths)} runs "
          f"({time.time() - t0:.0f}s)", flush=True)
    mask = InputMask(a.signals, not a.no_presence, not a.no_turn_signals, not a.no_leader)
    model = EgoNet(d=a.d)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M; inputs {mask.to_dict()}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    n_tr = len(tr["man"])
    steps = a.epochs * math.ceil(n_tr * a.subsample / a.bs)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(k, steps) / steps))) + 0.01)
    best, hist = 1e9, []
    rng = np.random.default_rng(0)
    for ep in range(a.epochs):
        t1 = time.time()
        sel = rng.choice(n_tr, int(n_tr * a.subsample), replace=False) if a.subsample < 1.0 else None
        r_tr = run_epoch(model, tr, mask, a.bs, opt, sched, seed=ep, idx=sel)
        r_va = run_epoch(model, va, mask, 1024)
        hist.append({"epoch": ep, "train": r_tr, "val": r_va, "sec": time.time() - t1})
        print(f"ep {ep:2d} train {r_tr['loss']:.3f} ade {r_tr['ade']:.2f} | val {r_va['loss']:.3f} ade {r_va['ade']:.2f} "
              f"man {r_va['man']:.3f} entry {r_va['entry']:.3f} occ {r_va['occ']:.3f} conf {r_va['conf']:.3f} "
              f"| {time.time() - t1:.0f}s", flush=True)
        if r_va["loss"] < best:
            best = r_va["loss"]
            torch.save({"model": model.state_dict(), "args": vars(a), "mask": mask.to_dict(), "d": a.d},
                       os.path.join(a.out, "best.pt"))
    json.dump(hist, open(os.path.join(a.out, "history.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
