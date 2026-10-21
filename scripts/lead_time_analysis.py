"""How early can each model warn? Recall with lead >= L seconds at matched false-alert rates.

Uses only the cached features (no JSON parsing), so it runs in a minute or two:
    python scripts/lead_time_analysis.py --ckpt runs/main/best.pt runs/no_spat/best.pt --split test heldout
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sav2x.data import InputMask, batch, load, manifest_paths  # noqa: E402
from sav2x.metrics import Calibrator, alert_events  # noqa: E402
from sav2x.model import EgoNet, predict  # noqa: E402


def scores(ck, f):
    c = torch.load(ck, weights_only=False)
    m = EgoNet(d=c.get("d", 96))
    m.load_state_dict(c["model"])
    m.eval()
    mask = InputMask(**c["mask"])
    sc, pe = [], []
    for i in range(0, len(f["man"]), 4096):
        o = predict(m, batch(f, np.arange(i, min(len(f["man"]), i + 4096)), mask))
        p = o["entry_pmf"].numpy()
        e = o["p_enter"].numpy()
        cn = np.cumsum(p, 1) / np.maximum(e, 1e-9)[:, None]
        pe.append(np.where(e >= 0.05, (np.argmax(cn >= 0.5, 1) + 0.5) * 0.5, np.inf))
        sc.append(o["conflict"].numpy())
    return np.concatenate(sc), np.concatenate(pe)


def decisions(f, sc, pe):
    return {"ep": np.unique(f["episode"], return_inverse=True)[1], "t0": f["t0"], "score": sc, "pred_entry": pe,
            "y": (f["conf"] > 0.5).astype(int), "t_conf": f["t0"] + np.maximum(f["conf_entry"], 0),
            "speed": f["speed"].astype(float), "known": f["conf_known"] > 0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_out/manifest.json")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--split", nargs="+", default=["test"])
    ap.add_argument("--logged-s", type=float, default=1200.0)
    ap.add_argument("--caps", nargs="+", type=float, default=[30, 60, 120])
    ap.add_argument("--leads", nargs="+", type=float, default=[0, 3, 5, 8])
    ap.add_argument("--out", default="results/lead_time.json")
    a = ap.parse_args()
    torch.set_num_threads(2)
    fv = load(manifest_paths(a.manifest, ["val"]))
    thr = np.unique(np.concatenate([np.geomspace(1e-3, 0.98, 60)]))
    res = {}
    for ck in a.ckpt:
        name = os.path.basename(os.path.dirname(ck))
        sv, pv = scores(ck, fv)
        Dv = decisions(fv, sv, pv)
        cal = Calibrator().fit(sv[Dv["known"]], Dv["y"][Dv["known"]])
        res[name] = {}
        for split in a.split:
            paths = manifest_paths(a.manifest, [split])
            f = load(paths)
            s, p = scores(ck, f)
            D = decisions(f, cal(s), p)
            hours = len(paths) * a.logged_s / 3600
            tab = {}
            for L in a.leads:
                curve = [alert_events(D, t, hours, min_lead=L) for t in thr]
                tab[f"lead>={L:g}s"] = {f"<= {c:g}/h": max((x["timely_recall"] for x in curve
                                                            if x["false_alert_vehicles_per_hour"] <= c), default=0.0)
                                         for c in a.caps}
            res[name][split] = tab
            print(name, split, json.dumps(tab))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
