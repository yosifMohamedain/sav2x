"""Evaluate SA-V2X-Ego checkpoints and the baselines.

    python evaluate.py --manifest data_out/manifest.json --split test heldout \
        --ckpt runs/main/best.pt runs/likely/best.pt --baselines cv rule --out results

Calibration (isotonic) and the alert threshold are fit on the validation runs and applied
unchanged to each evaluated split. Results go to <out>/<split>/results.json.
"""
import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
from scipy.stats import norm

from sav2x.baselines import Baselines
from sav2x.data import InputMask, batch, load, manifest_paths
from sav2x.features import MANS, NB
from sav2x.metrics import (Calibrator, alert_events, binary_metrics, choose_threshold, operating_curve,
                           reliability, slice_means, traj_errors)
from sav2x.model import EgoNet, predict

RED = {"stop-And-Remain", "permissive-clearance", "protected-clearance", "stop-Then-Proceed"}
TAUS = 0.5 * np.arange(1, NB + 1)


# ------------------------------------------------------------------ reading samples
def _read_one(args):
    path, maps, bl_modes = args
    bls = Baselines(maps)
    G = {k: [] for k in ("episode", "scenario", "t0", "speed", "role", "dist", "man", "crosses", "light", "fut",
                         "fmask", "inside", "entry", "exit", "censor", "y", "known", "t_conf")}
    B = {m: {k: [] for k in ("traj", "tin", "score", "maneuver")} for m in bl_modes}
    with gzip.open(path, "rt") as f:
        for line in f:
            s = json.loads(line)
            L = s["labels"]
            fut = np.asarray(L["fut"], dtype=np.float32)[1::2][:40]
            sig = {g["signal_group"]: g for g in (s["signals"] or {}).get("groups", [])}
            tr = [r for r in s["routes"] if r["maneuver"] == L["maneuver"]]
            mm = s["ego"]["map_match"]
            light = ""
            if tr and tr[0]["signal_group"] in sig and mm["role"] == "ingress":
                light = "red" if sig[tr[0]["signal_group"]]["state"] in RED else "green"
            cw, cf = L["cw"], L["conflict"]
            t0 = s["t0_ms"] / 1000.0
            for k, v in (("episode", f"{s['run_id']}|{L['truth_id']}"), ("scenario", s["scenario"]), ("t0", t0),
                         ("speed", [r for r in s["ego"]["hist"] if r[11]][-1][2]), ("role", mm["role"]),
                         ("dist", mm["dist_to_stopline_m"] if mm["dist_to_stopline_m"] is not None else np.nan),
                         ("man", L["maneuver"] or ""), ("crosses", L["crosses_target"]), ("light", light),
                         ("fut", fut[:, :2]), ("fmask", fut[:, 3] > 0.5), ("inside", cw["inside_at_t0"]),
                         ("entry", np.nan if cw["entry_s"] is None else cw["entry_s"]),
                         ("exit", np.nan if cw["exit_s"] is None else cw["exit_s"]), ("censor", cw["censor_s"]),
                         ("y", int(cf["label"])), ("known", cf["known"]), ("t_conf", t0 + (cf["ego_entry_s"] or 0.0))):
                G[k].append(v)
            for m in bl_modes:
                o = bls.predict(s, m)
                B[m]["traj"].append(o["traj"][0].astype(np.float32))
                B[m]["tin"].append(o["tin"])
                B[m]["score"].append(o["score"])
                B[m]["maneuver"].append(o["maneuver"] or "")
    G = {k: np.asarray(v) for k, v in G.items()}
    B = {m: {k: np.asarray(v) for k, v in d.items()} for m, d in B.items()}
    return G, B


def read_split(paths, maps, bl_modes, workers=2):
    jobs = [(p, {k: v for k, v in maps.items() if os.path.basename(p).startswith(k + ".jsonl")}, bl_modes)
            for p in paths]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_read_one, jobs))
    G = {k: np.concatenate([p[0][k] for p in parts]) for k in parts[0][0]}
    B = {m: {k: np.concatenate([p[1][m][k] for p in parts]) for k in parts[0][1][m]} for m in bl_modes}
    return G, B


# ------------------------------------------------------------------ predictor outputs
def model_arrays(ckpt, feats, bs=2048):
    ck = torch.load(ckpt, weights_only=False)
    model = EgoNet(d=ck.get("d", 96))
    model.load_state_dict(ck["model"])
    model.eval()
    mask = InputMask(**ck["mask"])
    n = len(feats["man"])
    out = {k: [] for k in ("p_route", "pmf", "p_enter", "score", "composed", "occ")}
    terr = []
    for i in range(0, n, bs):
        idx = np.arange(i, min(n, i + bs))
        b = batch(feats, idx, mask)
        o = predict(model, b)
        terr.append(traj_errors(o["loc"].numpy(), torch.softmax(o["logp"], -1).numpy(),
                                feats["fut"][idx], feats["fmask"][idx]))
        out["p_route"].append(o["p_route"].numpy())
        out["pmf"].append(o["entry_pmf"].numpy())
        out["p_enter"].append(o["p_enter"].numpy())
        out["score"].append(o["conflict"].numpy())
        out["occ"].append(o["occ"].numpy().astype(np.float16))
        win = torch.nn.functional.max_pool1d(o["occ"], kernel_size=9, stride=1, padding=4)        # +-2 s
        rz = torch.from_numpy(feats["rzone"][idx]).float()
        q = (rz.unsqueeze(-1) * win.unsqueeze(1)).max(2).values
        out["composed"].append((o["p_route"] * (o["pmf_r"] * q).sum(-1) * b["crosses"]).sum(-1).numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["terr"] = {k: np.concatenate([t[k] for t in terr]) for k in terr[0]}
    p_route, pmf, pe = res["p_route"], res["pmf"], res["p_enter"]
    y = feats["man"]
    ok = y >= 0
    res["man_acc"] = np.where(ok, (p_route.argmax(1) == np.maximum(y, 0)).astype(float), np.nan)
    res["man_nll"] = np.where(ok, -np.log(p_route[np.arange(n), np.maximum(y, 0)] + 1e-9), np.nan)
    c = np.cumsum(pmf, 1)
    cn = c / np.maximum(pe, 1e-9)[:, None]
    med = (np.argmax(cn >= 0.5, 1) + 0.5) * 0.5
    res["pred_entry"] = np.where(pe >= 0.05, med, np.inf)          # alert window: conditional median
    res["entry_point"] = np.where(pe >= 0.5, med, np.inf)          # arrival error: must predict an entry
    res["cdf"] = c.astype(np.float32)
    return res, ck


def baseline_arrays(B, G):
    n = len(B["tin"])
    terr = traj_errors(B["traj"][:, None], np.ones((n, 1)), G["fut"], G["fmask"])
    tin = B["tin"].astype(float)
    fin = np.isfinite(tin)
    cdf = np.zeros((n, NB), dtype=np.float32)
    sig = 0.3 + 0.15 * np.where(fin, tin, 0)
    cdf[fin] = norm.cdf(TAUS[None], loc=tin[fin, None], scale=sig[fin, None])
    man_acc = np.where((B["maneuver"] != "") & (G["man"] != ""), (B["maneuver"] == G["man"]).astype(float), np.nan)
    return {"terr": terr, "man_acc": man_acc, "man_nll": np.full(n, np.nan), "pred_entry": tin, "entry_point": tin,
            "cdf": cdf, "score": B["score"].astype(float)}


# ------------------------------------------------------------------ metric blocks
def traj_block(G, P):
    masks = {"all": np.ones(len(G["t0"]), bool), "role_ingress": G["role"] == "ingress",
             "role_connector": G["role"] == "connector",
             "ingress_red": G["light"] == "red", "ingress_green": G["light"] == "green"}
    for m in MANS:
        masks[f"man_{m}"] = G["man"] == m
    return slice_means(P["terr"], masks)


def maneuver_block(G, P):
    ing = G["role"] == "ingress"
    d = G["dist"]
    masks = {"all": ing, "turning": ing & np.isin(G["man"], ["left", "right"]),
             "d0": ing & (d < 25), "d25": ing & (d >= 25) & (d < 50), "d50": ing & (d >= 50)}
    return slice_means({"man_acc": P["man_acc"], "man_nll": P["man_nll"]}, masks)


def arrival_block(G, P):
    res = {}
    base = ~G["inside"].astype(bool)
    cr = G["crosses"].astype(bool)
    groups = {"all": base, "crossing": base & cr, "not_crossing": base & ~cr,
              "crossing_red": base & cr & (G["light"] == "red"),
              "crossing_green": base & cr & (G["light"] == "green")}
    entry, cens, cdf = G["entry"].astype(float), G["censor"].astype(float), P["cdf"]
    for name, msk in groups.items():
        for tau in (3, 5, 8, 12):
            known = msk & (~np.isnan(entry) | (cens >= tau))
            y = (np.nan_to_num(entry, nan=1e9) <= tau).astype(float)
            p = cdf[:, int(tau / 0.5) - 1].astype(float)
            if known.any():
                res[f"{name}/brier@{tau}s"] = float(((p[known] - y[known]) ** 2).mean())
        ent = msk & ~np.isnan(entry) & (np.nan_to_num(entry) > 0)
        if ent.any():
            ep = P["entry_point"][ent]
            e = entry[ent]
            res[f"{name}/detect"] = float(np.isfinite(ep).mean())
            res[f"{name}/mae_all"] = float(np.abs(np.where(np.isfinite(ep), ep, 20.0) - e).mean())
            c = cdf[ent].astype(float)
            top = c[:, -1]
            cn = c / np.maximum(top, 1e-9)[:, None]
            lo = (np.argmax(cn >= 0.1, 1) + 0.5) * 0.5
            hi = (np.argmax(cn >= 0.9, 1) + 0.5) * 0.5
            res[f"{name}/cover80"] = float(((top > 1e-6) & (lo - 0.25 <= e) & (e <= hi + 0.25)).mean())
            res[f"{name}/n_entries"] = int(ent.sum())
    return res


def decision_arrays(G, P, key, ep_codes):
    return {"ep": ep_codes, "t0": G["t0"], "score": np.asarray(P[key], float), "pred_entry": P["pred_entry"],
            "y": G["y"], "t_conf": G["t_conf"], "speed": G["speed"].astype(float), "known": G["known"].astype(bool),
            "scenario": G["scenario"]}


def conflict_block(Dv, Dt, hours_val, hours_te, hours_by_sc):
    cal = Calibrator().fit(Dv["score"][Dv["known"]], Dv["y"][Dv["known"]])
    Dv = dict(Dv, raw=Dv["score"], score=cal(Dv["score"]))
    Dt = dict(Dt, raw=Dt["score"], score=cal(Dt["score"]))
    p_star = choose_threshold(Dv, hours_val)
    k = Dt["known"]
    res = {"risk_raw": binary_metrics(Dt["raw"][k], Dt["y"][k]), "risk_cal": binary_metrics(Dt["score"][k], Dt["y"][k]),
           "reliability": reliability(Dt["score"][k], Dt["y"][k]),
           "alerts": alert_events(Dt, p_star, hours_te), "alerts_no_window": alert_events(Dt, p_star, hours_te, window=False),
           "operating_curve": operating_curve(Dt, hours_te)}
    res["alerts_by_scenario"] = {sc: alert_events(Dt, p_star, h, sel=Dt["scenario"] == sc) for sc, h in hours_by_sc.items()}
    res["risk_by_scenario"] = {sc: binary_metrics(Dt["score"][k & (Dt["scenario"] == sc)], Dt["y"][k & (Dt["scenario"] == sc)])
                               for sc in hours_by_sc}
    return res


def occupancy_block(feats, occ):
    y, om = feats["occ"].astype(np.float32), feats["omask"].astype(np.float32) > 0
    p = occ.astype(np.float32)
    res = {}
    for name, sl in (("0-5s", slice(0, 10)), ("5-10s", slice(10, 20)), ("10-20s", slice(20, 40))):
        msk = np.broadcast_to(om[:, None, sl], y[:, :, sl].shape)
        res[name] = binary_metrics(p[:, :, sl][msk], y[:, :, sl][msk])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_out/manifest.json")
    ap.add_argument("--split", nargs="+", default=["test"], choices=["test", "heldout", "val", "train"])
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--baselines", nargs="*", default=["cv", "rule"])
    ap.add_argument("--cache", default="data_out/cache")
    ap.add_argument("--logged-s", type=float, default=1200.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    man = json.load(open(a.manifest))
    maps = {}
    for r in man["runs"]:
        rid = f"{r['scenario']}_s{r['seed']}"
        mp = os.path.join(os.path.dirname(a.manifest), "runs", rid, "map.json")
        if os.path.exists(mp):
            maps[rid] = json.load(open(mp))
    va_paths = manifest_paths(a.manifest, ["val"])
    hours_val = len(va_paths) * a.logged_s / 3600
    Gv, Bv = read_split(va_paths, maps, a.baselines, a.workers)
    fv = load(va_paths, a.cache)
    assert len(fv["man"]) == len(Gv["t0"])
    ep_v = np.unique(Gv["episode"], return_inverse=True)[1]
    predictors = [(m, None) for m in a.baselines] + [(os.path.basename(os.path.dirname(c)), c) for c in a.ckpt]
    val_dec = {}
    for name, ck in predictors:
        P = model_arrays(ck, fv)[0] if ck else baseline_arrays(Bv[name], Gv)
        val_dec[name] = {"score": decision_arrays(Gv, P, "score", ep_v)}
        if ck:
            val_dec[name]["composed"] = decision_arrays(Gv, P, "composed", ep_v)
        del P
    del Bv, fv
    for split in a.split:
        te_paths = manifest_paths(a.manifest, [split], a.scenarios)
        hours_te = len(te_paths) * a.logged_s / 3600
        hours_by_sc = {}
        for r in man["runs"]:
            if r["split"] == split and r["n"] and (a.scenarios is None or r["scenario"] in a.scenarios):
                hours_by_sc[r["scenario"]] = hours_by_sc.get(r["scenario"], 0.0) + a.logged_s / 3600
        Gt, Bt = read_split(te_paths, maps, a.baselines, a.workers)
        ft = load(te_paths, a.cache)
        assert len(ft["man"]) == len(Gt["t0"])
        ep_t = np.unique(Gt["episode"], return_inverse=True)[1]
        results = {"split": split, "runs": [os.path.basename(p).replace(".jsonl.gz", "") for p in te_paths],
                   "n": int(len(Gt["t0"])), "hours": hours_te, "n_val": int(len(Gv["t0"]))}
        for name, ck in predictors:
            print(f"== {split} {name}", flush=True)
            if ck:
                P, ckd = model_arrays(ck, ft)
            else:
                P = baseline_arrays(Bt[name], Gt)
            res = {"trajectory": traj_block(Gt, P), "maneuver": maneuver_block(Gt, P), "arrival": arrival_block(Gt, P),
                   "conflict": conflict_block(val_dec[name]["score"], decision_arrays(Gt, P, "score", ep_t),
                                              hours_val, hours_te, hours_by_sc)}
            if ck:
                res["conflict_composed"] = conflict_block(val_dec[name]["composed"],
                                                          decision_arrays(Gt, P, "composed", ep_t),
                                                          hours_val, hours_te, hours_by_sc)
                res["occupancy"] = occupancy_block(ft, P["occ"])
                res["inputs"] = ckd["mask"]
            results[name] = res
            c = res["conflict"]
            print(json.dumps({"minFDE@8s": res["trajectory"]["all"].get("minFDE@8s"),
                              "man_acc": res["maneuver"]["all"].get("man_acc"),
                              "crossing_mae": res["arrival"].get("crossing/mae_all"), "auprc": c["risk_cal"].get("auprc"),
                              "alerts": {k: c["alerts"][k] for k in ("recall", "timely_recall", "precision",
                                                                    "false_alert_vehicles_per_hour", "p_star")}},
                             default=float), flush=True)
            del P
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        json.dump(results, open(os.path.join(a.out, split, "results.json"), "w"), indent=1, default=float)
        del Gt, Bt, ft


if __name__ == "__main__":
    main()
