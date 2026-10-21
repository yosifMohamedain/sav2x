"""Metrics for the vehicle-side predictor and baselines."""
from collections import defaultdict

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

A_COMF, T_PR, T_SYS = 3.4, 1.5, 0.3
TTA_MAX = 12.0
HZ = {"3s": 14, "5s": 24, "8s": 39}


def tta_min(v):
    """Minimum useful warning lead: latency + perception-reaction + time to brake at a comfortable rate."""
    return T_SYS + T_PR + np.asarray(v) / (2 * A_COMF)


def traj_errors(loc, prob, fut, fmask):
    """Vectorized. loc (N,K,T,2), prob (N,K), fut (N,T,2), fmask (N,T) -> dict of (N,) arrays (nan where the
    horizon is not observed)."""
    err = np.linalg.norm(loc - fut[:, None], axis=-1)                       # (N,K,T)
    out = {}
    top = prob.argmax(1)
    t3 = np.argsort(-prob, 1)[:, :3]
    w = np.take_along_axis(prob, t3, 1)
    w = w / w.sum(1, keepdims=True)
    rows = np.arange(len(loc))
    for h, k in HZ.items():
        ok = fmask[:, :k + 1].all(1)
        ade, fde = err[:, :, :k + 1].mean(-1), err[:, :, k]
        vals = {"minADE": ade.min(1), "minFDE": fde.min(1), "FDE1": fde[rows, top], "ADE1": ade[rows, top],
                "MR2": (fde.min(1) > 2.0).astype(float), "ISS_ADE3": (w * np.take_along_axis(ade, t3, 1)).sum(1)}
        for m, v in vals.items():
            out[f"{m}@{h}"] = np.where(ok, v, np.nan)
    return out


def slice_means(arrays, masks):
    """arrays: name -> (N,) with nan for not applicable; masks: slice -> (N,) bool."""
    res = {}
    for g, msk in masks.items():
        d = {}
        for k, v in arrays.items():
            vv = v[msk]
            vv = vv[~np.isnan(vv)]
            if len(vv):
                d[k] = float(vv.mean())
        d["n"] = int(msk.sum())
        res[g] = d
    return res


def binary_metrics(p, y):
    p, y = np.asarray(p, float), np.asarray(y, int)
    res = {"n": int(len(y)), "pos_rate": float(y.mean()) if len(y) else None}
    if len(y) and 0 < y.sum() < len(y):
        res.update(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)))
    if len(y):
        res.update(brier=float(((p - y) ** 2).mean()), ece=ece(p, y))
    return res


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y)
    out = []
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "pred": float(p[m].mean()) if m.any() else None, "obs": float(y[m].mean()) if m.any() else None})
    return out


class Calibrator:
    def __init__(self):
        self.iso = None

    def fit(self, p, y):
        if 0 < np.sum(y) < len(y):
            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
        return self

    def __call__(self, p):
        return np.asarray(p) if self.iso is None else self.iso.predict(np.asarray(p))


def alert_events(D, p_star, hours, window=True, sel=None, min_lead=None):
    """Driver alerts scored per vehicle episode (vectorized).

    D: dict of arrays over decisions: ep (int episode code), t0, score (calibrated), pred_entry, y, t_conf
    (absolute time the ego enters the crosswalk in a conflict), speed. An event is an episode with a conflict;
    it is detected if an alert fires before the conflict entry, and timely if the lead at the first alert is at
    least TTA_min(speed at that alert)."""
    idx = np.arange(len(D["ep"])) if sel is None else np.flatnonzero(sel)
    if len(idx) == 0:
        return {"events": 0, "detected": 0, "recall": 0.0, "precision": 0.0, "F2": 0.0, "timely_recall": 0.0,
                "median_lead_s": None, "false_alert_vehicles_per_hour": 0.0, "false_alert_share": 0.0,
                "vehicles": 0, "p_star": p_star}
    ep, t0, sc, pe, y, tc, sp = (D[k][idx] for k in ("ep", "t0", "score", "pred_entry", "y", "t_conf", "speed"))
    uniq, inv = np.unique(ep, return_inverse=True)
    G, n = len(uniq), len(ep)
    te = np.full(G, np.inf)
    yy = y > 0
    np.minimum.at(te, inv[yy], tc[yy])
    al = (sc >= p_star) & (t0 < te[inv])
    if window:
        al &= pe <= TTA_MAX
    first_t = np.full(G, np.inf)                 # earliest alert per episode
    if al.any():
        np.minimum.at(first_t, inv[al], t0[al])
    has = np.isfinite(first_t)
    first_idx = np.full(G, -1)
    if al.any():
        cand = np.flatnonzero(al)
        is_first = t0[cand] == first_t[inv[cand]]
        first_idx[inv[cand[is_first]]] = cand[is_first]
    ev = np.isfinite(te)
    tp = int((ev & has).sum()); fn = int((ev & ~has).sum())
    fp = int((~ev & has).sum()); tn = int((~ev & ~has).sum())
    lead = te[ev & has] - first_t[ev & has]
    need = tta_min(sp[first_idx[ev & has]]) if min_lead is None else min_lead
    timely = lead >= need
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"events": tp + fn, "detected": tp, "recall": rec, "precision": prec,
            "F2": 5 * prec * rec / max(4 * prec + rec, 1e-9),
            "timely_recall": float(timely.sum()) / max(tp + fn, 1),
            "median_lead_s": float(np.median(lead)) if len(lead) else None,
            "false_alert_vehicles_per_hour": fp / max(hours, 1e-9),
            "false_alert_share": fp / max(fp + tn, 1), "vehicles": G, "p_star": p_star}


GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def choose_threshold(dec_val, hours_val):
    return max(GRID, key=lambda ps: (alert_events(dec_val, ps, hours_val)["F2"], ps))


def operating_curve(dec, hours, n=40):
    thr = np.unique(np.concatenate([np.geomspace(1e-3, 0.98, n)]))
    return [{"thr": float(t), **{k: v for k, v in alert_events(dec, float(t), hours).items()
                                 if k in ("recall", "timely_recall", "precision", "false_alert_vehicles_per_hour")}}
            for t in thr]
