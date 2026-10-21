"""Turn schema-conformant JSON samples into fixed-size numpy arrays.

Signal encodings (selected with signal_mode):
    none      : no SPaT input (learned 'no signal' token only)
    likely    : current state + remaining time to likelyTime, with the
                sin (I2XTraj) and arctan (V2X-RECT) color-scaled encodings
    interval  : likely + minEnd / maxEnd / confidence / elapsed, plus a
                phase-change CDF: P(the movement has changed state by tau)
                for tau on the hazard bins, from a triangular distribution on
                [minEnd, maxEnd] with mode likelyTime (uniform if likelyTime is
                absent, a step at minEnd if only minEnd is sent)
    oracle    : interval features built from the realized next change
                (upper bound; sim only)
"""
import math

import numpy as np

from data.geom import LEGS, MANEUVERS, IntersectionMap, enu_yaw

STATE6 = {"stop-And-Remain": 0, "stop-Then-Proceed": 1, "permissive-Movement-Allowed": 2,
          "protected-Movement-Allowed": 3, "permissive-clearance": 4, "protected-clearance": 4}
COLOR = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 0}   # red / green / yellow index for the sin/arctan encodings
SCALE_XY = 50.0
SCALE_V = 15.0
T_OMEGA = 60.0


class FeatureConfig:
    def __init__(self, signal_mode="interval", n_bins=40, bin_s=0.5, fut_stride=2, max_agents=48):
        self.signal_mode = signal_mode
        self.n_bins = n_bins
        self.bin_s = bin_s
        self.fut_stride = fut_stride       # 10 Hz labels -> 5 Hz targets
        self.max_agents = max_agents


# ---------------------------------------------------------------- map tokens
def resample(pts, n):
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(pts[:1], n, axis=0)
    q = np.linspace(0, s[-1], n)
    return np.stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])], axis=1)


def cut_polyline(pts, s0, s1):
    pts = np.asarray(pts, dtype=np.float64)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.linspace(max(0, s0), min(s[-1], s1), 10)
    return np.stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])], axis=1), s[-1]


def build_map_tokens(imap: IntersectionMap, radius=120.0, seg_len=20.0, n_pts=10):
    """Static map tokens: lane segments, connectors, crosswalks.

    Returns dict of arrays: pts (M,P,2) local meters, attr (M,F), pose (M,3) [x,y,yaw],
    sg (M,) index into imap signal-group order (-1 none), plus lookup tables."""
    sg_order = [g["signal_group"] for g in imap.raw["signal_groups"]]
    sg_idx = {g: i for i, g in enumerate(sg_order)}
    toks = []

    def add(pts, kind, approach, man_multi, sg, is_target, width, dist_stop):
        pts = resample(pts, n_pts)
        attr = np.zeros(4 + 5 + 3 + 4, dtype=np.float32)
        attr[kind] = 1
        attr[4 + (LEGS.index(approach) if approach in LEGS else 4)] = 1
        for m in man_multi:
            if m in MANEUVERS:
                attr[9 + MANEUVERS.index(m)] = 1
        attr[12] = float(is_target)
        attr[13] = width / 4.0
        attr[14] = dist_stop / 100.0
        attr[15] = 1.0 if sg is not None else 0.0
        d = pts[-1] - pts[0]
        toks.append((pts, attr, (pts[:, 0].mean(), pts[:, 1].mean(), math.atan2(d[1], d[0])),
                     sg_idx[sg] if sg is not None else -1, approach, man_multi))

    for l in imap.raw["lanes"]:
        if l["kind"] != "vehicle":
            continue
        cl = np.asarray(l["centerline"])
        L = np.linalg.norm(np.diff(cl, axis=0), axis=1).sum()
        if l["role"] == "ingress":
            start = max(0.0, L - radius)
            for s0 in np.arange(start, L - 1e-3, seg_len):
                pts, _ = cut_polyline(cl, s0, min(L, s0 + seg_len))
                add(pts, 0, l["approach"], l.get("maneuvers", []), None, False, l["lane_width_m"], L - min(L, s0 + seg_len))
        else:
            for s0 in np.arange(0, min(L, 60.0) - 1e-3, seg_len):
                pts, _ = cut_polyline(cl, s0, min(L, s0 + seg_len))
                add(pts, 1, l["approach"], [], None, False, l["lane_width_m"], 0.0)
    conn_tok = {}
    for c in imap.raw["connectors"]:
        if len(c["centerline"]) < 2:
            continue
        conn_tok[c["connection_id"]] = len(toks)
        add(c["centerline"], 2, c["approach"], [c["maneuver"]], c["signal_group"], False, 3.2, 0.0)
    target_tok = -1
    for cw in imap.raw["crosswalks"]:
        if cw["is_target"]:
            target_tok = len(toks)
        add(cw["centerline"], 3, cw["leg"], [], cw["signal_group"], cw["is_target"], cw["width_m"], 0.0)
    # (approach, maneuver) -> connector token indices, for the maneuver head
    man_tok = np.full((4, 3, 2), -1, dtype=np.int64)
    for c in imap.raw["connectors"]:
        if c["connection_id"] not in conn_tok or c["maneuver"] not in MANEUVERS:
            continue
        a, m = LEGS.index(c["approach"]), MANEUVERS.index(c["maneuver"])
        slot = 0 if man_tok[a, m, 0] < 0 else 1
        man_tok[a, m, slot] = conn_tok[c["connection_id"]]
    man_sg = np.full((4, 3), -1, dtype=np.int64)
    for a, leg in enumerate(LEGS):
        for m, man in enumerate(MANEUVERS):
            g = imap.movement_sg(leg, man)
            if g is not None:
                man_sg[a, m] = sg_idx[g]
    return {"pts": np.stack([t[0] for t in toks]).astype(np.float32),
            "attr": np.stack([t[1] for t in toks]).astype(np.float32),
            "pose": np.asarray([t[2] for t in toks], dtype=np.float32),
            "sg": np.asarray([t[3] for t in toks], dtype=np.int64),
            "target_tok": target_tok, "man_tok": man_tok, "man_sg": man_sg,
            "sg_order": sg_order, "target_sg_idx": sg_idx[imap.target_sg],
            "sg_is_ped": np.asarray([imap.sgs[g]["kind"] == "pedestrian" for g in sg_order], dtype=np.float32)}


# ---------------------------------------------------------------- signals
def change_cdf(a, c, b, taus):
    """P(change time <= tau) for a triangular(a, c, b); uniform if c is None; step if b is None."""
    a = max(0.0, a)
    if b is None:
        t = a if c is None else max(a, c)
        return (taus >= t).astype(np.float32)
    b = max(a, b)
    if b - a < 1e-3:
        return (taus >= a).astype(np.float32)
    if c is None:
        return np.clip((taus - a) / (b - a), 0, 1).astype(np.float32)
    c = min(max(c, a), b)
    out = np.zeros_like(taus, dtype=np.float64)
    for i, x in enumerate(taus):
        if x <= a:
            out[i] = 0.0
        elif x <= c:
            out[i] = (x - a) ** 2 / ((b - a) * max(c - a, 1e-6)) if c > a else 1.0
        elif x < b:
            out[i] = 1 - (b - x) ** 2 / ((b - a) * max(b - c, 1e-6))
        else:
            out[i] = 1.0
    return out.astype(np.float32)


SIG_F = 6 + 1 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 1  # 19 base features (+ n_bins CDF)


def signal_features(sig, signal_truth, sg_order, cfg: FeatureConfig):
    G = len(sg_order)
    F = SIG_F + cfg.n_bins
    out = np.zeros((G, F), dtype=np.float32)
    mask = np.zeros(G, dtype=np.float32)
    if sig is None or cfg.signal_mode == "none":
        return out, mask
    taus = cfg.bin_s * np.arange(1, cfg.n_bins + 1)
    truth = {t["signal_group"]: t for t in (signal_truth or [])}
    by_g = {g["signal_group"]: g for g in sig["groups"]}
    for i, g in enumerate(sg_order):
        s = by_g.get(g)
        if s is None:
            continue
        mask[i] = 1
        st = STATE6.get(s["state"], 5)
        f = out[i]
        f[st] = 1
        f[6] = 1.0 if s["kind"] == "pedestrian" else 0.0
        mn, lk, mx = s["min_end_s"], s["likely_end_s"], s["max_end_s"]
        if cfg.signal_mode == "oracle" and g in truth:
            mn = lk = mx = truth[g]["next_change_s"]
        rem = lk if lk is not None else mn
        rem = max(0.0, rem if rem is not None else 0.0)
        d = COLOR[st]
        f[7] = math.sin((rem / T_OMEGA) * (1 / 3) ** d)        # I2XTraj-style
        f[8] = math.atan((rem / T_OMEGA) * (1 / 3) ** d)       # V2X-RECT-style
        f[9] = rem / 30.0
        f[10] = sig["spat_age_s"] * 10.0
        if cfg.signal_mode == "likely":
            continue
        f[11] = (lk is not None)
        f[12] = (mx is not None)
        f[13] = max(0.0, mn) / 30.0 if mn is not None else 0.0
        f[14] = max(0.0, mx) / 30.0 if mx is not None else 0.0
        f[15] = s["conf_p"] if s.get("conf_p") is not None else 0.0
        f[16] = (s.get("conf_p") is not None)
        f[17] = min(s.get("elapsed_s") or 0.0, 120.0) / 30.0
        f[18] = 1.0 if s.get("timing_valid", True) else 0.0
        f[SIG_F:] = change_cdf(mn if mn is not None else 0.0, lk, mx, taus)
    return out, mask


# ---------------------------------------------------------------- agents
SRC = {"sdsm": 0, "bsm": 1, "self": 2, "fused": 1}
ROLE = {"ingress": 0, "egress": 1, "connector": 2}
HIST_F = 13
BSM_F = 6
ATTR_F = 2 + 3 + 2 + 4 + 5 + 3


def agent_arrays(agent, is_ego, case_vehicle):
    h = np.asarray(agent["hist"], dtype=np.float64)        # (H, 9)
    valid = h[:, 8] > 0.5
    yaw = enu_yaw(h[:, 3])
    feat = np.zeros((len(h), HIST_F), dtype=np.float32)
    feat[:, 0] = h[:, 0] / SCALE_XY
    feat[:, 1] = h[:, 1] / SCALE_XY
    feat[:, 2] = h[:, 2] * np.cos(yaw) / SCALE_V
    feat[:, 3] = h[:, 2] * np.sin(yaw) / SCALE_V
    feat[:, 4] = np.cos(yaw)
    feat[:, 5] = np.sin(yaw)
    feat[:, 6] = h[:, 2] / SCALE_V
    feat[:, 7] = h[:, 4] / 2.0
    feat[:, 8] = h[:, 5]
    feat[:, 9] = h[:, 6] / 10.0
    feat[:, 10] = h[:, 7] * 10.0
    feat[:, 11] = valid
    feat[:, 12] = np.arange(len(h)) / len(h)
    feat[~valid, :11] = 0
    bsm = np.zeros((len(h), BSM_F), dtype=np.float32)
    if agent.get("bsm_ext") is not None and len(agent["bsm_ext"]):
        b = np.asarray(agent["bsm_ext"], dtype=np.float64)
        bsm[:, 0] = b[:, 0] / 3.0
        bsm[:, 1] = b[:, 1] / 30.0
        bsm[:, 2:5] = b[:, 2:5]
        r = b[:, 5]
        bsm[:, 5] = np.where(np.abs(r) > 3000, 0.0, np.clip(10.0 / np.where(np.abs(r) < 1e-3, 1e-3, r), -1, 1))
    last = np.flatnonzero(valid)[-1] if valid.any() else len(h) - 1
    pose = np.array([h[last, 0], h[last, 1], yaw[last]], dtype=np.float32)
    attr = np.zeros(ATTR_F, dtype=np.float32)
    attr[0 if agent["type"] == "vehicle" else 1] = 1
    attr[2 + SRC.get(agent["source"], 0)] = 1
    attr[5] = (agent.get("length_m") or 0.5) / 5.0
    attr[6] = (agent.get("width_m") or 0.5) / 2.0
    mm = agent.get("map_match")
    attr[7 + (ROLE[mm["role"]] if mm else 3)] = 1
    attr[11 + (LEGS.index(mm["approach"]) if mm and mm.get("approach") in LEGS else 4)] = 1
    attr[16] = (mm.get("dist_to_stopline_m", 0.0) / 50.0) if mm else 0.0
    attr[17] = float(is_ego)
    attr[18] = float(case_vehicle)
    appr = LEGS.index(mm["approach"]) if mm and mm["role"] == "ingress" and mm.get("approach") in LEGS else -1
    return feat, bsm, attr, pose, valid.any(), appr


def labels_arrays(agent, pose, cfg: FeatureConfig, horizon_s):
    fut = np.asarray(agent["fut"], dtype=np.float64)[cfg.fut_stride - 1::cfg.fut_stride]
    fxy = fut[:, :2].astype(np.float32)
    fmask = fut[:, 3] > 0.5
    lab = agent["labels"]
    man = MANEUVERS.index(lab["maneuver"]) if lab.get("maneuver") in MANEUVERS else -1
    cw = lab["cw"]
    nb, bs = cfg.n_bins, cfg.bin_s
    # entry: (bin, event) with event=1 observed, 0 censored at bin
    if cw["entry_s"] is not None:
        e_bin, e_evt = min(int(cw["entry_s"] / bs), nb - 1), 1
    else:
        e_bin, e_evt = min(int(cw["censor_s"] / bs), nb), 0
    if cw["entry_s"] is not None and cw["exit_s"] is not None:
        x_bin, x_evt = min(int(cw["exit_s"] / bs), nb - 1), 1
    elif cw["entry_s"] is not None:
        x_bin, x_evt = min(int(cw["censor_s"] / bs), nb), 0
    else:
        x_bin, x_evt = 0, -1
    return fxy, fmask, man, (e_bin, e_evt, x_bin, x_evt), bool(lab.get("crosses_target"))


def featurize(sample, mt, cfg: FeatureConfig, agent_weight_other=0.25):
    """One sample -> dict of numpy arrays (agents padded later by the collate fn)."""
    case_vehicle = sample["case"] == "vehicle"
    agents = ([sample["ego"]] if case_vehicle else []) + sample["agents"]
    agents = agents[:cfg.max_agents]
    H = sample["n_hist"]
    A = len(agents)
    T = sample["n_fut"] // cfg.fut_stride
    out = {k: [] for k in ("hist", "bsm", "attr", "pose", "fut", "fmask", "man", "surv", "crosses", "appr",
                           "weight", "valid", "is_vru")}
    for i, ag in enumerate(agents):
        is_ego = case_vehicle and i == 0
        feat, bsm, attr, pose, ok, appr = agent_arrays(ag, is_ego, case_vehicle)
        fxy, fmask, man, surv, crosses = labels_arrays(ag, pose, cfg, sample["label_horizon_s"])
        out["hist"].append(feat); out["bsm"].append(bsm); out["attr"].append(attr); out["pose"].append(pose)
        out["fut"].append(fxy); out["fmask"].append(fmask); out["man"].append(man); out["surv"].append(surv)
        out["crosses"].append(crosses); out["appr"].append(appr); out["valid"].append(ok)
        out["is_vru"].append(ag["type"] == "vru")
        out["weight"].append(1.0 if (not case_vehicle or is_ego) else agent_weight_other)
    sig, sig_mask = signal_features(sample.get("signals"), sample.get("signal_truth"), mt["sg_order"], cfg)
    pbd = (sample.get("signals") or {}).get("ped_bicycle_detect")
    res = {"hist": np.stack(out["hist"]), "bsm": np.stack(out["bsm"]), "attr": np.stack(out["attr"]),
           "pose": np.stack(out["pose"]), "fut": np.stack(out["fut"]), "fmask": np.stack(out["fmask"]),
           "man": np.asarray(out["man"], dtype=np.int64), "surv": np.asarray(out["surv"], dtype=np.int64),
           "crosses": np.asarray(out["crosses"]), "appr": np.asarray(out["appr"], dtype=np.int64),
           "weight": np.asarray(out["weight"], dtype=np.float32), "valid": np.asarray(out["valid"]),
           "is_vru": np.asarray(out["is_vru"]),
           "sig": sig, "sig_mask": sig_mask,
           "pbd": np.float32(-1 if pbd is None else float(pbd)),
           "ids": [a["agent_id"] for a in agents], "sample_id": sample["sample_id"], "t0_ms": sample["t0_ms"],
           "conflicts": sample.get("conflicts", []), "case": sample["case"]}
    assert res["hist"].shape[1] == H and res["fut"].shape[1] == T
    return res
