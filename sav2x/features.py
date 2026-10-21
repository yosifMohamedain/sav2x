"""Turn vehicle-side samples (schemas/sample.vehicle.schema.json) into fixed-size arrays.

Everything is expressed in the ego frame at t0: origin at the ego's BSM position,
x along its BSM heading, y to its left. That makes the model independent of which
approach or intersection it is on.

One featurization serves every ablation: inputs are masked at batch time
(see InputMask), so signal encodings and input groups can be switched off without
re-reading the JSON.

Signal features per group (route left / straight / right, then the pedestrian group):
    state one-hot(6), pedestrian flag, sin and arctan colour-scaled remaining time
    (I2XTraj / V2X-RECT style), remaining/30, SPaT age, then the interval block:
    likely/max availability, min/max/30, confidence, time in state/30, timing valid,
    and P(state has changed by tau) on the 0.5 s bins from a triangular distribution
    on [minEnd, maxEnd] with mode likelyTime (uniform without likelyTime, a step at
    minEnd when only minEnd is sent).
"""
import gzip
import json
import math

import numpy as np

MANS = ["left", "straight", "right"]
STATE6 = {"stop-And-Remain": 0, "stop-Then-Proceed": 1, "permissive-Movement-Allowed": 2,
          "protected-Movement-Allowed": 3, "permissive-clearance": 4, "protected-clearance": 4}
COLOR = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 0}
T_OMEGA = 60.0
H, T, NB, NP, NZ = 30, 40, 40, 31, 6
F_HIST = 14
F_PRES = NZ + 1
F_SIG_BASE = 11          # state(6) + ped + sin + atan + rem + age
F_SIG_INT = 7            # has_likely, has_max, min, max, conf, elapsed, valid
F_SIG = F_SIG_BASE + F_SIG_INT + NB
F_ROUTE = 10
F_CTX = 18
LIKELY_COLS = slice(0, F_SIG_BASE)
INTERVAL_COLS = slice(F_SIG_BASE, F_SIG)


def change_cdf(a, c, b, taus):
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
    x = taus
    left = np.where(x <= a, 0.0, (x - a) ** 2 / ((b - a) * max(c - a, 1e-6)))
    right = 1 - (b - x) ** 2 / ((b - a) * max(b - c, 1e-6))
    out = np.where(x <= c, left, np.where(x < b, right, 1.0))
    if c <= a:
        out = np.where(x < b, right, 1.0)
    return np.clip(out, 0, 1).astype(np.float32)


TAUS = 0.5 * np.arange(1, NB + 1)


def sig_vec(g, oracle_change=None):
    f = np.zeros(F_SIG, dtype=np.float32)
    st = STATE6.get(g["state"], 5)
    f[st] = 1
    f[6] = 1.0 if g["kind"] == "pedestrian" else 0.0
    mn, lk, mx = g["min_end_s"], g["likely_end_s"], g["max_end_s"]
    if oracle_change is not None:
        mn = lk = mx = oracle_change
    rem = lk if lk is not None else (mn if mn is not None else 0.0)
    rem = max(0.0, rem)
    d = COLOR[st]
    f[7] = math.sin((rem / T_OMEGA) * (1 / 3) ** d)
    f[8] = math.atan((rem / T_OMEGA) * (1 / 3) ** d)
    f[9] = rem / 30.0
    f[11] = lk is not None
    f[12] = mx is not None
    f[13] = max(0.0, mn or 0.0) / 30.0
    f[14] = max(0.0, mx) / 30.0 if mx is not None else 0.0
    f[15] = g["conf_p"] if g.get("conf_p") is not None else 0.0
    f[16] = min(g.get("elapsed_s") or 0.0, 120.0) / 30.0
    f[17] = 1.0 if g.get("timing_valid", True) else 0.0
    f[F_SIG_BASE + F_SIG_INT:] = change_cdf(mn or 0.0, lk, mx, TAUS)
    return f


def featurize(s, oracle=False):
    """One sample -> dict of arrays (inputs, targets, metadata)."""
    hist = np.asarray(s["ego"]["hist"], dtype=np.float64)
    valid = hist[:, 11] > 0.5
    last = np.flatnonzero(valid)[-1]
    x0, y0 = hist[last, 0], hist[last, 1]
    yaw0 = math.radians(90.0 - hist[last, 3])
    c, sn = math.cos(yaw0), math.sin(yaw0)

    def to_ego(xy):
        xy = np.asarray(xy, dtype=np.float64)
        dx, dy = xy[..., 0] - x0, xy[..., 1] - y0
        return np.stack([c * dx + sn * dy, -sn * dx + c * dy], -1)

    # ---- ego history
    h = np.zeros((H, F_HIST), dtype=np.float32)
    n = min(H, len(hist))
    hh = hist[-n:]
    p = to_ego(hh[:, :2])
    yaw = np.radians(90.0 - hh[:, 3])
    dyaw = yaw - yaw0
    h[-n:, 0] = p[:, 0] / 20.0
    h[-n:, 1] = p[:, 1] / 20.0
    h[-n:, 2] = hh[:, 2] / 15.0
    h[-n:, 3] = hh[:, 2] * np.cos(dyaw) / 15.0
    h[-n:, 4] = hh[:, 2] * np.sin(dyaw) / 15.0
    h[-n:, 5] = np.cos(dyaw)
    h[-n:, 6] = np.sin(dyaw)
    h[-n:, 7] = hh[:, 4] / 3.0
    h[-n:, 8] = hh[:, 5] / 30.0
    h[-n:, 9] = hh[:, 6]
    h[-n:, 10] = hh[:, 7]
    h[-n:, 11] = hh[:, 8]
    r = hh[:, 9]
    h[-n:, 12] = np.where(np.abs(r) > 3000, 0.0, np.clip(10.0 / np.where(np.abs(r) < 1e-3, 1e-3, r), -1, 1))
    h[-n:, 13] = hh[:, 11]
    h[~np.concatenate([np.zeros(H - n, bool), valid[-n:]]), :13] = 0
    # ---- routes (slots left / straight / right)
    rmask = np.zeros(3, dtype=np.float32)
    path = np.zeros((3, NP, 2), dtype=np.float32)
    rfeat = np.zeros((3, F_ROUTE), dtype=np.float32)
    rsg = [None, None, None]
    crosses = np.zeros(3, dtype=np.float32)
    rzone = np.zeros((3, 2), dtype=np.float32)
    for rt in s["routes"]:
        k = MANS.index(rt["maneuver"])
        rmask[k] = 1
        pts = np.asarray(rt["path"], dtype=np.float64)[:NP]
        path[k, :len(pts)] = to_ego(pts)
        if len(pts) < NP:
            path[k, len(pts):] = path[k, len(pts) - 1]
        rfeat[k] = [rt["lane_change"], rt["crosses_target"], "inbound" in rt["zones"], "outbound" in rt["zones"],
                    rt["s_stopline_m"] is not None, (rt["s_stopline_m"] or 0.0) / 50.0,
                    rt["s_cw_entry_m"] is not None, (rt["s_cw_entry_m"] or 0.0) / 50.0,
                    (rt["s_cw_exit_m"] or 0.0) / 50.0, 1.0]
        rsg[k] = rt["signal_group"]
        crosses[k] = rt["crosses_target"]
        rzone[k] = ["inbound" in rt["zones"], "outbound" in rt["zones"]]
    # ---- signals: three route groups + pedestrian group
    sig = np.zeros((4, F_SIG), dtype=np.float32)
    smask = np.zeros(4, dtype=np.float32)
    S = s.get("signals")
    truth = {t["signal_group"]: t["next_change_s"] for t in (s.get("signal_truth") or [])}
    if S is not None:
        by = {g["signal_group"]: g for g in S["groups"]}
        for k, g in enumerate(rsg + [s["target_signal_group"]]):
            if g is None or g not in by:
                continue
            sig[k] = sig_vec(by[g], truth.get(g) if oracle else None)
            sig[k, 10] = min(S["spat_age_s"], 2.0) * 5.0
            smask[k] = 1
    # ---- pedestrian presence (SDSM) by zone
    cnt = np.asarray(s["ped_presence"]["count"], dtype=np.float32)
    pv = np.asarray(s["ped_presence"]["valid"], dtype=np.float32)
    pres = np.zeros((H, F_PRES), dtype=np.float32)
    m = min(H, len(cnt))
    pres[-m:, :NZ] = np.minimum(cnt[-m:, :NZ], 6) / 3.0
    pres[-m:, NZ] = pv[-m:]
    # ---- context
    mm = s["ego"]["map_match"]
    ld = s.get("leader_bsm") or {"present": False}
    pbd = S.get("ped_bicycle_detect") if S else None
    ctx = np.array([mm["role"] == "ingress", mm["role"] == "connector",
                    (mm["dist_to_stopline_m"] or 0.0) / 50.0, mm["lat_m"],
                    s["ego"]["length_m"] / 5.0, s["ego"]["width_m"] / 2.0,
                    S is not None, min(S["spat_age_s"], 2.0) if S else 0.0,
                    min(s["ped_presence"]["sdsm_age_s"] or 2.0, 2.0), pv.mean(),
                    pbd is not None, bool(pbd),
                    ld["present"], (ld.get("gap_m") or 0.0) / 50.0 if ld["present"] else 0.0,
                    (ld.get("speed_mps") or 0.0) / 15.0 if ld["present"] else 0.0,
                    (ld.get("accel_long_mps2") or 0.0) / 3.0 if ld["present"] else 0.0,
                    bool(ld.get("brake_applied")) if ld["present"] else 0.0,
                    s.get("crosswalk_length_m", 12.8) / 20.0], dtype=np.float32)
    # ---- targets
    L = s["labels"]
    fut = np.asarray(L["fut"], dtype=np.float64)[1::2][:T]
    fxy = to_ego(fut[:, :2]).astype(np.float32)
    fm = fut[:, 3] > 0.5
    man = MANS.index(L["maneuver"]) if L["maneuver"] in MANS and rmask[MANS.index(L["maneuver"])] else -1
    cw = L["cw"]
    bs = s["bin_s"]
    if cw["entry_s"] is not None:
        eb, ee = min(int(cw["entry_s"] / bs), NB - 1), 1
    else:
        eb, ee = min(int(cw["censor_s"] / bs), NB), 0
    if cw["entry_s"] is not None and cw["exit_s"] is not None:
        xb, xe = min(int(cw["exit_s"] / bs), NB - 1), 1
    elif cw["entry_s"] is not None:
        xb, xe = min(int(cw["censor_s"] / bs), NB), 0
    else:
        xb, xe = 0, -1
    occ = np.stack([L["ped_occupancy"]["inbound"][:NB], L["ped_occupancy"]["outbound"][:NB]]).astype(np.float32)
    om = (np.arange(NB) < L["ped_occupancy"]["valid_bins"]).astype(np.float32)
    cf = L["conflict"]
    hist_last_speed = float(hist[last, 2])
    return {"hist": h, "pres": pres, "ctx": ctx, "path": path, "rfeat": rfeat, "rmask": rmask, "crosses": crosses,
            "rzone": rzone, "sig": sig, "smask": smask, "fut": fxy, "fmask": fm, "man": man,
            "surv": np.array([eb, ee, xb, xe], dtype=np.int64), "inside": bool(cw["inside_at_t0"]),
            "occ": occ, "omask": om, "conf": float(cf["label"]), "conf_known": float(cf["known"]),
            "conf_entry": cf["ego_entry_s"] if cf["ego_entry_s"] is not None else -1.0,
            "speed": hist_last_speed, "t0": s["t0_ms"] / 1000.0,
            "episode": f"{s['run_id']}|{L.get('truth_id', s['ego']['temp_id'])}", "scenario": s.get("scenario", ""),
            "role": mm["role"], "dist_stop": mm["dist_to_stopline_m"] if mm["dist_to_stopline_m"] is not None else -1.0,
            "true_crosses": bool(L["crosses_target"])}


ARRAY_KEYS = {"hist": np.float16, "pres": np.float16, "ctx": np.float32, "path": np.float32, "rfeat": np.float32,
              "rmask": np.float32, "crosses": np.float32, "rzone": np.float32, "sig": np.float16, "smask": np.float32,
              "fut": np.float32, "fmask": bool, "man": np.int64, "surv": np.int64, "inside": bool, "occ": np.float16,
              "omask": np.float16, "conf": np.float32, "conf_known": np.float32, "conf_entry": np.float32,
              "speed": np.float32, "t0": np.float64, "dist_stop": np.float32, "true_crosses": bool}
META_KEYS = ["episode", "scenario", "role"]


def featurize_file(path, oracle=False, limit=None):
    """One samples file -> dict of stacked arrays."""
    rows = []
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            rows.append(featurize(json.loads(line), oracle))
    out = {k: np.stack([r[k] for r in rows]).astype(dt) for k, dt in ARRAY_KEYS.items()}
    for k in META_KEYS:
        out[k] = np.array([r[k] for r in rows])
    return out
