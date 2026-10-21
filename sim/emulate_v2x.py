"""Turn SUMO ground truth into the message streams a connected vehicle works with.

Writes, per run directory:
  bsm.npz          Columnar BSM records for EVERY vehicle (fields as in
                   schemas/extracted/bsm.schema.json). In simulation every vehicle can
                   serve as the ego, since its own OBU builds these from its own
                   GNSS/CAN; `transmitted` marks the connected vehicles whose BSMs
                   others can receive (cv_share of the scenario). Kept columnar
                   because 10 Hz x every vehicle is large; --bsm-jsonl N also writes
                   the first N records as JSONL for schema checks.
  sdsm.jsonl.gz    RSU SDSMs with VRU objects only (the vehicle-side model uses SDSM
                   for pedestrian presence only): detection range, noise and J2735
                   quantization, occlusion dropouts, random misses, false positives
                   (lidar clutter), temporary object IDs, processing latency.
  spat.jsonl.gz    SPaT for every signal group: state, start, minEnd, maxEnd,
                   likelyTime, TimeIntervalConfidence; pedBicycleDetect for the
                   target crosswalk. spat_profile 'min_max' or 'min_only' drops fields
                   the way some controllers do.

Usage:
    python sim/emulate_v2x.py --run data_out/runs/base_s1 [--day 0]
"""
import argparse
import datetime as dt
import gzip
import json
import math
import os
import zlib

import numpy as np
import shapely
from shapely.geometry import Polygon

CFG = {
    "bsm": {"gnss_bias_sigma_m": 0.7, "gnss_bias_tau_s": 30.0, "gnss_white_m": 0.10, "speed_sigma_mps": 0.05,
            "heading_sigma_deg": 0.5, "accel_sigma_mps2": 0.1, "id_rotation_s": 300.0, "tx_latency_ms": [5, 20],
            "p_driver_signals": 0.8},
    "sdsm": {"range_vru_m": 60.0, "pos_sigma_m": 0.20, "pos_sigma_per_m": 0.002, "speed_sigma_mps": 0.25,
             "iid_miss": 0.03, "occl_start": 0.01, "occl_end": 0.15, "reinit_gap_frames": 5,
             "false_pos_per_frame": 0.03, "false_pos_life_s": [0.3, 2.0], "false_pos_radius_m": 25.0,
             "proc_latency_ms": [40, 80]},
    "spat": {"gen_latency_ms": [5, 30], "ped_detect_p": 0.95},
}
TIC_P = [0.21, 0.36, 0.47, 0.56, 0.62, 0.68, 0.73, 0.77, 0.81, 0.85, 0.88, 0.91, 0.94, 0.96, 0.98, 1.0]


def epoch_ms(day, t_s):
    base = dt.datetime(2026, 10, 6, 14, 0, tzinfo=dt.timezone.utc) + dt.timedelta(days=day)
    return int(round(base.timestamp() * 1000 + t_s * 1000))


def timemark(ms):
    return int((ms % 3_600_000) // 100)


def minute_of_year(ms):
    t = dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc)
    return int((t - dt.datetime(t.year, 1, 1, tzinfo=dt.timezone.utc)).total_seconds() // 60)


def quant(v, lsb):
    return np.round(np.asarray(v) / lsb) * lsb


def conf_code(spread):
    if spread <= 0.05:
        return 15
    for lim, code in ((2, 12), (5, 9), (10, 6), (20, 3)):
        if spread <= lim:
            return code
    return 1


def stable_rng(*parts):
    return np.random.default_rng(zlib.crc32(":".join(map(str, parts)).encode()))


def emulate_bsm(veh, vmeta, sc, day, rng, B):
    """Vectorized per vehicle. Positions are vehicle centers (SUMO logs the front bumper)."""
    ids = veh["id"]
    order = np.lexsort((veh["t"], ids))
    out = {k: [] for k in ("t_obs_ms", "t_tx_ms", "x_m", "y_m", "speed_mps", "heading_deg", "accel_long_mps2",
                           "yaw_rate_dps", "brake_applied", "left_turn", "right_turn", "path_radius_m",
                           "pos_acc_semi_major_m", "length_m", "width_m", "transmitted", "temp_id", "veh_id")}
    ids_s = ids[order]
    bounds = np.flatnonzero(ids_s[1:] != ids_s[:-1]) + 1
    cv_share = sc["cv_share"]
    for g in np.split(order, bounds):
        vid = str(ids[g[0]])
        m = vmeta[vid]
        r = stable_rng(sc["name"], sc["seed"], vid)
        n = len(g)
        t = veh["t"][g]
        ang = veh["angle"][g].astype(np.float64)
        a = np.radians(ang)
        L, W = m["length"], m["width"]
        x = veh["x"][g] - np.sin(a) * L / 2
        y = veh["y"][g] - np.cos(a) * L / 2
        # Gauss-Markov GNSS bias + white noise
        rho = math.exp(-0.1 / B["gnss_bias_tau_s"])
        e = r.normal(0, B["gnss_bias_sigma_m"] * math.sqrt(1 - rho ** 2), (n, 2))
        e[0] = r.normal(0, B["gnss_bias_sigma_m"], 2)
        bias = np.empty_like(e)
        bias[0] = e[0]
        for k in range(1, n):
            bias[k] = rho * bias[k - 1] + e[k]
        spd = veh["speed"][g].astype(np.float64)
        yaw = np.zeros(n)
        yaw[1:] = ((ang[1:] - ang[:-1] + 540) % 360 - 180) / 0.1
        sig = veh["signals"][g]
        signals_used = r.random() < B["p_driver_signals"]
        om = np.radians(yaw)
        radius = np.where((np.abs(om) < 0.01) | (spd < 0.5), 3276.7,
                          np.clip(spd / np.where(np.abs(om) < 1e-6, 1e-6, om), -3276.6, 3276.6))
        transmitted = r.random() < cv_share
        rot = np.floor((t - t[0]) / B["id_rotation_s"]).astype(int)
        tids = np.array(["%08x" % (zlib.crc32(f"{vid}:{k}".encode()) & 0xFFFFFFFF) for k in rot])
        t_ms = np.array([epoch_ms(day, tt) for tt in t])
        out["t_obs_ms"].append(t_ms)
        out["t_tx_ms"].append(t_ms + r.integers(*B["tx_latency_ms"], n))
        out["x_m"].append(quant(x + bias[:, 0] + r.normal(0, B["gnss_white_m"], n), 0.01))
        out["y_m"].append(quant(y + bias[:, 1] + r.normal(0, B["gnss_white_m"], n), 0.01))
        out["speed_mps"].append(quant(np.maximum(0, spd + r.normal(0, B["speed_sigma_mps"], n)), 0.02))
        out["heading_deg"].append(quant((ang + r.normal(0, B["heading_sigma_deg"], n)) % 360, 0.0125))
        out["accel_long_mps2"].append(quant(veh["accel"][g] + r.normal(0, B["accel_sigma_mps2"], n), 0.01))
        out["yaw_rate_dps"].append(quant(yaw, 0.01))
        out["brake_applied"].append(((sig & 8) > 0) | (spd < 0.1))
        out["left_turn"].append(((sig & 2) > 0) & signals_used)
        out["right_turn"].append(((sig & 1) > 0) & signals_used)
        out["path_radius_m"].append(np.round(radius, 1))
        out["pos_acc_semi_major_m"].append(np.full(n, round(2 * B["gnss_bias_sigma_m"], 2)))
        out["length_m"].append(np.full(n, L))
        out["width_m"].append(np.full(n, W))
        out["transmitted"].append(np.full(n, transmitted))
        out["temp_id"].append(tids)
        out["veh_id"].append(np.full(n, vid, dtype=object))
    res = {k: np.concatenate(v) for k, v in out.items()}
    for k in ("x_m", "y_m", "speed_mps", "heading_deg", "accel_long_mps2", "yaw_rate_dps", "path_radius_m"):
        res[k] = res[k].astype(np.float32)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--day", type=int, default=None, help="day offset for timestamps (default: seed)")
    ap.add_argument("--spat-profile", default=None, choices=["full", "min_max", "min_only"])
    ap.add_argument("--bsm-jsonl", type=int, default=0, help="also write the first N BSMs as JSONL")
    a = ap.parse_args()
    sc = json.load(open(os.path.join(a.run, "scenario.json")))
    meta = json.load(open(os.path.join(a.run, "meta.json")))
    mp = json.load(open(os.path.join(a.run, "map.json")))
    day = sc["seed"] if a.day is None else a.day
    profile = a.spat_profile or sc.get("spat_profile", "full")
    rng = stable_rng(sc["name"], sc["seed"], "emul")
    veh = dict(np.load(os.path.join(a.run, "gt_vehicles.npz"), allow_pickle=True))
    per = dict(np.load(os.path.join(a.run, "gt_persons.npz"), allow_pickle=True))
    tls = dict(np.load(os.path.join(a.run, "gt_tls.npz"), allow_pickle=True))
    iid = mp["intersection_id"]

    # ---------------- BSM ----------------
    bsm = emulate_bsm(veh, meta["vehicles"], sc, day, rng, CFG["bsm"])
    np.savez_compressed(os.path.join(a.run, "bsm.npz"), **bsm)
    if a.bsm_jsonl:
        with gzip.open(os.path.join(a.run, "bsm_sample.jsonl.gz"), "wt") as f:
            for i in range(min(a.bsm_jsonl, len(bsm["x_m"]))):
                f.write(json.dumps({
                    "schema": "sav2x.bsm/2.0", "temp_id": str(bsm["temp_id"][i]), "msg_cnt": int(i % 128),
                    "sec_mark_ms": int(bsm["t_obs_ms"][i] % 60000), "t_obs_ms": int(bsm["t_obs_ms"][i]),
                    "t_tx_ms": int(bsm["t_tx_ms"][i]), "x_m": float(bsm["x_m"][i]), "y_m": float(bsm["y_m"][i]),
                    "pos_acc_semi_major_m": float(bsm["pos_acc_semi_major_m"][i]),
                    "speed_mps": float(bsm["speed_mps"][i]), "heading_deg": float(bsm["heading_deg"][i]),
                    "accel_long_mps2": float(bsm["accel_long_mps2"][i]), "yaw_rate_dps": float(bsm["yaw_rate_dps"][i]),
                    "brake_applied": bool(bsm["brake_applied"][i]),
                    "lights": {"left_turn": bool(bsm["left_turn"][i]), "right_turn": bool(bsm["right_turn"][i]),
                               "hazard": False},
                    "path_prediction": {"radius_m": float(bsm["path_radius_m"][i]), "confidence_pct": 80.0},
                    "length_m": float(bsm["length_m"][i]), "width_m": float(bsm["width_m"][i]),
                    "transmitted": bool(bsm["transmitted"][i]), "truth_id": str(bsm["veh_id"][i])}) + "\n")

    # ---------------- SDSM (VRU only) ----------------
    S = CFG["sdsm"]
    times = np.round(tls["t"], 1)
    pt_order = np.argsort(per["t"], kind="stable")
    pt_t = np.round(per["t"][pt_order], 1)
    bounds = np.searchsorted(pt_t, times, side="left")
    bounds_r = np.searchsorted(pt_t, times, side="right")
    occl, missed, tmp_id = {}, {}, {}
    next_id, msg_cnt = 1, 0
    ghosts = []  # [x, y, t_end, id]
    with gzip.open(os.path.join(a.run, "sdsm.jsonl.gz"), "wt") as f:
        for k, t in enumerate(times):
            ms = epoch_ms(day, t)
            lat = int(rng.integers(*S["proc_latency_ms"]))
            objs = []
            for i in pt_order[bounds[k]:bounds_r[k]]:
                pid = str(per["id"][i])
                x, y = float(per["x"][i]), float(per["y"][i])
                r = math.hypot(x, y)
                if r > S["range_vru_m"]:
                    missed[pid] = missed.get(pid, 0) + 1
                    continue
                st = occl.get(pid, False)
                st = (rng.random() < S["occl_start"]) if not st else (rng.random() >= S["occl_end"])
                occl[pid] = st
                if st or rng.random() < S["iid_miss"]:
                    missed[pid] = missed.get(pid, 0) + 1
                    continue
                if pid not in tmp_id or missed.get(pid, 0) >= S["reinit_gap_frames"]:
                    tmp_id[pid], next_id = next_id, next_id % 65535 + 1
                missed[pid] = 0
                sig = S["pos_sigma_m"] + S["pos_sigma_per_m"] * r
                objs.append({"obj_type": "vru", "obj_type_cfd": int(rng.integers(70, 100)), "object_id": tmp_id[pid],
                             "measurement_time_ms": -lat, "time_confidence": 10,
                             "offset_x_m": float(quant(x + rng.normal(0, sig), 0.1)),
                             "offset_y_m": float(quant(y + rng.normal(0, sig), 0.1)), "pos_conf_m": round(2 * sig, 2),
                             "speed_mps": float(quant(max(0.0, float(per["speed"][i]) + rng.normal(0, S["speed_sigma_mps"])), 0.02)),
                             "speed_conf_mps": 0.5,
                             "heading_deg": float(quant((float(per["angle"][i]) + rng.normal(0, 15)) % 360, 0.0125)),
                             "heading_conf_deg": 30.0, "vru": {"basic_type": "aPEDESTRIAN"}, "truth_id": pid})
            # lidar clutter: short-lived false VRU detections near the intersection
            if rng.random() < S["false_pos_per_frame"]:
                ang, rad = rng.uniform(0, 2 * math.pi), S["false_pos_radius_m"] * math.sqrt(rng.random())
                ghosts.append([rad * math.cos(ang), rad * math.sin(ang), t + rng.uniform(*S["false_pos_life_s"]), next_id])
                next_id = next_id % 65535 + 1
            ghosts = [g for g in ghosts if g[2] > t]
            for gx, gy, _, gid in ghosts:
                objs.append({"obj_type": "vru", "obj_type_cfd": int(rng.integers(30, 70)), "object_id": gid,
                             "measurement_time_ms": -lat, "time_confidence": 10,
                             "offset_x_m": float(quant(gx + rng.normal(0, 0.3), 0.1)),
                             "offset_y_m": float(quant(gy + rng.normal(0, 0.3), 0.1)), "pos_conf_m": 0.6,
                             "speed_mps": 0.0, "speed_conf_mps": 0.5, "heading_deg": 0.0, "heading_conf_deg": 60.0,
                             "vru": {"basic_type": "aPEDESTRIAN"}, "truth_id": "false_positive"})
            msg_cnt = (msg_cnt + 1) % 128
            f.write(json.dumps({"schema": "sav2x.sdsm/2.0", "rsu_id": "rsu07", "intersection_id": iid,
                                "msg_cnt": msg_cnt, "source_id": "52535530", "equipment_type": "rsu",
                                "sdsm_ts_ms": ms + lat, "ref": mp["ref"] | {"x_m": 0.0, "y_m": 0.0},
                                "objects": objs}, separators=(",", ":")) + "\n")

    # ---------------- SPaT ----------------
    P = CFG["spat"]
    phases = meta["phases"]
    nph = len(phases)
    sgs = mp["signal_groups"]
    tgt = mp["target_crosswalk"]

    def is_actuated(p):
        return phases[p]["minDur"] not in (None, -1) and phases[p]["maxDur"] not in (None, -1) \
            and 0 < phases[p]["minDur"] < phases[p]["maxDur"] < 1e5

    def ph_min(p):
        return phases[p]["minDur"] if is_actuated(p) else phases[p]["duration"]

    def ph_max(p):
        return phases[p]["maxDur"] if is_actuated(p) else phases[p]["duration"]

    realized = {p: [] for p in range(nph)}

    def ph_likely(p):
        if realized[p]:
            return float(np.mean(realized[p][-20:]))
        return (ph_min(p) + ph_max(p)) / 2

    def sg_state(sg, p):
        seq = sg["state_sequence"]
        c = seq[p]
        if sg["kind"] == "pedestrian":
            if c in "Gg":
                return "protected-Movement-Allowed"
            if phases[p]["name"].endswith("ped_clear") and seq[(p - 1) % nph] in "Gg":
                return "permissive-clearance"
            return "stop-And-Remain"
        green = "G" if "G" in seq else "g"
        return {"G": "protected-Movement-Allowed", "g": "permissive-Movement-Allowed", "s": "stop-Then-Proceed",
                "r": "stop-And-Remain", "y": "protected-clearance" if green == "G" else "permissive-clearance"
                }.get(c, "unavailable")

    st_tab = {sg["signal_group"]: [sg_state(sg, p) for p in range(nph)] for sg in sgs}
    cur, ph_start = int(tls["phase"][0]), times[0] - float(tls["spent"][0])
    sg_start = {}
    on_cw = np.zeros(len(times), dtype=bool)
    pe = per["edge"][pt_order]
    for k in range(len(times)):
        seg = pe[bounds[k]:bounds_r[k]]
        on_cw[k] = bool((seg == tgt["crosswalk_id"]).any())
    with gzip.open(os.path.join(a.run, "spat.jsonl.gz"), "wt") as f:
        for k, t in enumerate(times):
            ms = epoch_ms(day, t)
            p = int(tls["phase"][k])
            spent, nxt = float(tls["spent"][k]), float(tls["next_switch"][k])
            if p != cur:
                realized[cur].append(t - ph_start)
                cur, ph_start = p, t - spent
            gen = ms + int(rng.integers(*P["gen_latency_ms"]))
            mvs = []
            for sg in sgs:
                g = sg["signal_group"]
                seq = st_tab[g]
                state = seq[p]
                if g not in sg_start or sg_start[g][0] != state:
                    sg_start[g] = (state, t)
                mx = max(0.0, ph_max(p) - spent)
                mn = max(0.0, min(ph_min(p) - spent, mx))
                lk = min(max(nxt - t, mn), mx)
                j = 1
                while j < nph and seq[(p + j) % nph] == state:
                    q = (p + j) % nph
                    mn += ph_min(q); mx += ph_max(q); lk += ph_likely(q)
                    j += 1
                lk = min(max(lk, mn), mx)
                ev = {"event_state": state, "start_ms": epoch_ms(day, sg_start[g][1]), "min_end_ms": int(ms + mn * 1000),
                      "timing_valid": j < nph}
                if profile in ("full", "min_max"):
                    ev["max_end_ms"] = int(ms + mx * 1000)
                if profile == "full":
                    code = conf_code(mx - mn)
                    ev.update(likely_end_ms=int(ms + lk * 1000), confidence_code=code, confidence_p=TIC_P[code])
                ev["raw"] = {"minEndTime": timemark(ev["min_end_ms"]),
                             "maxEndTime": timemark(ev["max_end_ms"]) if "max_end_ms" in ev else None,
                             "likelyTime": timemark(ev["likely_end_ms"]) if "likely_end_ms" in ev else None}
                mvs.append({"signal_group": g, "kind": sg["kind"], "events": [ev]})
            pbd = bool(on_cw[k] and rng.random() < P["ped_detect_p"])
            f.write(json.dumps({"schema": "sav2x.spat/2.0", "intersection_id": iid, "rsu_id": "rsu07", "revision": 1,
                                "t_gen_ms": gen, "moy": minute_of_year(gen), "timestamp_ms_in_minute": int(gen % 60000),
                                "status": ["trafficDependentOperation"], "movements": mvs,
                                "maneuver_assist": [{"connection_id": tgt["crosswalk_id"], "signal_group": tgt["signal_group"],
                                                     "ped_bicycle_detect": pbd}],
                                "truth": {"phase": p, "phase_name": phases[p]["name"], "spent_s": spent}},
                               separators=(",", ":")) + "\n")
    timeline = {}
    ph_arr = tls["phase"].astype(int)
    for sg in sgs:
        g = sg["signal_group"]
        states = [st_tab[g][p] for p in ph_arr]
        segs, s0 = [], 0
        for k in range(1, len(states) + 1):
            if k == len(states) or states[k] != states[s0]:
                segs.append([states[s0], epoch_ms(day, times[s0]), epoch_ms(day, times[k - 1]) + 100])
                s0 = k
        timeline[g] = segs
    json.dump({"day": day, "epoch0_ms": epoch_ms(day, 0), "spat_profile": profile, "config": CFG,
               "signal_timeline": timeline}, open(os.path.join(a.run, "emulation.json"), "w"))
    print(f"{a.run}: BSM {len(bsm['x_m'])} rows ({int(bsm['transmitted'].sum())} transmitted), "
          f"SDSM {len(times)} msgs, SPaT {len(times)} msgs, profile {profile}")


if __name__ == "__main__":
    main()
