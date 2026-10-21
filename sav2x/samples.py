"""Build vehicle-side training samples (schemas/sample.vehicle.schema.json) for one run.

In simulation every vehicle serves as an ego: its own BSM data gives its kinematics
at full rate. Other inputs are what that vehicle would have received by t0:
  * SPaT          over the air (loss grows with distance to the RSU, 20-80 ms latency)
  * SDSM          over the air, reduced to pedestrian counts per MAP presence zone
  * other BSMs    over the air, reduced to the nearest connected vehicle ahead in lane
Labels come from simulation ground truth.

Usage (one run):
    python -m sav2x.samples --run data_out/runs/base_s1 --out data_out/samples/base_s1.jsonl.gz
"""
import argparse
import gzip
import json
import math
import os
import zlib

import numpy as np
import shapely
from shapely.geometry import LineString, Point

from sav2x.geom import MANEUVERS, IntersectionMap, footprints, intervals

DT = 0.1
HIST_FIELDS = ["x_m", "y_m", "speed_mps", "heading_deg", "accel_long_mps2", "yaw_rate_dps", "brake_applied",
               "left_turn", "right_turn", "path_radius_m", "pos_acc_m", "valid"]
STATE_NAMES = ["unavailable", "dark", "stop-Then-Proceed", "stop-And-Remain", "pre-Movement",
               "permissive-Movement-Allowed", "protected-Movement-Allowed", "permissive-clearance",
               "protected-clearance", "caution-Conflicting-Traffic"]
RX = {"spat_lat": (0.020, 0.080), "sdsm_lat": (0.020, 0.080), "bsm_lat": (0.020, 0.060),
      "per_a": 0.02, "per_b": 0.10, "range_m": 300.0}
DEST = {"N": {"W": "right", "S": "straight", "E": "left"}, "E": {"N": "right", "W": "straight", "S": "left"},
        "S": {"E": "right", "N": "straight", "W": "left"}, "W": {"S": "right", "E": "straight", "N": "left"}}


def r3(v):
    return None if v is None else round(float(v), 3)


def per(d):
    return RX["per_a"] + RX["per_b"] * (np.asarray(d) / RX["range_m"]) ** 2


class RunData:
    """Everything one run provides, indexed for fast per-sample lookups. Times are simulation seconds."""

    def __init__(self, run, imap: IntersectionMap, theta):
        self.run = run
        self.m = imap
        self.theta = theta
        emu = json.load(open(os.path.join(run, "emulation.json")))
        self.sc = json.load(open(os.path.join(run, "scenario.json")))
        meta = json.load(open(os.path.join(run, "meta.json")))
        self.epoch0 = emu["epoch0_ms"] / 1000.0
        self.timeline = {int(k): v for k, v in emu["signal_timeline"].items()}
        self.vmeta = meta["vehicles"]
        self.t_end = meta["end"]
        self._gt(run)
        self._bsm(run)
        self._sdsm(run)
        self._spat(run)

    def tsim(self, ms):
        return np.asarray(ms) / 1000.0 - self.epoch0

    # ---------------- ground truth
    def _gt(self, run):
        v = dict(np.load(os.path.join(run, "gt_vehicles.npz"), allow_pickle=True))
        p = dict(np.load(os.path.join(run, "gt_persons.npz"), allow_pickle=True))
        self.gt = {}
        order = np.lexsort((v["t"], v["id"]))
        ids = v["id"][order]
        for g in np.split(order, np.flatnonzero(ids[1:] != ids[:-1]) + 1):
            vid = str(v["id"][g[0]])
            m = self.vmeta[vid]
            a = np.radians(v["angle"][g].astype(np.float64))
            x = v["x"][g] - np.sin(a) * m["length"] / 2
            y = v["y"][g] - np.cos(a) * m["length"] / 2
            t = np.round(v["t"][g], 1)
            tr = {"t": t, "x": x, "y": y, "speed": v["speed"][g].astype(np.float64), "occ": {}}
            near = np.hypot(x, y) < 30.0
            if near.any():
                idx = np.flatnonzero(near)
                fp = footprints(x[idx], y[idx], v["angle"][g][idx], m["length"], m["width"])
                tr["occ"]["cw"] = intervals(t[idx], shapely.intersects(fp, self.m.cw_poly))
                for z, poly in self.m.zone_polys.items():
                    tr["occ"][z] = intervals(t[idx], shapely.intersects(fp, poly))
            else:
                tr["occ"] = {"cw": [], "inbound": [], "outbound": []}
            route = m["route"]
            tr["approach"] = route[0][0] if route else None
            tr["maneuver"] = DEST.get(route[0][0], {}).get(route[1][0]) if len(route) > 1 else None
            self.gt[vid] = tr
        # pedestrians on the target crosswalk: per-person intervals per half, and an occupancy timeline
        self.t_grid0 = float(np.round(min(v["t"].min(), p["t"].min() if len(p["t"]) else 1e9), 1))
        n = int(round((self.t_end - self.t_grid0) / DT)) + 2
        self.ped_tl = {z: np.zeros(n, dtype=bool) for z in self.m.zone_polys}
        self.ped_iv = {z: [] for z in self.m.zone_polys}
        on = p["edge"] == self.m.target_cw
        if on.any():
            idx = np.flatnonzero(on)
            pid = p["id"][idx]
            order = np.lexsort((p["t"][idx], pid))
            idx, pid = idx[order], pid[order]
            for g in np.split(np.arange(len(idx)), np.flatnonzero(pid[1:] != pid[:-1]) + 1):
                ii = idx[g]
                t = np.round(p["t"][ii], 1)
                for z, poly in self.m.zone_polys.items():
                    ins = shapely.contains_xy(poly.buffer(0.3), p["x"][ii], p["y"][ii])
                    self.ped_iv[z] += intervals(t, ins)
                    k = np.round((t[ins] - self.t_grid0) / DT).astype(int)
                    self.ped_tl[z][k] = True
        for z in self.ped_iv:
            self.ped_iv[z] = np.array(self.ped_iv[z], dtype=np.float64).reshape(-1, 2)

    # ---------------- BSM
    def _bsm(self, run):
        b = dict(np.load(os.path.join(run, "bsm.npz"), allow_pickle=True))
        b["t"] = np.round(self.tsim(b["t_obs_ms"]), 1)
        b["t_tx"] = self.tsim(b["t_tx_ms"])
        self.b = b
        order = np.lexsort((b["t"], b["veh_id"]))
        ids = b["veh_id"][order]
        self.bsm_rows = {str(ids[g[0]]): order[g] for g in np.split(np.arange(len(order)), np.flatnonzero(ids[1:] != ids[:-1]) + 1)}
        tx = np.flatnonzero(b["transmitted"])
        self.tx = tx[np.argsort(b["t"][tx], kind="stable")]
        self.tx_t = b["t"][self.tx]

    # ---------------- SDSM -> presence counts per message
    def _sdsm(self, run):
        meas, counts = [], []
        with gzip.open(os.path.join(run, "sdsm.jsonl.gz"), "rt") as f:
            for line in f:
                s = json.loads(line)
                objs = [o for o in s["objects"] if o["obj_type"] == "vru"]
                off = objs[0]["measurement_time_ms"] if objs else 0
                meas.append(self.tsim(s["sdsm_ts_ms"] + off))
                xs = np.array([o["offset_x_m"] for o in objs], dtype=float)
                ys = np.array([o["offset_y_m"] for o in objs], dtype=float)
                counts.append(self.m.presence_counts(xs, ys))
        self.sd_t = np.asarray(meas, dtype=np.float64)
        self.sd_gen = self.sd_t + 0.06
        self.sd_counts = np.stack(counts)

    # ---------------- SPaT
    def _spat(self, run):
        rows = []
        with gzip.open(os.path.join(run, "spat.jsonl.gz"), "rt") as f:
            for line in f:
                rows.append(json.loads(line))
        self.sp_gen = self.tsim([r["t_gen_ms"] for r in rows])
        sgs = [mv["signal_group"] for mv in rows[0]["movements"]]
        self.sp_sg = {g: i for i, g in enumerate(sgs)}
        G, N = len(sgs), len(rows)
        self.sp_state = np.zeros((N, G), dtype=np.int8)
        self.sp_val = np.full((N, G, 5), np.nan)       # min, likely, max, conf_p, start (sim s)
        self.sp_valid = np.zeros((N, G), dtype=bool)
        self.sp_pbd = np.full(N, -1, dtype=np.int8)
        for i, r in enumerate(rows):
            for j, mv in enumerate(r["movements"]):
                e = mv["events"][0]
                self.sp_state[i, j] = STATE_NAMES.index(e["event_state"])
                self.sp_val[i, j] = [self.tsim(e["min_end_ms"]),
                                     self.tsim(e["likely_end_ms"]) if "likely_end_ms" in e else np.nan,
                                     self.tsim(e["max_end_ms"]) if "max_end_ms" in e else np.nan,
                                     e.get("confidence_p", np.nan), self.tsim(e["start_ms"])]
                self.sp_valid[i, j] = e.get("timing_valid", True)
            for ma in r.get("maneuver_assist", []):
                if ma.get("signal_group") == self.m.target_sg and ma.get("ped_bicycle_detect") is not None:
                    self.sp_pbd[i] = int(ma["ped_bicycle_detect"])


class EgoBuilder:
    def __init__(self, rd: RunData, args):
        self.rd, self.m, self.a = rd, rd.m, args
        self.run_id = os.path.basename(os.path.normpath(rd.run))

    def _rng(self, vid):
        return np.random.default_rng(zlib.crc32(f"{self.run_id}:{vid}:rx".encode()))

    @staticmethod
    def _latest(t_rx, t_val, tau):
        """For each tau, index of the message with the largest t_val among those with t_rx <= tau (or -1)."""
        order = np.argsort(t_rx, kind="stable")
        tr, tv = t_rx[order], t_val[order]
        tv = np.where(np.isfinite(tr), tv, -np.inf)
        best = np.maximum.accumulate(tv)
        arg = np.maximum.accumulate(np.where(tv >= best, np.arange(len(tv)), 0))   # running argmax
        k = np.searchsorted(tr, tau, side="right") - 1
        out = np.where(k >= 0, order[arg[np.maximum(k, 0)]], -1)
        out[(k < 0) | ~np.isfinite(best[np.maximum(k, 0)])] = -1
        return out

    def build_vehicle(self, vid, out):
        rd, m, a = self.rd, self.m, self.a
        rows = rd.bsm_rows.get(vid)
        gt = rd.gt.get(vid)
        if rows is None or gt is None or len(rows) < 15:
            return 0
        b = rd.b
        t = b["t"][rows]
        x, y = b["x_m"][rows].astype(float), b["y_m"][rows].astype(float)
        rng = self._rng(vid)
        t_lo, t_hi = t[0] - 1.0, t[-1]
        # reception of SPaT and SDSM for this receiver (distance at message time)
        def rx_times(gen, lat):
            sel = np.flatnonzero((gen >= t_lo) & (gen <= t_hi + 0.2))
            j = np.clip(np.searchsorted(t, gen[sel]), 0, len(t) - 1)
            d = np.hypot(x[j], y[j])
            t_rx = np.full(len(gen), np.inf)
            ok = rng.random(len(sel)) >= per(d)
            t_rx[sel[ok]] = gen[sel[ok]] + rng.uniform(*lat, ok.sum())
            return t_rx
        sp_rx = rx_times(rd.sp_gen, RX["spat_lat"])
        sd_rx = rx_times(rd.sd_gen, RX["sdsm_lat"])
        stride = int(round(a.stride / DT))
        k0 = int(round(t[0] / DT))
        cand = [i for i in range(len(t)) if (int(round(t[i] / DT)) - k0) % stride == 0 and i >= 10]
        n = 0
        for i in cand:
            t0 = float(t[i])
            if math.hypot(x[i], y[i]) > a.max_range:
                continue
            # the approach the vehicle came from, from its own BSM history (3 s back)
            j0 = max(0, i - (a.n_hist - 1))
            mm0 = m.match(x[j0], y[j0], b["heading_deg"][rows[j0]], b["speed_mps"][rows[j0]])
            appr = mm0["approach"] if mm0 is not None and mm0["role"] in ("ingress", "connector") else None
            mm = m.match(x[i], y[i], b["heading_deg"][rows[i]], b["speed_mps"][rows[i]], approach=appr)
            if mm is None or mm["role"] == "egress" or \
                    (mm["role"] == "ingress" and mm["dist_to_stopline_m"] > a.max_dist_stopline):
                continue
            gi = np.searchsorted(gt["t"], t0 - 1e-3)
            if gi >= len(gt["t"]) or abs(gt["t"][gi] - t0) > 0.05:
                continue
            rec = self.sample(vid, rows, i, t0, mm, gt, gi, sp_rx, sd_rx, rng)
            if rec is not None:
                out.write(json.dumps(rec, separators=(",", ":")) + "\n")
                n += 1
        return n

    def sample(self, vid, rows, i, t0, mm, gt, gi, sp_rx, sd_rx, rng):
        rd, m, a, b = self.rd, self.m, self.a, self.rd.b
        H, NF, NB = a.n_hist, a.n_fut, int(round(a.horizon / a.bin_s))
        # ---- ego history from its own BSM data
        hist = []
        for k in range(H):
            j = i - (H - 1 - k)
            if j < 0:
                hist.append([0.0] * 11 + [0])
                continue
            r = rows[j]
            hist.append([r3(b["x_m"][r]), r3(b["y_m"][r]), r3(b["speed_mps"][r]), r3(b["heading_deg"][r]),
                         r3(b["accel_long_mps2"][r]), r3(b["yaw_rate_dps"][r]), int(b["brake_applied"][r]),
                         int(b["left_turn"][r]), int(b["right_turn"][r]), r3(b["path_radius_m"][r]),
                         r3(b["pos_acc_semi_major_m"][r]), 1])
        r_last = rows[i]
        px, py = float(b["x_m"][r_last]), float(b["y_m"][r_last])
        # ---- candidate routes from MAP
        routes = []
        P = Point(px, py)
        for cid, lane_change in m.candidate_routes(mm, px, py, float(b["heading_deg"][r_last])):
            c = m.conns[cid]
            line = m.route_line[cid]
            meta = m.route_meta[cid]
            s0 = line.project(P)
            ss = s0 + a.path_spacing * np.arange(a.path_points)
            pts = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(ss, line.length)))
            over = ss > line.length                       # extrapolate past the end along the last segment
            if over.any():
                cc = np.asarray(line.coords)
                d = (cc[-1] - cc[-2]) / (np.linalg.norm(cc[-1] - cc[-2]) + 1e-9)
                pts[over] = cc[-1] + np.outer(ss[over] - line.length, d)
            zones = m.crossing.get(cid, [])
            routes.append({"maneuver": c["maneuver"], "connection_id": cid, "signal_group": c["signal_group"],
                           "lane_change": bool(lane_change), "crosses_target": bool(zones), "zones": zones,
                           "s_stopline_m": r3(meta["s_stop"] - s0) if meta["s_stop"] - s0 > -0.5 else None,
                           "s_cw_entry_m": r3(meta["s_cw_in"] - s0) if meta["s_cw_in"] is not None else None,
                           "s_cw_exit_m": r3(meta["s_cw_out"] - s0) if meta["s_cw_out"] is not None else None,
                           "path_spacing_m": a.path_spacing, "path": [[r3(u), r3(v)] for u, v in pts]})
        # ---- SPaT received by t0
        sgs = sorted({r["signal_group"] for r in routes} | {m.target_sg})
        j = self.rd_latest(rd.sp_gen, sp_rx, t0)
        if j >= 0:
            groups = []
            for g in sgs:
                c = rd.sp_sg[g]
                mn, lk, mx, cp, st = rd.sp_val[j, c]
                groups.append({"signal_group": g, "kind": m.sgs[g]["kind"], "state": STATE_NAMES[rd.sp_state[j, c]],
                               "min_end_s": r3(mn - t0), "likely_end_s": None if np.isnan(lk) else r3(lk - t0),
                               "max_end_s": None if np.isnan(mx) else r3(mx - t0),
                               "conf_p": None if np.isnan(cp) else r3(cp), "elapsed_s": r3(t0 - st),
                               "timing_valid": bool(rd.sp_valid[j, c])})
            pbd = int(rd.sp_pbd[j])
            signals = {"spat_age_s": r3(t0 - rd.sp_gen[j]), "groups": groups,
                       "ped_bicycle_detect": None if pbd < 0 else bool(pbd)}
        else:
            signals = None
        # ---- pedestrian presence from SDSM received by each history step
        taus = t0 - DT * np.arange(H - 1, -1, -1)
        idx = self._latest(sd_rx, rd.sd_t, taus)
        counts, valid = [], []
        for tau, k in zip(taus, idx):
            ok = k >= 0 and tau - rd.sd_t[k] <= 0.25 + 1e-6
            counts.append([int(v) for v in rd.sd_counts[k]] if ok else [0] * len(m.presence_names))
            valid.append(int(ok))
        sd_age = r3(t0 - rd.sd_t[idx[-1]]) if idx[-1] >= 0 else None
        # ---- leader: nearest received connected vehicle ahead in the ego's lane
        leader = {"present": False}
        if mm["role"] == "ingress" and not a.no_leader:
            lo, hi = np.searchsorted(rd.tx_t, t0 - 0.4), np.searchsorted(rd.tx_t, t0 + 1e-6)
            cand = rd.tx[lo:hi]
            cand = cand[b["veh_id"][cand] != vid]
            if len(cand):
                t_rx = b["t_tx"][cand] + rng.uniform(*RX["bsm_lat"], len(cand))
                d = np.hypot(b["x_m"][cand] - px, b["y_m"][cand] - py)
                ok = (t_rx <= t0) & (rng.random(len(cand)) >= per(d)) & (d < 110)
                cand = cand[ok]
                if len(cand):
                    # latest per vehicle
                    _, first = np.unique(b["veh_id"][cand][::-1], return_index=True)
                    cand = cand[::-1][first]
                    lane = m.lanes[mm["lane_id"]]
                    ln = LineString(lane["centerline"])
                    pts = shapely.points(b["x_m"][cand], b["y_m"][cand])
                    s = shapely.line_locate_point(ln, pts)
                    lat = shapely.distance(ln, pts)
                    gap = s - mm["s_m"] - b["length_m"][cand] / 2 - b["length_m"][rows[i]] / 2
                    sel = np.flatnonzero((s - mm["s_m"] > 0.5) & (gap < 100) & (lat < 1.6))
                    if len(sel):
                        q = cand[sel[np.argmin(gap[sel])]]
                        leader = {"present": True, "gap_m": r3(max(0.0, gap[sel].min())), "speed_mps": r3(b["speed_mps"][q]),
                                  "accel_long_mps2": r3(b["accel_long_mps2"][q]), "brake_applied": bool(b["brake_applied"][q]),
                                  "age_s": r3(t0 - b["t"][q])}
        # ---- labels from ground truth
        tt = t0 + DT * np.arange(1, a.n_fut + 1)
        gidx = np.searchsorted(gt["t"], tt - 1e-3)
        fut = []
        for k, g in enumerate(gidx):
            if g < len(gt["t"]) and abs(gt["t"][g] - tt[k]) < 0.05:
                fut.append([r3(gt["x"][g]), r3(gt["y"][g]), r3(gt["speed"][g]), 1])
            else:
                fut.append([0.0, 0.0, 0.0, 0])
        left_sim = gt["t"][-1] < rd.t_end - 0.25
        avail = a.horizon if left_sim else max(0.0, min(a.horizon, rd.t_end - t0))

        def first_after(ivs):
            for s_, e_ in ivs:
                if e_ > t0 and s_ <= t0 + avail:
                    return [max(0.0, s_ - t0), e_ - t0]
            return None
        cw_iv = first_after(gt["occ"].get("cw", []))
        cw = {"inside_at_t0": bool(cw_iv is not None and cw_iv[0] == 0.0), "entry_s": None, "exit_s": None,
              "entry_censored": False, "exit_censored": False, "censor_s": r3(avail)}
        if cw_iv is None:
            cw["entry_censored"] = (not left_sim) and avail < a.horizon
        else:
            cw["entry_s"] = r3(cw_iv[0])
            if cw_iv[1] <= avail:
                cw["exit_s"] = r3(cw_iv[1])
            else:
                cw["exit_censored"] = True
        zones = {z: first_after(gt["occ"].get(z, [])) for z in m.zone_polys}
        zones = {z: None if v is None else [r3(v[0]), r3(min(v[1], avail))] for z, v in zones.items()}
        # pedestrian occupancy per half and bin
        nvalid = int(min(NB, math.floor(avail / a.bin_s + 1e-6)))
        occ = {}
        for z, tl in rd.ped_tl.items():
            k0 = int(round((t0 - rd.t_grid0) / DT))
            per_bin = int(round(a.bin_s / DT))
            seg = tl[k0 + 1:k0 + 1 + NB * per_bin]
            seg = np.pad(seg, (0, NB * per_bin - len(seg)))
            occ[z] = [int(v) for v in seg.reshape(NB, per_bin).any(1)]
        # conflict: same half within theta of a pedestrian
        best = (None, None)
        for z, iv in zones.items():
            if iv is None or len(rd.ped_iv[z]) == 0:
                continue
            e0, e1 = t0 + iv[0], t0 + iv[1]
            P_ = rd.ped_iv[z]
            pet = np.maximum(0.0, np.maximum(P_[:, 0] - e1, e0 - P_[:, 1]))
            k = int(np.argmin(pet))
            if best[0] is None or pet[k] < best[0]:
                best = (float(pet[k]), z)
        known = (cw["entry_s"] is None and not cw["entry_censored"]) or (cw["exit_s"] is not None)
        conflict = {"label": bool(best[0] is not None and best[0] <= rd.theta), "theta_s": rd.theta,
                    "min_pet_s": r3(best[0]), "zone": best[1], "ego_entry_s": cw["entry_s"], "known": bool(known)}
        man = gt["maneuver"] if (mm["role"] == "ingress" or mm["role"] == "connector") else None
        appr = gt["approach"]
        labels = {"truth_id": vid, "fut": fut, "maneuver": man,
                  "crosses_target": bool(appr and man and m.crosses(appr, man)), "cw": cw, "zones": zones,
                  "ped_occupancy": {"inbound": occ["inbound"], "outbound": occ["outbound"], "valid_bins": nvalid},
                  "conflict": conflict}
        # ---- realized next change of the sample's signal groups (analysis only)
        st = []
        t0_ms = (rd.epoch0 + t0) * 1000
        for g in sgs:
            for state, s_ms, e_ms in rd.timeline[g]:
                if s_ms <= t0_ms < e_ms:
                    st.append({"signal_group": g, "next_change_s": r3(e_ms / 1000.0 - rd.epoch0 - t0)})
                    break
        return {"schema": "sav2x.sample.vehicle/2.0",
                "sample_id": f"{self.run_id}_{vid}_{int(round(t0 * 10))}", "origin": "sim",
                "scenario": rd.sc["name"], "run_id": self.run_id, "seed": rd.sc["seed"],
                "intersection_id": m.raw["intersection_id"], "map_version": m.version,
                "target_crosswalk_id": m.target_cw, "target_signal_group": m.target_sg,
                "crosswalk_length_m": m.target["length_m"],
                "t0_ms": int(round((rd.epoch0 + t0) * 1000)), "dt_ms": 100, "n_hist": a.n_hist, "n_fut": a.n_fut,
                "label_horizon_s": a.horizon, "bin_s": a.bin_s,
                "ego": {"temp_id": str(b["temp_id"][r_last]), "length_m": r3(b["length_m"][r_last]),
                        "width_m": r3(b["width_m"][r_last]), "hist_fields": HIST_FIELDS, "hist": hist, "map_match": mm},
                "routes": routes, "signals": signals,
                "ped_presence": {"zones": m.presence_names, "count": counts, "valid": valid, "sdsm_age_s": sd_age},
                "leader_bsm": leader, "labels": labels, "signal_truth": st}

    def rd_latest(self, gen, t_rx, t0):
        ok = np.flatnonzero(t_rx <= t0)
        if len(ok) == 0:
            return -1
        return int(ok[np.argmax(gen[ok])])


def build(run, out_path, args):
    imap = IntersectionMap(os.path.join(run, "map.json"))
    rd = RunData(run, imap, args.theta)
    eb = EgoBuilder(rd, args)
    n = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        for vid in sorted(rd.bsm_rows):
            n += eb.build_vehicle(vid, f)
    return n


def add_args(ap):
    ap.add_argument("--stride", type=float, default=1.0, help="seconds between decision times per vehicle")
    ap.add_argument("--n-hist", type=int, default=30)
    ap.add_argument("--n-fut", type=int, default=80)
    ap.add_argument("--horizon", type=float, default=20.0, help="label horizon for crosswalk and conflict labels (s)")
    ap.add_argument("--bin-s", type=float, default=0.5)
    ap.add_argument("--theta", type=float, default=1.0, help="conflict margin (s)")
    ap.add_argument("--path-spacing", type=float, default=4.0)
    ap.add_argument("--path-points", type=int, default=31)
    ap.add_argument("--max-range", type=float, default=130.0, help="ego distance from the intersection center (m)")
    ap.add_argument("--max-dist-stopline", type=float, default=100.0)
    ap.add_argument("--no-leader", action="store_true")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    add_args(ap)
    a = ap.parse_args()
    n = build(a.run, a.out, a)
    print(f"{a.run}: {n} samples -> {a.out}")


if __name__ == "__main__":
    main()
