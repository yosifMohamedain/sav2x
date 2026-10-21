"""Build training samples for the two deployment cases from V2X message streams.

    infra   : the predictor runs at the RSU / edge computer. It sees every SDSM
              object (local, low latency), BSMs received over the air from
              connected vehicles, and SPaT from the controller. Every agent in
              the scene is a prediction target.
    vehicle : the predictor runs on a connected vehicle's OBU. It sees its own
              state at full rate (the data its BSM is built from, plus brake,
              turn-signal and yaw-rate), SDSMs and SPaT received over the air
              with loss and latency, and other vehicles' BSMs. The ego vehicle
              is the primary target.

Every history step only uses data whose receive time is <= t0, so latency and
loss show up in the inputs exactly as they would in deployment.

Output: JSONL.gz, one sample per line, matching
schemas/sample.infra.schema.json or schemas/sample.vehicle.schema.json.

Usage:
    python data/build_samples.py --run sim_out/runs/run1 --map sim_out/map_C.json \
        --case infra --assoc estimated --out sim_out/samples/run1_infra.jsonl.gz
"""
import argparse
import gzip
import json
import math
import os
import sys
import zlib

import numpy as np
import shapely
from shapely.geometry import Point

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.association import association_quality, stitch  # noqa: E402
from data.geom import IntersectionMap, footprints, occupancy_intervals  # noqa: E402

DT = 0.1
HIST_FIELDS = ["x_m", "y_m", "speed_mps", "heading_deg", "pos_conf_m", "speed_conf_mps",
               "heading_conf_deg", "age_s", "valid"]
BSM_FIELDS = ["accel_long_mps2", "yaw_rate_dps", "brake_applied", "left_turn", "right_turn", "path_radius_m"]
FUT_FIELDS = ["x_m", "y_m", "speed_mps", "valid"]
RX = {  # reception models (seconds); PER = a + b * (d / 300 m)^2
    "infra": {"sdsm_lat": (0.002, 0.010), "spat_lat": (0.001, 0.005), "bsm_lat": (0.020, 0.060),
              "bsm_per": (0.02, 0.10)},
    "vehicle": {"sdsm_lat": (0.020, 0.080), "spat_lat": (0.020, 0.080), "bsm_lat": (0.020, 0.060),
                "sdsm_per": (0.02, 0.10), "spat_per": (0.02, 0.10), "bsm_per": (0.02, 0.15),
                "self_lat": (0.0, 0.005)},
}


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def r3(v):
    return None if v is None else round(float(v), 3)


def load_jsonl(p):
    with gzip.open(p, "rt") as f:
        return [json.loads(l) for l in f]


class Streams:
    """Message streams of one run as flat numpy arrays."""

    def __init__(self, run):
        sd = load_jsonl(os.path.join(run, "sdsm.jsonl.gz"))
        rows = []
        for mi, m in enumerate(sd):
            for o in m["objects"]:
                v = o.get("veh") or {}
                rows.append((mi, m["sdsm_ts_ms"] / 1000.0, (m["sdsm_ts_ms"] + o["measurement_time_ms"]) / 1000.0,
                             o["object_id"], o["obj_type"], o["offset_x_m"], o["offset_y_m"], o["speed_mps"],
                             o["heading_deg"], o["pos_conf_m"], o["speed_conf_mps"], o["heading_conf_deg"],
                             v.get("length_m", 0.5), v.get("width_m", 0.5), o.get("truth_id", "")))
        c = list(zip(*rows))
        self.sd = {"msg": np.array(c[0]), "t_gen": np.array(c[1]), "t_obs": np.array(c[2]),
                   "oid": np.array(c[3]), "type": np.array(c[4], dtype=object), "x": np.array(c[5]),
                   "y": np.array(c[6]), "speed": np.array(c[7]), "heading": np.array(c[8]),
                   "pos_conf": np.array(c[9]), "speed_conf": np.array(c[10]), "heading_conf": np.array(c[11]),
                   "length": np.array(c[12]), "width": np.array(c[13]), "truth": np.array(c[14], dtype=object)}
        self.sd_msg_t = np.array([m["sdsm_ts_ms"] / 1000.0 for m in sd])
        bs = load_jsonl(os.path.join(run, "bsm.jsonl.gz"))
        self.bsm = {k: np.array([b[k] for b in bs]) for k in
                    ("t_obs_ms", "t_tx_ms", "x_m", "y_m", "speed_mps", "heading_deg", "accel_long_mps2",
                     "yaw_rate_dps", "brake_applied", "length_m", "width_m", "pos_acc_semi_major_m")}
        self.bsm["temp_id"] = np.array([b["temp_id"] for b in bs], dtype=object)
        self.bsm["truth"] = np.array([b["truth_id"] for b in bs], dtype=object)
        self.bsm["left"] = np.array([b["lights"]["left_turn"] for b in bs])
        self.bsm["right"] = np.array([b["lights"]["right_turn"] for b in bs])
        self.bsm["radius"] = np.array([b["path_prediction"]["radius_m"] for b in bs])
        self.bsm["t_obs"] = self.bsm["t_obs_ms"] / 1000.0
        self.bsm["t_tx"] = self.bsm["t_tx_ms"] / 1000.0
        self.spat = load_jsonl(os.path.join(run, "spat.jsonl.gz"))
        self.spat_t = np.array([s["t_gen_ms"] / 1000.0 for s in self.spat])


class GroundTruth:
    """Simulation ground truth (sim only): centers, footprints, routes, occupancy."""

    def __init__(self, run, imap, epoch0, sim_end):
        meta = json.load(open(os.path.join(run, "meta.json")))
        self.vmeta = meta["vehicles"]
        self.sim_end = epoch0 + sim_end
        v = dict(np.load(os.path.join(run, "gt_vehicles.npz"), allow_pickle=True))
        p = dict(np.load(os.path.join(run, "gt_persons.npz"), allow_pickle=True))
        self.tracks = {}
        for src, kind in ((v, "vehicle"), (p, "vru")):
            ids = src["id"]
            order = np.lexsort((src["t"], ids))
            ids_s = ids[order]
            bounds = np.flatnonzero(ids_s[1:] != ids_s[:-1]) + 1
            for g in np.split(order, bounds):
                tid = str(ids[g[0]])
                t = src["t"][g] + epoch0
                x, y, ang = src["x"][g].astype(float), src["y"][g].astype(float), src["angle"][g].astype(float)
                if kind == "vehicle":
                    L = self.vmeta[tid]["length"]
                    a = np.radians(ang)
                    x, y = x - np.sin(a) * L / 2, y - np.cos(a) * L / 2
                tr = {"kind": kind, "t": t, "x": x, "y": y, "angle": ang, "speed": src["speed"][g].astype(float)}
                if kind == "vehicle":
                    tr["lane"] = src["lane"][g]
                else:
                    tr["edge"] = src["edge"][g]
                self.tracks[tid] = tr
        self._occupancy(imap)

    def _occupancy(self, imap):
        cx, cy = imap.target_center
        polys = {"cw": imap.target_poly, **imap.zone_polys}
        for tid, tr in self.tracks.items():
            near = np.hypot(tr["x"] - cx, tr["y"] - cy) < 20.0
            tr["occ"] = {k: [] for k in polys}
            if not near.any():
                continue
            idx = np.flatnonzero(near)
            t = tr["t"][idx]
            if tr["kind"] == "vehicle":
                m = self.vmeta[tid]
                fp = footprints(tr["x"][idx], tr["y"][idx], tr["angle"][idx], m["length"], m["width"])
                for k, poly in polys.items():
                    tr["occ"][k] = occupancy_intervals(t, shapely.intersects(fp, poly))
            else:
                # a pedestrian occupies the crosswalk while on the crossing edge itself;
                # zones split that by which half of the crosswalk they are on
                on_cw = tr["edge"][idx] == imap.target_cw
                pts = shapely.points(tr["x"][idx], tr["y"][idx])
                for k, poly in polys.items():
                    ins = on_cw if k == "cw" else on_cw & shapely.intersects(pts, poly.buffer(0.3))
                    tr["occ"][k] = occupancy_intervals(t, ins)

    def state_at(self, tid, t):
        tr = self.tracks.get(tid)
        if tr is None:
            return None
        i = np.searchsorted(tr["t"], t - 1e-3)
        if i >= len(tr["t"]) or abs(tr["t"][i] - t) > 0.051:
            return None
        return i

    def future(self, tid, t0, n):
        tr = self.tracks.get(tid)
        out = [[0.0, 0.0, 0.0, 0] for _ in range(n)]
        if tr is None:
            return out
        tt = t0 + DT * np.arange(1, n + 1)
        idx = np.searchsorted(tr["t"], tt - 1e-3)
        for k, i in enumerate(idx):
            if i < len(tr["t"]) and abs(tr["t"][i] - tt[k]) < 0.051:
                out[k] = [round(tr["x"][i], 2), round(tr["y"][i], 2), round(tr["speed"][i], 2), 1]
        return out

    def crosswalk_label(self, tid, t0, horizon):
        tr = self.tracks.get(tid)
        res = {"inside_at_t0": False, "entry_s": None, "exit_s": None, "entry_censored": False,
               "exit_censored": False, "censor_s": horizon}
        zones = {}
        if tr is None:
            res["entry_censored"] = True
            res["censor_s"] = 0.0
            return res, zones
        left_sim = tr["t"][-1] < self.sim_end - 0.25
        avail = horizon if left_sim else max(0.0, min(horizon, self.sim_end - t0))
        res["censor_s"] = round(avail, 2)
        for k, ivs in tr["occ"].items():
            hit = None
            for a, b in ivs:
                if b > t0 and a <= t0 + avail:
                    hit = (max(0.0, a - t0), b - t0)
                    break
            if k == "cw":
                if hit is None:
                    res["entry_censored"] = not left_sim and avail < horizon
                else:
                    res["inside_at_t0"] = hit[0] == 0.0
                    res["entry_s"] = round(hit[0], 2)
                    if hit[1] <= avail:
                        res["exit_s"] = round(hit[1], 2)
                    else:
                        res["exit_censored"] = True
            else:
                zones[k] = None if hit is None else [round(hit[0], 2), round(min(hit[1], avail), 2)]
        return res, zones


def maneuver_of(imap, route):
    """(approach, maneuver) of a vehicle route through the intersection."""
    edges = [e for e in route]
    for c in imap.raw["connectors"]:
        fe = c["ingress_lane"].rsplit("_", 1)[0]
        te = c["egress_lane"].rsplit("_", 1)[0]
        for i in range(len(edges) - 1):
            if edges[i] == fe and edges[i + 1] == te:
                return c["approach"], c["maneuver"]
    return None, None


def pet(v, p):
    """Crosswalk-level post-encroachment time between two [in, out] intervals; 0 if they overlap."""
    if v is None or p is None:
        return None
    if v[0] >= p[1]:
        return v[0] - p[1]
    if p[0] >= v[1]:
        return p[0] - v[1]
    return 0.0


class Builder:
    def __init__(self, a):
        self.a = a
        self.imap = IntersectionMap(a.map)
        emu = json.load(open(os.path.join(a.run, "emulation.json")))
        meta = json.load(open(os.path.join(a.run, "meta.json")))
        self.epoch0 = emu["epoch0_ms"] / 1000.0
        self.st = Streams(a.run)
        self.gt = GroundTruth(a.run, self.imap, self.epoch0, meta["end"])
        self.timeline = {int(k): v for k, v in emu["signal_timeline"].items()}
        self.run_id = os.path.basename(os.path.normpath(a.run))
        self.day = emu["day"]
        self.rng = np.random.default_rng(zlib.crc32(f"{self.run_id}:{a.case}".encode()))
        sd = self.st.sd
        # tracks over SDSM detections
        if a.assoc == "ground_truth":
            uniq = {t: i for i, t in enumerate(sorted(set(sd["truth"])))}
            sd["track"] = np.array([uniq[t] for t in sd["truth"]])
        else:
            sd["track"] = stitch(sd["t_obs"], sd["oid"], sd["x"], sd["y"], sd["speed"], sd["heading"],
                                 sd["type"], sd["length"])
        self.assoc_q = association_quality(sd["track"], sd["truth"])
        self.sd_by_track = self._index(sd["track"], sd["t_obs"])
        b = self.st.bsm
        if a.assoc == "ground_truth":
            uniq = {t: i for i, t in enumerate(sorted(set(b["truth"])))}
            b["track"] = np.array([uniq[t] for t in b["truth"]])
        else:  # link rotating temp ids by kinematic continuity
            b["track"] = stitch(b["t_obs"], np.array([int(t, 16) % 65536 for t in b["temp_id"]]), b["x_m"],
                                b["y_m"], b["speed_mps"], b["heading_deg"],
                                np.array(["vehicle"] * len(b["t_obs"]), dtype=object), b["length_m"],
                                max_gap_s=0.5, gate_m=3.0)
        self.bsm_by_track = self._index(b["track"], b["t_obs"])
        self.bsm_track_truth = {k: b["truth"][v[-1]] for k, v in self.bsm_by_track.items()}

    @staticmethod
    def _index(track, t):
        out = {}
        order = np.lexsort((t, track))
        tr_s = track[order]
        bounds = np.flatnonzero(tr_s[1:] != tr_s[:-1]) + 1
        for g in np.split(order, bounds):
            out[int(track[g[0]])] = g
        return out

    # ------------------------------------------------------------ reception
    def per(self, d, ab):
        return ab[0] + ab[1] * (d / 300.0) ** 2

    def rx_sdsm(self, case, recv_xy=None):
        """Receive time per SDSM message (inf = lost) for one receiver."""
        n = len(self.st.sd_msg_t)
        lat = RX[case]["sdsm_lat"]
        t_rx = self.st.sd_msg_t + self.rng.uniform(*lat, n)
        if case == "vehicle":
            d = np.hypot(*recv_xy) if recv_xy is not None else 0.0
            lost = self.rng.random(n) < self.per(d, RX[case]["sdsm_per"])
            t_rx[lost] = np.inf
        return t_rx

    def rx_spat(self, case, recv_xy=None):
        n = len(self.st.spat_t)
        t_rx = self.st.spat_t + self.rng.uniform(*RX[case]["spat_lat"], n)
        if case == "vehicle":
            d = np.hypot(*recv_xy) if recv_xy is not None else 0.0
            t_rx[self.rng.random(n) < self.per(d, RX[case]["spat_per"])] = np.inf
        return t_rx

    def rx_bsm(self, case, recv_xy):
        b = self.st.bsm
        n = len(b["t_tx"])
        d = np.hypot(b["x_m"] - recv_xy[0], b["y_m"] - recv_xy[1])
        t_rx = b["t_tx"] + self.rng.uniform(*RX[case]["bsm_lat"], n)
        t_rx[self.rng.random(n) < self.per(d, RX[case]["bsm_per"])] = np.inf
        return t_rx

    # ------------------------------------------------------------ histories
    def history(self, idx, t_obs, t_rx, t0, fields):
        """Resample detections idx (sorted by t_obs) to the history grid, receive-time causal."""
        H = self.a.n_hist
        ok = idx[t_rx[idx] <= t0]
        hist = [[0.0] * (len(HIST_FIELDS) - 1) + [0] for _ in range(H)]
        if len(ok) == 0:
            return hist, None
        to = t_obs[ok]
        last = None
        for k in range(H):
            tau = t0 - (H - 1 - k) * DT
            j = np.searchsorted(to, tau + 1e-4) - 1
            if j < 0 or tau - to[j] > DT + 1e-4:
                continue
            i = ok[j]
            hist[k] = [r3(v) for v in fields(i)] + [r3(tau - to[j]), 1]
            last = i
        return hist, last

    def sdsm_fields(self, i):
        sd = self.st.sd
        return (sd["x"][i], sd["y"][i], sd["speed"][i], sd["heading"][i], sd["pos_conf"][i],
                sd["speed_conf"][i], sd["heading_conf"][i])

    def bsm_fields(self, i):
        b = self.st.bsm
        return (b["x_m"][i], b["y_m"][i], b["speed_mps"][i], b["heading_deg"][i],
                b["pos_acc_semi_major_m"][i], 0.1, 1.0)

    def bsm_ext(self, idx, t_rx, t0):
        b = self.st.bsm
        H = self.a.n_hist
        ok = idx[t_rx[idx] <= t0]
        out = [[0.0] * len(BSM_FIELDS) for _ in range(H)]
        if len(ok) == 0:
            return out
        to = b["t_obs"][ok]
        for k in range(H):
            tau = t0 - (H - 1 - k) * DT
            j = np.searchsorted(to, tau + 1e-4) - 1
            if j < 0 or tau - to[j] > DT + 1e-4:
                continue
            i = ok[j]
            out[k] = [r3(b["accel_long_mps2"][i]), r3(b["yaw_rate_dps"][i]), int(b["brake_applied"][i]),
                      int(b["left"][i]), int(b["right"][i]), r3(b["radius"][i])]
        return out

    # ------------------------------------------------------------ agents
    def agent_record(self, aid, source, typ, hist, last_xy_h, length, width, truth, t0, bsm_ext=None):
        x, y, spd, hd = last_xy_h
        mm = self.imap.match(x, y, hd, spd) if typ == "vehicle" else None
        lab, zones = self.gt.crosswalk_label(truth, t0, self.a.label_horizon)
        man = appr = None
        crosses = False
        if typ == "vehicle" and truth in self.gt.vmeta:
            appr, man = maneuver_of(self.imap, self.gt.vmeta[truth]["route"])
            crosses = bool(appr and self.imap.crosses_target(appr, man))
            gi = self.gt.state_at(truth, t0)
            lane_now = str(self.gt.tracks[truth]["lane"][gi]) if gi is not None else ""
            on_ingress = lane_now in self.imap.lanes and self.imap.lanes[lane_now]["role"] == "ingress"
            if not on_ingress:
                man = None  # maneuver is only a target before the stop line
        rec = {"agent_id": aid, "source": source, "type": typ, "length_m": r3(length), "width_m": r3(width),
               "map_match": mm, "hist": hist,
               "fut": self.gt.future(truth, t0, self.a.n_fut),
               "labels": {"truth_id": truth, "maneuver": man, "approach_truth": appr,
                          "crosses_target": crosses, "cw": lab, "zones": zones}}
        if bsm_ext is not None:
            rec["bsm_ext"] = bsm_ext
        return rec

    def sdsm_agents(self, t0, t_rx_msg, exclude_truth=None):
        sd = self.st.sd
        t_rx = t_rx_msg[sd["msg"]]
        agents = []
        for tid, idx in self.sd_by_track.items():
            to = sd["t_obs"][idx]
            if to[0] > t0 or to[-1] < t0 - 0.5:
                continue
            hist, last = self.history(idx, sd["t_obs"], t_rx, t0, self.sdsm_fields)
            if last is None or t0 - sd["t_obs"][last] > 0.5:
                continue
            truth = sd["truth"][last]
            if exclude_truth is not None and truth == exclude_truth:
                continue
            typ = sd["type"][last]
            agents.append(self.agent_record(
                f"sdsm_{tid}", "sdsm", typ, hist,
                (sd["x"][last], sd["y"][last], sd["speed"][last], sd["heading"][last]),
                sd["length"][last], sd["width"][last], truth, t0))
        return agents

    def bsm_agents(self, t0, t_rx, exclude_track=None):
        b = self.st.bsm
        agents = []
        for tid, idx in self.bsm_by_track.items():
            if tid == exclude_track:
                continue
            to = b["t_obs"][idx]
            if to[0] > t0 or to[-1] < t0 - 0.5:
                continue
            hist, last = self.history(idx, b["t_obs"], t_rx, t0, self.bsm_fields)
            if last is None or t0 - b["t_obs"][last] > 0.5:
                continue
            agents.append(self.agent_record(
                f"bsm_{tid}", "bsm", "vehicle", hist,
                (b["x_m"][last], b["y_m"][last], b["speed_mps"][last], b["heading_deg"][last]),
                b["length_m"][last], b["width_m"][last], b["truth"][last], t0,
                bsm_ext=self.bsm_ext(idx, t_rx, t0)))
        return agents

    def signals(self, t0, t_rx):
        i = np.searchsorted(self.st.spat_t, t0 + 1e-6) - 1
        while i >= 0 and t_rx[i] > t0:
            i -= 1
        if i < 0:
            return None
        s = self.st.spat[i]
        groups = []
        for mv in s["movements"]:
            e = mv["events"][0]

            def rel(k):
                return None if e.get(k) is None else r3(e[k] / 1000.0 - t0)
            groups.append({"signal_group": mv["signal_group"], "kind": mv["kind"], "state": e["event_state"],
                           "min_end_s": rel("min_end_ms"), "likely_end_s": rel("likely_end_ms"),
                           "max_end_s": rel("max_end_ms"), "conf_p": e.get("confidence_p"),
                           "elapsed_s": r3(t0 - e["start_ms"] / 1000.0), "timing_valid": e["timing_valid"]})
        pbd = None
        for m in s.get("maneuver_assist", []):
            if m.get("signal_group") == self.imap.target_sg:
                pbd = m.get("ped_bicycle_detect")
        return {"spat_age_s": r3(t0 - s["t_gen_ms"] / 1000.0), "groups": groups, "ped_bicycle_detect": pbd}

    def signal_truth(self, t0):
        out = []
        t0_ms = t0 * 1000
        for g, segs in self.timeline.items():
            nxt = None
            for st, a_ms, b_ms in segs:
                if a_ms <= t0_ms < b_ms:
                    nxt = (b_ms / 1000.0 - t0, None)
                elif a_ms >= t0_ms and nxt is not None and nxt[1] is None:
                    nxt = (nxt[0], st)
                    break
            if nxt:
                out.append({"signal_group": g, "next_change_s": r3(nxt[0]), "next_state": nxt[1]})
        return out

    def conflicts(self, agents):
        out = []
        vs = [a for a in agents if a["type"] == "vehicle"]
        ps = [a for a in agents if a["type"] == "vru"]
        for v in vs:
            lv = v["labels"]["cw"]
            if lv["entry_s"] is None:
                continue
            vi = [lv["entry_s"], lv["exit_s"] if lv["exit_s"] is not None else lv["censor_s"]]
            for p in ps:
                lp = p["labels"]["cw"]
                if lp["entry_s"] is None:
                    continue
                pi = [lp["entry_s"], lp["exit_s"] if lp["exit_s"] is not None else lp["censor_s"]]
                pz = None
                for z in ("inbound", "outbound"):
                    a_, b_ = v["labels"]["zones"].get(z), p["labels"]["zones"].get(z)
                    q = pet(a_, b_)
                    if q is not None:
                        pz = q if pz is None else min(pz, q)
                out.append({"vehicle_id": v["agent_id"], "ped_id": p["agent_id"], "pet_cw_s": r3(pet(vi, pi)),
                            "pet_zone_s": r3(pz), "v_entry_s": vi[0], "v_exit_s": r3(vi[1]),
                            "p_entry_s": pi[0], "p_exit_s": r3(pi[1])})
        return out

    def header(self, t0, case):
        return {"schema": f"sav2x.sample.{case}/1.0", "sample_id": f"{self.run_id}_{case}_{int(t0 * 1000)}",
                "case": case, "origin": "sim", "run_id": self.run_id, "day": self.day,
                "intersection_id": self.imap.raw["intersection_id"], "map_version": self.imap.version,
                "target_crosswalk_id": self.imap.target_cw, "target_signal_group": self.imap.target_sg,
                "t0_ms": int(round(t0 * 1000)), "dt_ms": 100, "n_hist": self.a.n_hist, "n_fut": self.a.n_fut,
                "label_horizon_s": self.a.label_horizon, "bin_s": self.a.bin_s, "assoc": self.a.assoc,
                "hist_fields": HIST_FIELDS, "bsm_fields": BSM_FIELDS, "fut_fields": FUT_FIELDS}

    def cap(self, agents, k):
        cx, cy = self.imap.target_center

        def d(a):
            h = [r for r in a["hist"] if r[-1]]
            return math.hypot(h[-1][0] - cx, h[-1][1] - cy) if h else 1e9
        return sorted(agents, key=d)[:k]

    # ------------------------------------------------------------ cases
    def build_infra(self, out):
        t_sd = self.rx_sdsm("infra")
        t_sp = self.rx_spat("infra")
        t_bsm = self.rx_bsm("infra", (0.0, 0.0))
        t0s = np.arange(self.st.spat_t[0] + self.a.n_hist * DT + 0.5, self.st.spat_t[-1] - 1.0, self.a.stride)
        n = 0
        for t0 in t0s:
            t0 = round(float(t0), 1)
            agents = self.sdsm_agents(t0, t_sd) + self.bsm_agents(t0, t_bsm)
            if not agents:
                continue
            agents = self.cap(agents, self.a.max_agents)
            sig = self.signals(t0, t_sp)
            if sig is None:
                continue
            rec = self.header(t0, "infra")
            rec.update({"receiver": {"type": "rsu", "id": "rsu07"}, "agents": agents, "signals": sig,
                        "signal_truth": self.signal_truth(t0), "conflicts": self.conflicts(agents)})
            out.write(json.dumps(rec, separators=(",", ":"), default=_np) + "\n")
            n += 1
        return n

    def build_vehicle(self, out):
        b = self.st.bsm
        n = 0
        for ego_tid, idx in self.bsm_by_track.items():
            to = b["t_obs"][idx]
            t_rx_self = b["t_obs"] + self.rng.uniform(*RX["vehicle"]["self_lat"], len(b["t_obs"]))
            ego_xy = (float(np.median(b["x_m"][idx])), float(np.median(b["y_m"][idx])))
            t_sd = self.rx_sdsm("vehicle", ego_xy)
            t_sp = self.rx_spat("vehicle", ego_xy)
            t_bsm = self.rx_bsm("vehicle", ego_xy)
            for t0 in np.arange(to[0] + 1.5, to[-1] - 0.5, self.a.stride):
                t0 = round(float(t0), 1)
                hist, last = self.history(idx, b["t_obs"], t_rx_self, t0, self.bsm_fields)
                if last is None or math.hypot(b["x_m"][last], b["y_m"][last]) > self.a.ego_radius:
                    continue
                truth = b["truth"][last]
                ego = self.agent_record(f"bsm_{ego_tid}", "self", "vehicle", hist,
                                        (b["x_m"][last], b["y_m"][last], b["speed_mps"][last], b["heading_deg"][last]),
                                        b["length_m"][last], b["width_m"][last], truth, t0,
                                        bsm_ext=self.bsm_ext(idx, t_rx_self, t0))
                others = self.sdsm_agents(t0, t_sd, exclude_truth=truth) + \
                    self.bsm_agents(t0, t_bsm, exclude_track=ego_tid)
                others = self.cap(others, self.a.max_agents - 1)
                sig = self.signals(t0, t_sp)
                rec = self.header(t0, "vehicle")
                rec["sample_id"] += f"_{ego_tid}"
                rec.update({"receiver": {"type": "obu", "id": f"bsm_{ego_tid}"}, "ego": ego, "agents": others,
                            "signals": sig, "signal_truth": self.signal_truth(t0),
                            "conflicts": [c for c in self.conflicts([ego] + others)
                                          if c["vehicle_id"] == ego["agent_id"]]})
                out.write(json.dumps(rec, separators=(",", ":"), default=_np) + "\n")
                n += 1
        return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--case", choices=["infra", "vehicle"], required=True)
    ap.add_argument("--assoc", choices=["ground_truth", "estimated"], default="estimated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-hist", type=int, default=30)
    ap.add_argument("--n-fut", type=int, default=80)
    ap.add_argument("--label-horizon", type=float, default=20.0)
    ap.add_argument("--bin-s", type=float, default=0.5)
    ap.add_argument("--stride", type=float, default=1.0, help="seconds between sample times")
    ap.add_argument("--max-agents", type=int, default=48)
    ap.add_argument("--ego-radius", type=float, default=130.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    bld = Builder(a)
    with gzip.open(a.out, "wt") as f:
        n = bld.build_infra(f) if a.case == "infra" else bld.build_vehicle(f)
    stats = {"samples": n, "case": a.case, "assoc": a.assoc, "association_quality": bld.assoc_q}
    json.dump(stats, open(a.out.replace(".jsonl.gz", ".stats.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
