"""Non-learned baselines on the same vehicle-side inputs.

Trajectory and arrival:
    cv     constant velocity along the current BSM heading
    rule   follow the most likely route (turn signal if on, else straight) on its MAP path;
           stop at the stop line on red / clearance or behind a stopped leader (BSM), go at
           likelyTime (or minEnd); accelerate at 2 m/s^2 to the speed limit

Conflict score (both): the pedestrian-window rule, i.e. the original idea of comparing
the vehicle's arrival with how long pedestrians need to clear the crossing, with
MUTCD's design walking speed of 1.07 m/s (3.5 ft/s):
    pedestrian on the crosswalk       occupied until it reaches the far curb
    waiting at a curb during WALK     occupied from +1 s for length / 1.07 s
    waiting during DON'T WALK         occupied from the next WALK (likelyTime or minEnd) + 1 s
score = 1 when the vehicle's crosswalk interval comes within theta of a pedestrian
window, decaying as exp(-gap / 1 s) beyond it.
"""
import math

import numpy as np
import shapely
from shapely.geometry import Polygon

from sav2x.geom import footprints

DT = 0.1
H_S = 20.0
A_ACC, A_STOP_MAX, A_COMF = 2.0, 4.5, 1.5
RED = {"stop-And-Remain", "permissive-clearance", "protected-clearance"}
V_PED = 1.07


def last_state(s):
    h = [r for r in s["ego"]["hist"] if r[11]]
    return h[-1]


def crossing_interval(xs, ys, yaws_deg, poly, L, W):
    fp = footprints(xs, ys, yaws_deg, L, W)
    ins = shapely.intersects(fp, poly)
    if not ins.any():
        return math.inf, math.inf
    i0 = int(np.argmax(ins))
    rest = ins[i0:]
    i1 = i0 + (int(np.argmin(rest)) if (~rest).any() else len(rest))
    return i0 * DT, (i1 * DT if i1 < len(ins) else H_S + 5.0)


class Baselines:
    def __init__(self, maps):
        """maps: run_id -> IntersectionMap-like dict (raw map json)."""
        self.maps = maps
        self.cache = {}

    def poly(self, run_id):
        if run_id not in self.cache:
            m = self.maps[run_id]
            cw = [c for c in m["crosswalks"] if c["is_target"]][0]
            self.cache[run_id] = Polygon(cw["polygon"])
        return self.cache[run_id]

    # ------------------------------------------------------------ trajectories
    def cv(self, s):
        x, y, v, hd = last_state(s)[:4]
        yaw = math.radians(90 - hd)
        t = DT * np.arange(int(H_S / DT) + 1)
        xs, ys = x + v * math.cos(yaw) * t, y + v * math.sin(yaw) * t
        tin, tout = crossing_interval(xs, ys, np.full(len(t), hd), self.poly(s["run_id"]),
                                      s["ego"]["length_m"], s["ego"]["width_m"])
        return xs, ys, tin, tout, None

    def pick_route(self, s):
        routes = {r["maneuver"]: r for r in s["routes"]}
        st = last_state(s)
        if st[7] and "left" in routes:
            return routes["left"]
        if st[8] and "right" in routes:
            return routes["right"]
        if "straight" in routes:
            return routes["straight"]
        return s["routes"][0]

    def rule(self, s):
        r = self.pick_route(s)
        x, y, v, hd = last_state(s)[:4]
        L = s["ego"]["length_m"]
        sig = {g["signal_group"]: g for g in (s["signals"] or {}).get("groups", [])}
        g = sig.get(r["signal_group"])
        state = g["state"] if g else "protected-Movement-Allowed"
        t_change = (g["likely_end_s"] if g["likely_end_s"] is not None else g["min_end_s"]) if g else None
        vmax = 13.4
        s_stop = r["s_stopline_m"]
        stop_at, go = None, math.inf
        if s_stop is not None:
            front = s_stop - L / 2 - 1.0
            # queue discharge: vehicles ahead ~ distance to the stop line / 7.5 m jam spacing,
            # each adding a ~1.8 s start-up headway after the light turns green
            n_ahead = int(max(0.0, front) // 7.5) if v < 1.0 else 0
            delay = 1.0 + 1.8 * n_ahead
            if state in RED and (v * v / (2 * max(front, 0.1)) <= A_STOP_MAX or v < 1.0):
                stop_at = 0.0 if (v < 1.0 and n_ahead > 0) else front
                go = t_change + delay if (state == "stop-And-Remain" and t_change is not None) else math.inf
            elif state == "stop-Then-Proceed":
                stop_at, go = front, 2.0 + 1.8 * n_ahead
            else:
                if v < 1.0:
                    elapsed = g["elapsed_s"] if g and g.get("elapsed_s") is not None else 0.0
                    stop_at, go = 0.0, max(0.0, delay - elapsed)
                    if state == "permissive-Movement-Allowed":      # a stopped vehicle on a permissive
                        go = max(go, 3.0)                           # movement is yielding: assume a 3 s wait
                if t_change is not None:
                    eta = max(front, 0) / max(v, 1.0)
                    if v >= 1.0 and eta > t_change + 3.0 and v * v / (2 * max(front, 0.1)) <= A_STOP_MAX:
                        stop_at, go = front, math.inf
        if s_stop is None and v < 1.0:
            stop_at, go = 0.0, 3.0
        ld = s.get("leader_bsm") or {}
        if ld.get("present") and (ld.get("speed_mps") or 0) < 2.0:
            lead_stop = ld["gap_m"] - 2.0
            if stop_at is None or lead_stop < stop_at:
                stop_at, go = max(0.0, lead_stop), max(1.5, 0.0 if go == math.inf else go)
        n = int(H_S / DT)
        S, sv, vv = [0.0], 0.0, v
        for k in range(1, n + 1):
            t = k * DT
            if stop_at is not None and t < go:
                gap = max(0.0, stop_at - sv)
                v_env = math.sqrt(2 * A_COMF * gap) if gap > 0.3 else 0.0
                vv = max(0.0, max(vv - A_STOP_MAX * DT, min(vmax, vv + A_ACC * DT, v_env)))
            else:
                vv = min(vmax, vv + A_ACC * DT)
            sv += vv * DT
            S.append(sv)
        S = np.asarray(S)
        P = np.asarray(r["path"], dtype=float)
        sp = r["path_spacing_m"]
        u = S / sp
        i = np.clip(np.floor(u).astype(int), 0, len(P) - 2)
        f = (u - i)[:, None]
        seg = P[i + 1] - P[i]
        xy = P[i] + f * seg
        # start from the vehicle, blending away the lateral offset to the path over 2 s
        off = np.array([x, y]) - P[0]
        w = np.clip(1 - DT * np.arange(len(S)) / 2.0, 0, 1)[:, None]
        xy = xy + w * off
        yaw_deg = (90 - np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))) % 360
        tin, tout = math.inf, math.inf
        if r["s_cw_entry_m"] is not None:
            e_in = np.flatnonzero(S + L / 2 >= r["s_cw_entry_m"])
            e_out = np.flatnonzero(S - L / 2 >= r["s_cw_exit_m"])
            if len(e_in):
                tin = e_in[0] * DT
                tout = e_out[0] * DT if len(e_out) else H_S + 5.0
        return xy[:, 0], xy[:, 1], tin, tout, r["maneuver"]

    # ------------------------------------------------------------ conflict rule
    @staticmethod
    def ped_windows(s):
        zones = s["ped_presence"]["zones"]
        cnt = np.asarray(s["ped_presence"]["count"])
        val = np.asarray(s["ped_presence"]["valid"]) > 0
        if not val.any():
            return []
        k = np.flatnonzero(val)[-1]
        now = cnt[k]
        Lcw = s["crosswalk_length_m"]
        nseg = sum(z.startswith("cw_") for z in zones)
        seg_idx = [i for i, z in enumerate(zones) if z.startswith("cw_")]
        win = []
        on = [i for i in seg_idx if now[i] > 0]
        if on:
            # direction from the change of the occupied segments' centroid over the last second
            kb = np.flatnonzero(val[:max(1, k - 9)])
            prev = cnt[kb[-1]] if len(kb) else now
            c_now = np.average([seg_idx.index(i) for i in on], weights=[now[i] for i in on])
            on_prev = [i for i in seg_idx if prev[i] > 0]
            c_prev = np.average([seg_idx.index(i) for i in on_prev], weights=[prev[i] for i in on_prev]) if on_prev else c_now
            seg_len = Lcw / nseg
            to_out = (nseg - seg_idx.index(min(on))) * seg_len
            to_in = (seg_idx.index(max(on)) + 1) * seg_len
            rem = to_out if c_now > c_prev + 0.1 else to_in if c_now < c_prev - 0.1 else max(to_out, to_in)
            win.append((0.0, rem / V_PED))
        waiting = now[zones.index("wait_in")] + now[zones.index("wait_out")] > 0
        sig = {g["signal_group"]: g for g in (s["signals"] or {}).get("groups", [])}
        ped = sig.get(s["target_signal_group"])
        if waiting and ped is not None:
            if ped["state"] == "protected-Movement-Allowed":
                win.append((1.0, 1.0 + Lcw / V_PED))
            elif ped["state"] == "stop-And-Remain":
                tw = ped["likely_end_s"] if ped["likely_end_s"] is not None else ped["min_end_s"]
                if tw is not None and tw <= H_S:
                    win.append((tw + 1.0, tw + 1.0 + Lcw / V_PED))
        return win

    @staticmethod
    def rule_score(tin, tout, windows, theta=1.0):
        if not math.isfinite(tin) or not windows:
            return 0.0
        gap = min(max(0.0, max(a - tout, tin - b)) for a, b in windows)
        return 1.0 if gap <= theta else math.exp(-(gap - theta) / 1.0)

    def predict(self, s, mode):
        xs, ys, tin, tout, man = self.cv(s) if mode == "cv" else self.rule(s)
        score = self.rule_score(tin, tout, self.ped_windows(s))
        return {"traj": np.stack([xs[2::2][:40], ys[2::2][:40]], -1)[None], "prob": np.ones(1),
                "tin": tin, "tout": tout, "score": score, "maneuver": man}
