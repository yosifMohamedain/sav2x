"""Non-learned baselines, evaluated with the same metrics as the model.

    cv       : constant velocity along the current heading (vehicles and pedestrians)
    cv_spat  : signal-aware kinematic rule. Vehicles on an ingress lane follow
               the MAP path for their lane's most likely movement (or their turn
               signal when a BSM carries one), stop at the stop line if their
               movement is red / clearance / right-on-red and the stop is
               feasible, wait for likelyTime (or minEndTime), then accelerate to
               the speed limit. Pedestrians at the curb wait for WALK.

Each returns, per agent, a single trajectory (5 Hz, world frame) and the
crosswalk occupancy interval [t_in, t_out] (inf when it never enters), which
the evaluator turns into distributions with an uncertainty that grows with
time-to-arrival.
"""
import math

import numpy as np
from shapely.geometry import LineString, Point

from data.geom import IntersectionMap, enu_yaw

DT = 0.1
H_S = 20.0
A_ACC, A_STOP_MAX, A_STOP_START = 2.0, 4.5, 1.2
RED_LIKE = {"stop-And-Remain", "permissive-clearance", "protected-clearance"}


def last_state(agent):
    h = [r for r in agent["hist"] if r[-1]]
    if not h:
        return None
    x, y, v, hd = h[-1][:4]
    return x, y, v, hd


def interval_from_track(xs, ys, yaws, ts, poly, length, width, is_vehicle):
    """First interval when the agent (rectangle or point) occupies poly."""
    import shapely
    if is_vehicle:
        c, s = np.cos(yaws), np.sin(yaws)
        hl, hw = length / 2, width / 2
        ring = np.stack([np.stack([xs + dl * c - dw * s, ys + dl * s + dw * c], -1)
                         for dl, dw in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))], 1)
        geo = shapely.polygons(ring)
    else:
        geo = shapely.points(xs, ys)
    ins = shapely.intersects(geo, poly)
    if not ins.any():
        return math.inf, math.inf
    i0 = int(np.argmax(ins))
    rest = ins[i0:]
    i1 = i0 + (int(np.argmin(rest)) if (~rest).any() else len(rest))
    return float(ts[i0]), float(ts[min(i1, len(ts) - 1)]) if i1 < len(ts) else H_S + 5.0


def cv_rollout(x, y, v, hd, n=int(H_S / DT)):
    yaw = float(enu_yaw(hd))
    t = DT * np.arange(0, n + 1)
    return x + v * np.cos(yaw) * t, y + v * np.sin(yaw) * t, np.full_like(t, yaw), t


class Baselines:
    def __init__(self, imap: IntersectionMap):
        self.m = imap
        self.poly = imap.target_poly
        self.lines = {l["lane_id"]: LineString(l["centerline"]) for l in imap.raw["lanes"] if l["kind"] == "vehicle"}
        self.conn = {}
        for c in imap.raw["connectors"]:
            self.conn.setdefault(c["ingress_lane"], {})[c["maneuver"]] = c

    def _path(self, lane_id, s0, maneuver):
        c = self.conn.get(lane_id, {}).get(maneuver)
        if c is None:
            return None, None, None
        ing = self.m.lanes[lane_id]["centerline"]
        line = LineString(ing)
        pts = [line.interpolate(s0).coords[0]]
        cum = 0.0
        for p in ing:
            if line.project(Point(p)) > s0 + 0.1:
                pts.append(tuple(p))
        s_stop = line.length - s0
        pts += [tuple(p) for p in c["centerline"]]
        eg = self.m.lanes[c["egress_lane"]]["centerline"]
        pts += [tuple(p) for p in eg[1:]]
        return LineString(pts), s_stop, c["signal_group"]

    def lane_context(self, agents):
        """(ingress lane, distance along it, speed) of every map-matched vehicle; a vehicle
        inside the box counts on the ingress lane its connector starts from."""
        ctx = []
        for a in agents:
            mm = a.get("map_match")
            st = last_state(a)
            if a["type"] != "vehicle" or not mm or not st:
                continue
            if mm["role"] == "ingress":
                ctx.append((mm["lane_id"], mm["s_m"], st[2]))
            elif mm["role"] == "connector" and mm["lane_id"] in self.m.connectors:
                c = self.m.connectors[mm["lane_id"]]
                ctx.append((c["ingress_lane"], self.lines[c["ingress_lane"]].length + mm["s_m"], st[2]))
        return ctx

    def predict(self, sample, agent, mode, signals, ctx=()):
        st = last_state(agent)
        L, W = agent.get("length_m") or 4.7, agent.get("width_m") or 1.8
        veh = agent["type"] == "vehicle"
        if st is None:
            return None
        x, y, v, hd = st
        mm = agent.get("map_match")
        if mode == "cv" or not veh or mm is None or mm["role"] != "ingress":
            if not veh and mode == "cv_spat":
                return self._ped(x, y, v, hd, signals)
            xs, ys, yaws, ts = cv_rollout(x, y, v, hd)
            tin, tout = interval_from_track(xs, ys, yaws, ts, self.poly, L, W, veh)
            return xs, ys, tin, tout
        # signal-aware path following for vehicles on an ingress lane
        lane = self.m.lanes[mm["lane_id"]]
        mans = lane.get("maneuvers", ["straight"])
        man = "straight" if "straight" in mans else mans[0]
        ext = agent.get("bsm_ext")
        if ext is not None and len(ext):
            last = [r for r, h in zip(ext, agent["hist"]) if h[-1]]
            if last and last[-1][3] and "left" in mans:
                man = "left"
            elif last and last[-1][4] and "right" in mans:
                man = "right"
        path, s_stop, sg = self._path(mm["lane_id"], mm["s_m"], man)
        if path is None:
            xs, ys, yaws, ts = cv_rollout(x, y, v, hd)
            tin, tout = interval_from_track(xs, ys, yaws, ts, self.poly, L, W, veh)
            return xs, ys, tin, tout
        g = signals.get(sg) if signals else None
        state = g["state"] if g else "protected-Movement-Allowed"
        t_change = None
        if g:
            t_change = g["likely_end_s"] if g["likely_end_s"] is not None else g["min_end_s"]
        vmax = lane.get("speed_limit_mps", 13.4)
        # queue: slow or stopped vehicles ahead in the same lane push the stop point back
        # by one jam spacing each and delay the start by one saturation headway each
        n_ahead = sum(1 for (lid, s_o, v_o) in ctx
                      if lid == mm["lane_id"] and 1.0 < s_o - mm["s_m"] < 80.0 and v_o < 2.0)
        stop_at = s_stop - 1.0 - 7.5 * n_ahead      # front bumper target
        must_stop, go_time = False, None
        if state in RED_LIKE:
            must_stop = v ** 2 / (2 * max(stop_at - L / 2, 0.1)) <= A_STOP_MAX or v < 1.0
            go_time = (t_change if state == "stop-And-Remain" else 1e9) if t_change is not None else 1e9
            go_time += 1.0 + 1.8 * n_ahead
        elif state == "stop-Then-Proceed":
            must_stop, go_time = True, 2.0
        elif t_change is not None and state != "stop-And-Remain":
            # green ending: stop if the stop line cannot be reached before green + 3 s yellow
            eta = max(stop_at - L / 2, 0) / max(v, 1.0)
            if eta > t_change + 3.0 and v ** 2 / (2 * max(stop_at - L / 2, 0.1)) <= A_STOP_MAX:
                must_stop, go_time = True, 1e9
        n = int(H_S / DT)
        s, vv = 0.0, v
        S = [0.0]
        # a queued vehicle (stopped short of the stop line on green) starts after a discharge delay
        hold = 1.0 + 1.8 * n_ahead if (v < 0.5 and not must_stop) else 0.0
        if stop_at - L / 2 < -0.5:
            must_stop = False                       # already past the stop line
        for k in range(1, n + 1):
            t = k * DT
            if must_stop and t < go_time:
                gap = max(0.0, stop_at - (s + L / 2))
                # stay under the comfortable stopping envelope, never brake harder than A_STOP_MAX
                v_env = math.sqrt(2 * A_STOP_START * gap) if gap > 0.3 else 0.0
                vv = max(vv - A_STOP_MAX * DT, min(vmax, vv + A_ACC * DT, v_env))
                vv = max(vv, 0.0)
            elif t < hold:
                vv = 0.0
            else:
                vv = min(vmax, vv + A_ACC * DT)
            s += vv * DT
            S.append(s)
        S = np.asarray(S)
        S = np.clip(S, 0, path.length - 0.01)
        pts = [path.interpolate(si) for si in S]
        xs = np.array([p.x for p in pts])
        ys = np.array([p.y for p in pts])
        ahead = [path.interpolate(min(path.length, si + 1.0)) for si in S]
        yaws = np.array([math.atan2(a.y - p.y, a.x - p.x) for a, p in zip(ahead, pts)])
        ts = DT * np.arange(0, n + 1)
        tin, tout = interval_from_track(xs, ys, yaws, ts, self.poly, L, W, True)
        return xs, ys, tin, tout

    def _ped(self, x, y, v, hd, signals):
        p = Point(x, y)
        ped = signals.get(self.m.target_sg) if signals else None
        cw = self.m.crosswalks[self.m.target_cw]
        length = cw["length_m"]
        d = self.poly.distance(p)
        xs, ys, yaws, ts = cv_rollout(x, y, v, hd)
        if self.poly.contains(p):
            tin, tout = interval_from_track(xs, ys, yaws, ts, self.poly, 0.5, 0.5, False)
            return xs, ys, tin, tout
        if d < 4.0 and ped is not None:
            walk_speed = max(v, 1.2)
            if ped["state"] == "protected-Movement-Allowed":
                tin = d / walk_speed
            elif ped["state"] == "stop-And-Remain" and ped["likely_end_s"] is not None:
                tin = max(0.0, ped["likely_end_s"]) + 1.0 + d / walk_speed
            elif ped["state"] == "stop-And-Remain":
                tin = max(0.0, ped["min_end_s"] or 0.0) + 1.0 + d / walk_speed
            else:
                tin = math.inf
            tout = tin + length / walk_speed if math.isfinite(tin) else math.inf
            return np.full_like(xs, x), np.full_like(ys, y), tin, tout
        tin, tout = interval_from_track(xs, ys, yaws, ts, self.poly, 0.5, 0.5, False)
        return xs, ys, tin, tout
