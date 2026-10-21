"""MAP utilities: map matching, candidate routes, zones and occupancy intervals.

All geometry is in the intersection's local ENU frame (meters, origin at the MAP
refPoint, x east, y north). Headings follow J2735 (degrees clockwise from north);
enu_yaw() converts to radians counter-clockwise from east.
"""
import json
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

MANEUVERS = ["left", "straight", "right"]
LEGS = ["N", "E", "S", "W"]


def enu_yaw(heading_deg):
    return np.radians(90.0 - np.asarray(heading_deg, dtype=np.float64))


def footprints(x, y, heading_deg, length, width):
    """Oriented rectangles for vehicle centers (vectorized)."""
    yaw = enu_yaw(heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    ring = np.stack([np.stack([x + dl * c - dw * s, y + dl * s + dw * c], -1)
                     for dl, dw in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))], 1)
    return shapely.polygons(ring)


def intervals(t, inside, max_gap=0.35):
    """Merge a boolean series into [t_in, t_out] intervals (t_out = last inside step + 0.1 s)."""
    out, t_in, last = [], None, None
    for ti, ins in zip(t, inside):
        if ins:
            if t_in is not None and ti - last > max_gap:
                out.append([t_in, last + 0.1])
                t_in = None
            if t_in is None:
                t_in = ti
            last = ti
        elif t_in is not None:
            out.append([t_in, last + 0.1])
            t_in = None
    if t_in is not None:
        out.append([t_in, last + 0.1])
    return out


class IntersectionMap:
    def __init__(self, path):
        self.raw = json.load(open(path))
        self.version = f"{self.raw['intersection_id']}:{self.raw['revision']}"
        self.lanes = {l["lane_id"]: l for l in self.raw["lanes"]}
        self.conns = {c["connection_id"]: c for c in self.raw["connectors"]}
        self.sgs = {g["signal_group"]: g for g in self.raw["signal_groups"]}
        tc = self.raw["target_crosswalk"]
        self.target = tc
        self.target_cw = tc["crosswalk_id"]
        self.target_sg = tc["signal_group"]
        self.cw_poly = Polygon([c for c in self.raw["crosswalks"] if c["is_target"]][0]["polygon"])
        self.zone_polys = {z["zone"]: Polygon(z["polygon"]) for z in tc["zones"]}
        self.presence_names = [z["zone"] for z in tc["presence_zones"]]
        self.presence_polys = [Polygon(z["polygon"]) for z in tc["presence_zones"]]
        self.crossing = {}
        for m in tc["conflicting_movements"]:
            self.crossing[m["connection_id"]] = m["zones"]
        # full route line per connection: ingress lane + connector + egress lane
        self.route_line, self.route_meta = {}, {}
        for c in self.raw["connectors"]:
            ing = self.lanes[c["ingress_lane"]]["centerline"]
            eg = self.lanes[c["egress_lane"]]["centerline"]
            pts = ing + c["centerline"] + eg
            clean = [pts[0]]
            for p in pts[1:]:
                if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 0.05:
                    clean.append(p)
            line = LineString(clean)
            s_stop = LineString(ing).length
            s_in = s_out = None
            if c["connection_id"] in self.crossing:
                inter = line.intersection(self.cw_poly)
                if not inter.is_empty:
                    coords = [p for g in getattr(inter, "geoms", [inter]) for p in g.coords]
                    ss = [line.project(Point(p)) for p in coords]
                    s_in, s_out = min(ss), max(ss)
            self.route_line[c["connection_id"]] = line
            self.route_meta[c["connection_id"]] = {"s_stop": s_stop, "s_cw_in": s_in, "s_cw_out": s_out,
                                                   "len_ingress": s_stop,
                                                   "len_box": s_stop + LineString(c["centerline"]).length
                                                   if len(c["centerline"]) > 1 else s_stop}
        # map-matching candidates: ingress vehicle lanes and connectors
        self._mm = []
        for l in self.raw["lanes"]:
            if l["kind"] == "vehicle":
                self._mm.append((l["lane_id"], l["role"], l["approach"], LineString(l["centerline"])))
        for c in self.raw["connectors"]:
            if len(c["centerline"]) >= 2:
                self._mm.append((c["connection_id"], "connector", c["approach"], LineString(c["centerline"])))
        self._mm_lines = np.array([m[3] for m in self._mm], dtype=object)
        # per approach: maneuver -> serving ingress lanes
        self.serving = {}
        for l in self.raw["lanes"]:
            if l["kind"] == "vehicle" and l["role"] == "ingress":
                for ct in l.get("connects_to", []):
                    self.serving.setdefault((l["approach"], ct["maneuver"]), []).append(ct["connection_id"])

    def movement_sg(self, approach, maneuver):
        for c in self.raw["connectors"]:
            if c["approach"] == approach and c["maneuver"] == maneuver:
                return c["signal_group"]
        return None

    def crosses(self, approach, maneuver):
        return any(m["approach"] == approach and m["maneuver"] == maneuver
                   for m in self.target["conflicting_movements"])

    def match(self, x, y, heading_deg, speed, approach=None):
        """Best lane (ingress or egress) or connector for a vehicle state, or None.
        approach, if given, restricts ingress lanes and connectors to that approach (the one the
        vehicle's own history says it came from); egress lanes stay candidates."""
        p = Point(x, y)
        d = shapely.distance(self._mm_lines, p)
        cand = np.flatnonzero(d < 4.5)
        if approach is not None:
            cand = [i for i in cand if self._mm[i][1] == "egress" or self._mm[i][2] == approach]
        if len(cand) == 0:
            return None
        yaw = float(enu_yaw(heading_deg))
        best, best_cost = None, 1e9
        for i in cand:
            lid, role, appr, line = self._mm[i]
            s = line.project(p)
            a = line.interpolate(max(0.0, s - 1.0))
            b = line.interpolate(min(line.length, s + 1.0))
            dh = 1 - math.cos(yaw - math.atan2(b.y - a.y, b.x - a.x))
            cost = d[i] + (6.0 * dh if speed > 1.0 else 1.0 * dh)
            # a point beyond the end of an ingress lane belongs to the box
            if role == "ingress" and s >= line.length - 0.05 and d[i] > 0.5:
                cost += 2.0
            if role == "egress" and s <= 0.05 and d[i] > 0.5:   # and a point before an egress lane, too
                cost += 2.0
            if cost < best_cost:
                best_cost, best = cost, (lid, role, appr, line, s, d[i])
        if best is None:
            return None
        lid, role, appr, line, s, dist = best
        return {"lane_id": lid, "role": role, "approach": appr, "s_m": round(float(s), 2),
                "lat_m": round(float(dist), 2),
                "dist_to_stopline_m": round(float(line.length - s), 2) if role == "ingress" else None}

    def candidate_routes(self, mm, x=None, y=None, heading_deg=None):
        """Connection ids to consider: one per maneuver on the approach (preferring the current lane).
        Inside the box, connectors overlap, so every connector of the approach that passes within 2.5 m
        of the vehicle with a consistent heading stays a candidate (one per maneuver)."""
        if mm["role"] == "connector":
            if x is None:
                return [(mm["lane_id"], False)]
            p = Point(x, y)
            yaw = float(enu_yaw(heading_deg))
            best = {}
            for c in self.raw["connectors"]:
                if c["approach"] != mm["approach"] or len(c["centerline"]) < 2:
                    continue
                line = LineString(c["centerline"])
                d = line.distance(p)
                if d > 2.5:
                    continue
                s = line.project(p)
                a = line.interpolate(max(0.0, s - 1.0))
                b = line.interpolate(min(line.length, s + 1.0))
                dh = 1 - math.cos(yaw - math.atan2(b.y - a.y, b.x - a.x))
                if dh > 0.5:
                    continue
                cost = d + 3.0 * dh
                m = c["maneuver"]
                if m not in best or cost < best[m][0]:
                    best[m] = (cost, c["connection_id"])
            if mm["lane_id"] in self.conns:
                m0 = self.conns[mm["lane_id"]]["maneuver"]
                best.setdefault(m0, (0.0, mm["lane_id"]))
            return [(best[m][1], False) for m in MANEUVERS if m in best]
        out = []
        for man in MANEUVERS:
            cands = self.serving.get((mm["approach"], man), [])
            if not cands:
                continue
            same = [c for c in cands if self.conns[c]["ingress_lane"] == mm["lane_id"]]
            if same:
                out.append((same[0], False))
            else:
                cur = self.lanes[mm["lane_id"]]["centerline"][-1]

                def dist(cid):
                    e = self.lanes[self.conns[cid]["ingress_lane"]]["centerline"][-1]
                    return math.hypot(e[0] - cur[0], e[1] - cur[1])
                out.append((min(cands, key=dist), True))
        return out

    def presence_counts(self, xs, ys):
        """Counts of points per presence zone (vectorized)."""
        if len(xs) == 0:
            return np.zeros(len(self.presence_polys), dtype=np.int32)
        return np.array([int(shapely.contains_xy(p, xs, ys).sum()) for p in self.presence_polys], dtype=np.int32)
