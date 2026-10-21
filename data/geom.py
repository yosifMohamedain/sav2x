"""Map utilities shared by the sample builder, baselines and the risk module.

All geometry is in the intersection's local ENU frame (meters, origin at the
MAP refPoint, x east, y north). Headings follow J2735: degrees clockwise from
north. Use enu_yaw() to convert to the math convention (radians CCW from east).
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


def heading_from_yaw(yaw):
    return (90.0 - np.degrees(yaw)) % 360.0


class IntersectionMap:
    """Parsed MAP JSON (schemas/extracted/map.schema.json) with lookups."""

    def __init__(self, path):
        self.raw = json.load(open(path))
        self.version = f"{self.raw['intersection_id']}:{self.raw['revision']}"
        self.lanes = {l["lane_id"]: l for l in self.raw["lanes"]}
        self.connectors = {c["connection_id"]: c for c in self.raw["connectors"]}
        self.sgs = {g["signal_group"]: g for g in self.raw["signal_groups"]}
        self.crosswalks = {c["crosswalk_id"]: c for c in self.raw["crosswalks"]}
        tc = self.raw["target_crosswalk"]
        self.target_cw = tc["crosswalk_id"]
        self.target_sg = tc["signal_group"]
        self.target_poly = Polygon(self.crosswalks[self.target_cw]["polygon"])
        self.zone_polys = {z["zone"]: Polygon(z["polygon"]) for z in tc["zones"]}
        self.conflict_movements = tc["conflicting_movements"]
        # lines for map matching
        self._match = []
        for l in self.raw["lanes"]:
            if l["kind"] == "vehicle":
                self._match.append((l["lane_id"], l["role"], l["approach"], LineString(l["centerline"])))
        for c in self.raw["connectors"]:
            if len(c["centerline"]) >= 2:
                self._match.append((c["connection_id"], "connector", c["approach"], LineString(c["centerline"])))
        self._lines = [m[3] for m in self._match]
        cen = self.target_poly.centroid
        self.target_center = (cen.x, cen.y)

    def movement_sg(self, approach, maneuver):
        """Signal group controlling (approach, maneuver), or None."""
        for c in self.raw["connectors"]:
            if c["approach"] == approach and c["maneuver"] == maneuver:
                return c["signal_group"]
        return None

    def crosses_target(self, approach, maneuver):
        return any(m["approach"] == approach and m["maneuver"] == maneuver for m in self.conflict_movements)

    def match(self, x, y, heading_deg, speed):
        """Map-match a vehicle state to the best lane or connector.

        Returns dict(lane_id, role, approach, s_m, lat_m, dist_to_stopline_m) or None.
        Uses distance plus a heading term when the heading is reliable."""
        p = Point(x, y)
        yaw = float(enu_yaw(heading_deg))
        best, best_cost = None, 1e9
        for (lid, role, appr, line) in self._match:
            d = line.distance(p)
            if d > 6.0:
                continue
            s = line.project(p)
            a = line.interpolate(max(0.0, s - 1.0))
            b = line.interpolate(min(line.length, s + 1.0))
            lyaw = math.atan2(b.y - a.y, b.x - a.x)
            dh = 1 - math.cos(yaw - lyaw)
            cost = d + (4.0 * dh if speed > 1.0 else 0.5 * dh)
            if cost < best_cost:
                best_cost, best = cost, (lid, role, appr, line, s, d)
        if best is None:
            return None
        lid, role, appr, line, s, d = best
        out = {"lane_id": lid, "role": role, "approach": appr, "s_m": round(s, 2), "lat_m": round(d, 2)}
        if role == "ingress":
            out["dist_to_stopline_m"] = round(line.length - s, 2)
        return out


def footprints(x, y, heading_deg, length, width):
    """Vectorized oriented-rectangle polygons for vehicle centers."""
    yaw = enu_yaw(heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    corners = []
    for dl, dw in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
        corners.append(np.stack([x + dl * c - dw * s, y + dl * s + dw * c], axis=-1))
    ring = np.stack(corners, axis=1)  # (N, 4, 2)
    return shapely.polygons(ring)


def occupancy_intervals(t, inside, max_gap=0.35):
    """Merge a boolean inside-track into [t_in, t_out] intervals (seconds)."""
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
