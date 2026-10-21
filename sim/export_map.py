"""Export the intersection MAP (schemas/extracted/map.schema.json) from a SUMO network.

Besides lanes, connections and signal groups (what a J2735 MAP carries), it adds
what the vehicle-side predictor needs about the target crosswalk:
  * zones      : the crosswalk split into its inbound half (the lanes entering the
                 intersection) and outbound half, used for conflict labels;
  * presence   : zones for detecting pedestrians from SDSM, ordered from the
                 inbound curb to the outbound curb: wait_in, cw_1..cw_4, wait_out;
  * conflicting_movements : every (approach, maneuver) whose path crosses each half.

Signal groups are numbered the same way for every geometry:
vehicle SG = 1 + 3*leg_index + maneuver_index (legs N,E,S,W; maneuvers left, straight,
right), pedestrian SG = 21 + leg_index of the leg the crosswalk crosses.

Usage:
    python sim/export_map.py --run data_out/runs/base_s1 [--target-leg N]
"""
import argparse
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import LineString, Point, Polygon

LEGS = "NESW"
MANS = ["left", "straight", "right"]
DIR2MAN = {"s": "straight", "l": "left", "L": "left", "r": "right", "R": "right", "t": "uturn"}
N_SEG = 4


def parse_shape(s):
    return [tuple(map(float, p.split(","))) for p in s.split()]


def leg_of(dx, dy):
    ang = math.degrees(math.atan2(dx, dy)) % 360
    return LEGS[int(((ang + 45) % 360) // 90)]


def strip(p0, p1, half_w):
    d = np.subtract(p1, p0)
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    a, b = np.asarray(p0), np.asarray(p1)
    return [list(map(float, a + half_w * n)), list(map(float, b + half_w * n)),
            list(map(float, b - half_w * n)), list(map(float, a - half_w * n))]


def r2(pts):
    return [[round(float(x), 2), round(float(y), 2)] for x, y in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory (reads x.net.xml and tls.add.xml, writes map.json)")
    ap.add_argument("--net")
    ap.add_argument("--tls")
    ap.add_argument("--out")
    ap.add_argument("--junction", default="C")
    ap.add_argument("--target-leg", default="N", help="leg whose crosswalk is the target")
    ap.add_argument("--intersection-id", type=int, default=7)
    ap.add_argument("--ref-lat", type=float, default=35.0400, help="placeholder; use the MAP refPoint")
    ap.add_argument("--ref-lon", type=float, default=-85.3000)
    a = ap.parse_args()
    net = a.net or os.path.join(a.run, "x.net.xml")
    tlsf = a.tls or os.path.join(a.run, "tls.add.xml")
    out = a.out or os.path.join(a.run, "map.json")

    root = ET.parse(net).getroot()
    junc = [j for j in root.findall("junction") if j.get("id") == a.junction][0]
    cx, cy = float(junc.get("x")), float(junc.get("y"))

    def loc(pts):
        return [[round(x - cx, 2), round(y - cy, 2)] for x, y in pts]

    edges = {e.get("id"): e for e in root.findall("edge")}
    lane_xml = {l.get("id"): l for e in root.findall("edge") for l in e.findall("lane")}
    tl = [t for t in ET.parse(tlsf).getroot().findall("tlLogic") if t.get("id") == a.junction][0]
    phase_states = [p.get("state") for p in tl.findall("phase")]
    phase_names = [p.get("name") for p in tl.findall("phase")]

    def column(idx):
        return "".join(s[idx] for s in phase_states)

    def edge_leg(eid):
        e = edges[eid]
        pts = parse_shape(e.findall("lane")[-1].get("shape"))
        far = pts[0] if e.get("to") == a.junction else pts[-1]
        return leg_of(far[0] - cx, far[1] - cy)

    # internal lanes can be chained (SUMO splits turns at internal junctions): follow via -> via
    via_next = {}
    for c in root.findall("connection"):
        if c.get("from", "").startswith(":") and c.get("via"):
            via_next[f"{c.get('from')}_{c.get('fromLane')}"] = c.get("via")

    def via_chain(v):
        pts = []
        while v in lane_xml:
            seg = parse_shape(lane_xml[v].get("shape"))
            pts += seg if not pts else seg[1:]
            v = via_next.get(v)
        return pts

    # ---- vehicle connections and signal groups
    conns, groups = [], {}
    for c in root.findall("connection"):
        if c.get("tl") != a.junction or c.get("linkIndex") is None or c.get("from").startswith(":"):
            continue
        frm, to = c.get("from"), c.get("to")
        if edges[to].get("function") == "crossing":
            continue
        man = DIR2MAN.get(c.get("dir"), "straight")
        leg = edge_leg(frm)
        idx = int(c.get("linkIndex"))
        sg = 1 + 3 * LEGS.index(leg) + MANS.index(man)
        g = groups.setdefault(sg, {"signal_group": sg, "kind": "vehicle", "movement_name": f"{leg}_{man}",
                                   "link_indices": []})
        g["link_indices"].append(idx)
        conns.append({"connection_id": f"{frm}_{c.get('fromLane')}>{to}_{c.get('toLane')}",
                      "ingress_lane": f"{frm}_{c.get('fromLane')}", "egress_lane": f"{to}_{c.get('toLane')}",
                      "approach": leg, "maneuver": man, "signal_group": sg,
                      "centerline": loc(via_chain(c.get("via"))) if c.get("via") in lane_xml else []})

    # ---- crosswalks
    crosswalks = []
    for e in root.findall("edge"):
        if e.get("function") != "crossing":
            continue
        l = e.find("lane")
        crossed = e.get("crossingEdges").split()
        leg = edge_leg([x for x in crossed if edges[x].get("to") == a.junction][0])
        sg = 21 + LEGS.index(leg)
        idx = [int(c.get("linkIndex")) for c in root.findall("connection")
               if c.get("to") == e.get("id") and c.get("linkIndex") is not None]
        groups[sg] = {"signal_group": sg, "kind": "pedestrian", "movement_name": f"ped_{leg}", "link_indices": idx}
        cl = parse_shape(l.get("shape"))
        w = float(l.get("width"))
        crosswalks.append({"crosswalk_id": e.get("id"), "lane_id": l.get("id"), "leg": leg, "signal_group": sg,
                           "crosses_edges": crossed, "length_m": round(float(l.get("length")), 2), "width_m": w,
                           "centerline": loc(cl), "polygon": r2(LineString(loc(cl)).buffer(w / 2, cap_style="flat").exterior.coords[:-1]),
                           "is_target": leg == a.target_leg})
    for g in groups.values():
        cols = {column(i) for i in g["link_indices"]}
        assert len(cols) <= 1, f"signal group {g['signal_group']} mixes state sequences: {cols}"
        g["state_sequence"] = cols.pop() if cols else ""

    # ---- lanes
    lanes = []
    for eid, e in edges.items():
        if e.get("function") in ("internal", "crossing", "walkingarea") or a.junction not in (e.get("from"), e.get("to")):
            continue
        ingress = e.get("to") == a.junction
        for l in e.findall("lane"):
            kind = "sidewalk" if l.get("allow") == "pedestrian" else "vehicle"
            lane = {"lane_id": l.get("id"), "edge_id": eid, "kind": kind, "role": "ingress" if ingress else "egress",
                    "approach": edge_leg(eid), "lane_width_m": float(l.get("width", 3.2)),
                    "speed_limit_mps": float(l.get("speed")), "centerline": loc(parse_shape(l.get("shape")))}
            if ingress and kind == "vehicle":
                outs = [c for c in conns if c["ingress_lane"] == l.get("id")]
                lane["maneuvers"] = sorted({c["maneuver"] for c in outs}, key=MANS.index)
                lane["connects_to"] = [{"connection_id": c["connection_id"], "egress_lane": c["egress_lane"],
                                        "maneuver": c["maneuver"], "signal_group": c["signal_group"]} for c in outs]
                lane["stop_line"] = lane["centerline"][-1]
            lanes.append(lane)

    # ---- target crosswalk: halves, presence zones, conflicting movements
    tgt = [c for c in crosswalks if c["is_target"]][0]
    in_edge = [x for x in tgt["crosses_edges"] if edges[x].get("to") == a.junction][0]
    in_pts = [Point(loc(parse_shape(l.get("shape")))[-1]) for l in edges[in_edge].findall("lane")
              if l.get("allow") != "pedestrian"]
    cl = np.asarray(tgt["centerline"], dtype=float)
    p_a, p_b = cl[0], cl[-1]
    d_a = min(Point(p_a).distance(p) for p in in_pts)
    d_b = min(Point(p_b).distance(p) for p in in_pts)
    p_in, p_out = (p_a, p_b) if d_a < d_b else (p_b, p_a)   # inbound curb end first
    u = (p_out - p_in) / np.linalg.norm(p_out - p_in)
    L = float(np.linalg.norm(p_out - p_in))
    hw = tgt["width_m"] / 2
    mid = p_in + u * L / 2
    zones = [{"zone": "inbound", "polygon": r2(strip(p_in, mid, hw))},
             {"zone": "outbound", "polygon": r2(strip(mid, p_out, hw))}]
    presence = [{"zone": "wait_in", "polygon": r2(strip(p_in - u * 3.5, p_in - u * 0.2, hw + 0.75))}]
    for k in range(N_SEG):
        presence.append({"zone": f"cw_{k + 1}",
                         "polygon": r2(strip(p_in + u * L * k / N_SEG, p_in + u * L * (k + 1) / N_SEG, hw))})
    presence.append({"zone": "wait_out", "polygon": r2(strip(p_out + u * 0.2, p_out + u * 3.5, hw + 0.75))})
    lane_by = {l["lane_id"]: l for l in lanes}
    movements = []
    for c in conns:
        path = lane_by[c["ingress_lane"]]["centerline"][-2:] + c["centerline"] + lane_by[c["egress_lane"]]["centerline"][:2]
        corridor = LineString(path).buffer(1.0)
        hit = [z["zone"] for z in zones if corridor.intersects(Polygon(z["polygon"]))]
        if hit:
            movements.append({"connection_id": c["connection_id"], "approach": c["approach"], "maneuver": c["maneuver"],
                              "signal_group": c["signal_group"], "zones": hit})
    out_d = {"schema": "sav2x.map/2.0", "intersection_id": a.intersection_id, "revision": 1, "source": "sumo",
             "junction_id": a.junction, "ref": {"lat": a.ref_lat, "lon": a.ref_lon},
             "lanes": lanes, "connectors": conns, "signal_groups": sorted(groups.values(), key=lambda g: g["signal_group"]),
             "phase_names": phase_names, "crosswalks": crosswalks,
             "target_crosswalk": {"crosswalk_id": tgt["crosswalk_id"], "signal_group": tgt["signal_group"],
                                  "length_m": round(L, 2), "inbound_end": r2([p_in])[0], "outbound_end": r2([p_out])[0],
                                  "zones": zones, "presence_zones": presence, "conflicting_movements": movements}}
    json.dump(out_d, open(out, "w"), indent=1)
    print(f"MAP {out}: {len(lanes)} lanes, {len(conns)} connections, {len(groups)} signal groups; "
          f"target {tgt['crosswalk_id']} ({L:.1f} m) conflicts with "
          f"{sorted({(m['approach'], m['maneuver']) for m in movements})}")


if __name__ == "__main__":
    main()
