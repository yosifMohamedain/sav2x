"""Build the SUMO inputs for one scenario run: network (per geometry), signal program
(generated from the network's own connections, so any lane count works), and demand.

Usage:
    python sim/scenario.py --catalog configs/scenarios.json --scenario base --seed 1 --out data_out/runs/base_s1
writes   <out>/x.net.xml  <out>/tls.add.xml  <out>/demand.rou.xml  <out>/scenario.json

Signal plan (two actuated vehicle phases; each followed by a fixed pedestrian
clearance interval sized by the MUTCD rule crossing_length / 1.07 m/s, then yellow
and all-red):
    A_walk  : N and S approaches move (lefts and rights permissive); crossings over
              the E and W legs show WALK; E/W rights may turn on red.
    A_clear : same vehicle states, all crossings DON'T WALK (flashing in practice)
    A_yellow, A_allred, then the same four for B (E and W approaches).
"""
import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    try:
        import sumo  # pip install eclipse-sumo
        SUMO_HOME = sumo.SUMO_HOME
    except ImportError:
        sys.exit("Set SUMO_HOME or pip install eclipse-sumo")
NETCONVERT = os.path.join(SUMO_HOME, "bin", "netconvert")
LEGS = "NESW"
DEST = {"N": {"right": "W", "straight": "S", "left": "E"},
        "E": {"right": "N", "straight": "W", "left": "S"},
        "S": {"right": "E", "straight": "N", "left": "W"},
        "W": {"right": "S", "straight": "E", "left": "N"}}


def deep_update(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_scenario(catalog_path, name):
    cat = json.load(open(catalog_path))
    if name not in cat["scenarios"]:
        sys.exit(f"unknown scenario {name}; have {sorted(cat['scenarios'])}")
    p = deep_update(cat["defaults"], cat["scenarios"][name])
    p["name"] = name
    p["geometry_params"] = cat["geometries"][p["geometry"]]
    return p


# ------------------------------------------------------------------ network
def build_network(geom, out_dir):
    L, n, v = geom["leg_m"], geom["lanes_per_direction"], geom["speed_mps"]
    nod = os.path.join(out_dir, "x.nod.xml")
    edg = os.path.join(out_dir, "x.edg.xml")
    with open(nod, "w") as f:
        f.write(f"""<nodes>
  <node id="C" x="0" y="0" type="traffic_light_right_on_red"/>
  <node id="N" x="0" y="{L}" type="priority"/>
  <node id="S" x="0" y="{-L}" type="priority"/>
  <node id="E" x="{L}" y="0" type="priority"/>
  <node id="W" x="{-L}" y="0" type="priority"/>
</nodes>
""")
    with open(edg, "w") as f:
        f.write("<edges>\n")
        for leg in LEGS:
            f.write(f'  <edge id="{leg}_in" from="{leg}" to="C" numLanes="{n}" speed="{v}"/>\n')
            f.write(f'  <edge id="{leg}_out" from="C" to="{leg}" numLanes="{n}" speed="{v}"/>\n')
        f.write("</edges>\n")
    net = os.path.join(out_dir, "x.net.xml")
    subprocess.run([NETCONVERT, "--node-files", nod, "--edge-files", edg,
                    "--sidewalks.guess", "true", "--crossings.guess", "true",
                    "--tls.default-type", "actuated", "--no-turnarounds", "true",
                    "--junctions.corner-detail", "5", "--offset.disable-normalization", "true",
                    "--output-file", net], check=True, stdout=subprocess.DEVNULL)
    return net


def tls_links(net_path, tls_id="C"):
    """Describe every controlled link: ('veh', approach, dir) or ('ped', crossed_leg)."""
    root = ET.parse(net_path).getroot()
    edges = {e.get("id"): e for e in root.findall("edge")}
    links = {}
    for c in root.findall("connection"):
        if c.get("tl") != tls_id or c.get("linkIndex") is None:
            continue
        idx = int(c.get("linkIndex"))
        frm, to = c.get("from"), c.get("to")
        if edges[to].get("function") == "crossing":
            crossed = edges[to].get("crossingEdges").split()
            leg = [x for x in crossed if x.endswith("_in")][0][0]
            links[idx] = ("ped", leg, None, edges[to].get("id"))
        elif not frm.startswith(":"):
            d = c.get("dir").lower()
            d = {"s": "straight", "l": "left", "r": "right", "t": "uturn"}.get(d, "straight")
            links[idx] = ("veh", frm[0], d, None)
    n = max(links) + 1
    assert sorted(links) == list(range(n)), "link indices are not contiguous"
    return [links[i] for i in range(n)]


def crossing_length(net_path, crossed_leg):
    root = ET.parse(net_path).getroot()
    for e in root.findall("edge"):
        if e.get("function") == "crossing" and f"{crossed_leg}_in" in e.get("crossingEdges", "").split():
            return float(e.find("lane").get("length"))
    raise ValueError(crossed_leg)


def write_tls(net_path, p, out_path, tls_id="C"):
    links = tls_links(net_path, tls_id)
    groups = {"A": ("N", "S"), "B": ("E", "W")}
    peds_with = {"A": ("E", "W"), "B": ("N", "S")}   # crossings parallel to the moving traffic
    # clearance sized for the longest crossing
    L = max(crossing_length(net_path, leg) for leg in LEGS)
    clear = int(math.ceil(L / p["ped_walk_speed_design_mps"]))

    def state(phase, grp):
        s = []
        for kind, a, d, _ in links:
            if kind == "ped":
                s.append("G" if (phase == "walk" and a in peds_with[grp]) else "r")
                continue
            moving = a in groups[grp]
            if phase == "allred":
                s.append("r")
            elif moving and phase in ("walk", "clear"):
                s.append("G" if d == "straight" else "g")
            elif moving and phase == "yellow":
                s.append("y")
            elif not moving and d == "right" and phase != "allred":
                s.append("s")                       # right turn on red
            else:
                s.append("r")
        return "".join(s)

    phases = []
    for grp in ("A", "B"):
        phases.append((state("walk", grp), 20, p["min_green_s"], p["max_green_s"], f"{grp}_walk"))
        phases.append((state("clear", grp), clear, None, None, f"{grp}_ped_clear"))
        phases.append((state("yellow", grp), 3, None, None, f"{grp}_yellow"))
        phases.append((state("allred", grp), 1, None, None, f"{grp}_allred"))
    lines = ['<additional>', f'  <tlLogic id="{tls_id}" type="actuated" programID="sav2x" offset="0">',
             '    <param key="max-gap" value="3.0"/>', '    <param key="detector-gap" value="25"/>']
    for st, dur, mn, mx, name in phases:
        extra = f' minDur="{mn}" maxDur="{mx}"' if mn is not None else ""
        lines.append(f'    <phase duration="{dur}" state="{st}"{extra} name="{name}"/>')
    lines += ['  </tlLogic>', '</additional>']
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"ped_clearance_s": clear, "crossing_length_m": L, "phases": [ph[4] for ph in phases]}


# ------------------------------------------------------------------ demand
def write_demand(p, seed, out_path, edge_len):
    rnd = random.Random(seed)
    sf = p["speed_factor"]
    aggr, truck, cv = p["aggr_share"], p["truck_share"], p["cv_share"]
    car = max(0.0, 1.0 - aggr - truck)
    spd = f'normc({sf["mean"]},{sf["sd"]},{sf["min"]},{sf["max"]})'
    ps = p["ped_speed"]
    # SUMO persons walk at min(maxSpeed, speedFactor * maxSpeed); speedFactor carries the spread
    ped_max = ps["max"]
    ped_sf = f'normc({ps["mean"] / ped_max:.3f},{ps["sd"] / ped_max:.3f},{ps["min"] / ped_max:.3f},1.0)'
    lines = ["<routes>",
             f'  <vType id="car" vClass="passenger" length="4.7" width="1.8" accel="2.6" decel="4.5" sigma="0.5" speedFactor="{spd}" jmCrossingGap="8" impatience="0.2"/>',
             f'  <vType id="aggr" vClass="passenger" length="4.7" width="1.8" accel="3.0" decel="5.0" sigma="0.7" speedFactor="{spd}" jmCrossingGap="3" impatience="0.8" jmIgnoreFoeProb="0.3" jmIgnoreFoeSpeed="3" jmDriveAfterRedTime="2" jmDriveAfterYellowTime="2"/>',
             f'  <vType id="truck" vClass="truck" length="10.0" width="2.5" accel="1.3" decel="4.0" sigma="0.5" speedFactor="{spd}" jmCrossingGap="8"/>',
             f'  <vTypeDistribution id="mix" vTypes="car aggr truck" probabilities="{car:.3f} {aggr:.3f} {truck:.3f}"/>',
             f'  <vType id="ped" vClass="pedestrian" maxSpeed="{ped_max}" speedFactor="{ped_sf}"/>']
    T0, T1 = 0, p["warmup_s"] + p["logged_s"]
    fid = 0
    for o in LEGS:
        vph = p["veh_per_hour"] * rnd.uniform(1 - p["approach_jitter"], 1 + p["approach_jitter"])
        for mv, share in p["turn_shares"].items():
            rate = vph * share / 3600.0
            if rate <= 0:
                continue
            lines.append(f'  <flow id="f{fid}_{o}_{mv}" type="mix" from="{o}_in" to="{DEST[o][mv]}_out" '
                         f'begin="{T0}" end="{T1}" period="exp({rate:.5f})" departLane="best" departSpeed="max"/>')
            fid += 1
    pid = 0
    for leg in LEGS:
        rate = p["ped_per_hour"] * (p["target_ped_factor"] if leg == "N" else 0.6) / 3600.0 / 2
        for frm, to in ((f"{leg}_in", f"{leg}_out"), (f"{leg}_out", f"{leg}_in")):
            dpos = edge_len - 25 if frm.endswith("_in") else 25
            apos = 25 if to.endswith("_out") else edge_len - 25
            lines.append(f'  <personFlow id="p{pid}_{leg}" type="ped" begin="{T0}" end="{T1}" period="exp({rate:.5f})" '
                         f'departPos="{dpos:.1f}">\n    <walk from="{frm}" to="{to}" arrivalPos="{apos:.1f}"/>\n  </personFlow>')
            pid += 1
    lines.append("</routes>")
    # the CV flag is assigned per vehicle id at emulation time with the same seed (see emulate_v2x.py)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def edge_length(net_path, eid="N_in"):
    root = ET.parse(net_path).getroot()
    for e in root.findall("edge"):
        if e.get("id") == eid:
            return float(e.find("lane").get("length"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="configs/scenarios.json")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--net-cache", default=None, help="directory to cache one network per geometry")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override scalar scenario parameters, e.g. --set logged_s=200 ped_per_hour=300")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    p = load_scenario(a.catalog, a.scenario)
    for kv in a.set:
        k, v = kv.split("=", 1)
        p[k] = json.loads(v)
    p["seed"] = a.seed
    cache = a.net_cache or os.path.join(os.path.dirname(os.path.normpath(a.out)), "_nets")
    gdir = os.path.join(cache, p["geometry"])
    if not os.path.exists(os.path.join(gdir, "x.net.xml")):
        os.makedirs(gdir, exist_ok=True)
        build_network(p["geometry_params"], gdir)
    net = os.path.join(a.out, "x.net.xml")
    shutil.copy(os.path.join(gdir, "x.net.xml"), net)
    info = write_tls(net, p, os.path.join(a.out, "tls.add.xml"))
    write_demand(p, a.seed, os.path.join(a.out, "demand.rou.xml"), edge_length(net))
    p["tls"] = info
    json.dump(p, open(os.path.join(a.out, "scenario.json"), "w"), indent=1)
    print(f"{a.scenario} seed {a.seed}: clearance {info['ped_clearance_s']} s for {info['crossing_length_m']:.1f} m -> {a.out}")


if __name__ == "__main__":
    main()
