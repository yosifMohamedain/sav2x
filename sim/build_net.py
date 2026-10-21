"""Build a 4-leg signalized test intersection in SUMO with crosswalks and pedestrians.

This is a stand-in for one MLK corridor intersection so the whole pipeline
(emulated SDSM/BSM/SPaT -> samples -> model -> alerts) runs end to end.
To use the real twin instead, skip this script and point run_emulator.py at
your MLK .net.xml/.rou.xml and set the target intersection and crosswalk ids.

Signal plan (two actuated vehicle phases, each followed by a fixed pedestrian
clearance interval that stands in for flashing DON'T WALK, yellow, all-red):
  0 NS walk (actuated 7..40 s)  1 NS ped clearance 10 s  2 NS yellow 3 s  3 all-red 1 s
  4 EW walk (actuated 7..40 s)  5 EW ped clearance 10 s  6 EW yellow 3 s  7 all-red 1 s
Lefts are permissive (g), rights are permissive on green and right-on-red (s)
otherwise, so turning vehicles share the crosswalks with pedestrians.

Usage:
    python sim/build_net.py --out sim_out/net
    python sim/build_net.py --out sim_out/net --demand-only --seed 3 --veh-per-hour 450
"""
import argparse
import os
import random
import subprocess
import sys

SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    try:
        import sumo  # pip install eclipse-sumo
        SUMO_HOME = sumo.SUMO_HOME
    except ImportError:
        sys.exit("Set SUMO_HOME or pip install eclipse-sumo")
NETCONVERT = os.path.join(SUMO_HOME, "bin", "netconvert")

LEG = 300.0          # leg length in m
SPEED = 13.41        # 30 mph posted speed
LEGS = "NESW"

# link index layout produced by netconvert for this node (checked in README):
# 0-3 N_in (right, thru, thru, left), 4-7 E_in, 8-11 S_in, 12-15 W_in,
# 16 crossing c0 (north), 17 c1 (east), 18 c2 (south), 19 c3 (west)
PHASES = [
    # state string                      dur  min  max   name
    ("gGGgsrrrgGGgsrrr" + "rGrG", 20, 7, 40, "NS_walk"),
    ("gGGgsrrrgGGgsrrr" + "rrrr", 10, None, None, "NS_ped_clear"),
    ("yyyysrrryyyysrrr" + "rrrr", 3, None, None, "NS_yellow"),
    ("rrrrrrrrrrrrrrrr" + "rrrr", 1, None, None, "allred1"),
    ("srrrgGGgsrrrgGGg" + "GrGr", 20, 7, 40, "EW_walk"),
    ("srrrgGGgsrrrgGGg" + "rrrr", 10, None, None, "EW_ped_clear"),
    ("srrryyyysrrryyyy" + "rrrr", 3, None, None, "EW_yellow"),
    ("rrrrrrrrrrrrrrrr" + "rrrr", 1, None, None, "allred2"),
]


def write_plain(out):
    nod = f"""<nodes>
  <node id="C" x="0" y="0" type="traffic_light_right_on_red"/>
  <node id="N" x="0" y="{LEG}" type="priority"/>
  <node id="S" x="0" y="{-LEG}" type="priority"/>
  <node id="E" x="{LEG}" y="0" type="priority"/>
  <node id="W" x="{-LEG}" y="0" type="priority"/>
</nodes>
"""
    edges = []
    for leg in LEGS:
        edges.append(f'  <edge id="{leg}_in" from="{leg}" to="C" numLanes="2" speed="{SPEED}"/>')
        edges.append(f'  <edge id="{leg}_out" from="C" to="{leg}" numLanes="2" speed="{SPEED}"/>')
    edg = "<edges>\n" + "\n".join(edges) + "\n</edges>\n"
    with open(os.path.join(out, "x.nod.xml"), "w") as f:
        f.write(nod)
    with open(os.path.join(out, "x.edg.xml"), "w") as f:
        f.write(edg)


def write_tls(out):
    lines = ['<additional>',
             '  <tlLogic id="C" type="actuated" programID="mlk" offset="0">',
             '    <param key="max-gap" value="3.0"/>',
             '    <param key="detector-gap" value="25"/>',
             '    <param key="show-detectors" value="false"/>']
    for state, dur, mn, mx, name in PHASES:
        extra = f' minDur="{mn}" maxDur="{mx}"' if mn is not None else ""
        lines.append(f'    <phase duration="{dur}" state="{state}"{extra} name="{name}"/>')
    lines += ['  </tlLogic>', '</additional>']
    with open(os.path.join(out, "tls.add.xml"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_netconvert(out):
    cmd = [NETCONVERT,
           "--node-files", os.path.join(out, "x.nod.xml"),
           "--edge-files", os.path.join(out, "x.edg.xml"),
           "--sidewalks.guess", "true",
           "--crossings.guess", "true",
           "--tls.default-type", "actuated",
           "--no-turnarounds", "true",
           "--junctions.corner-detail", "5",
           "--offset.disable-normalization", "true",
           "--output-file", os.path.join(out, "x.net.xml")]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


VTYPES = """  <vType id="car" vClass="passenger" length="4.7" width="1.8" accel="2.6" decel="4.5" sigma="0.5" speedFactor="normc(1,0.1,0.75,1.3)" jmCrossingGap="8" impatience="0.2"/>
  <vType id="cv" vClass="passenger" length="4.7" width="1.8" accel="2.6" decel="4.5" sigma="0.5" speedFactor="normc(1,0.1,0.75,1.3)" jmCrossingGap="8" impatience="0.2" color="0,1,1"/>
  <vType id="aggr" vClass="passenger" length="4.7" width="1.8" accel="3.0" decel="5.0" sigma="0.7" speedFactor="normc(1.15,0.1,0.9,1.4)" jmCrossingGap="3" impatience="0.8" jmIgnoreFoeProb="0.3" jmIgnoreFoeSpeed="3" jmDriveAfterRedTime="2" jmDriveAfterYellowTime="2" color="1,0,0"/>
  <vType id="truck" vClass="truck" length="10.0" width="2.5" accel="1.3" decel="4.0" sigma="0.5" speedFactor="normc(0.95,0.08,0.75,1.2)" jmCrossingGap="8"/>
  <vTypeDistribution id="mix">
    <vType id="car_d" vClass="passenger" probability="0"/>
  </vTypeDistribution>
"""


def write_demand(out, seed, veh_per_hour, ped_per_hour, end, cv_share, aggr_share):
    rnd = random.Random(seed)
    p_car = max(0.0, 1.0 - cv_share - aggr_share - 0.05)
    vt = VTYPES.replace(
        '  <vTypeDistribution id="mix">\n    <vType id="car_d" vClass="passenger" probability="0"/>\n  </vTypeDistribution>\n',
        f'  <vTypeDistribution id="mix" vTypes="car cv aggr truck" probabilities="{p_car:.3f} {cv_share:.3f} {aggr_share:.3f} 0.05"/>\n')
    lines = ["<routes>", vt.rstrip("\n"),
             '  <vType id="ped" vClass="pedestrian" maxSpeed="1.45" speedFactor="normc(0.95,0.18,0.55,1.4)"/>']
    turn = {"right": 0.15, "thru": 0.7, "left": 0.15}
    dest = {  # origin leg -> movement -> destination leg
        "N": {"right": "W", "thru": "S", "left": "E"},
        "E": {"right": "N", "thru": "W", "left": "S"},
        "S": {"right": "E", "thru": "N", "left": "W"},
        "W": {"right": "S", "thru": "E", "left": "N"},
    }
    fid = 0
    for o in LEGS:
        vph = veh_per_hour * rnd.uniform(0.7, 1.3)
        for mv, share in turn.items():
            rate = vph * share / 3600.0
            lines.append(f'  <flow id="f{fid}_{o}_{mv}" type="mix" from="{o}_in" to="{dest[o][mv]}_out" '
                         f'begin="0" end="{end}" period="exp({rate:.5f})" departLane="best" departSpeed="max"/>')
            fid += 1
    # pedestrians: each crossing both directions; the target (north) crossing is busier
    L = LEG - 10.4  # edge length
    pid = 0
    for leg in LEGS:
        rate = (ped_per_hour * (1.6 if leg == "N" else 0.6)) / 3600.0 / 2
        for frm, to in ((f"{leg}_in", f"{leg}_out"), (f"{leg}_out", f"{leg}_in")):
            dpos = L - 25 if frm.endswith("_in") else 25
            apos = 25 if to.endswith("_out") else L - 25
            lines.append(f'  <personFlow id="p{pid}_{leg}" type="ped" begin="0" end="{end}" period="exp({rate:.5f})" '
                         f'departPos="{dpos:.1f}">\n    <walk from="{frm}" to="{to}" arrivalPos="{apos:.1f}"/>\n  </personFlow>')
            pid += 1
    lines.append("</routes>")
    with open(os.path.join(out, f"demand_s{seed}.rou.xml"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sim_out/net")
    ap.add_argument("--demand-only", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--veh-per-hour", type=float, default=450, help="per approach, before random scaling")
    ap.add_argument("--ped-per-hour", type=float, default=150, help="per crossing, both directions")
    ap.add_argument("--end", type=float, default=2100)
    ap.add_argument("--cv-share", type=float, default=0.12)
    ap.add_argument("--aggr-share", type=float, default=0.08)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.demand_only:
        write_plain(a.out)
        run_netconvert(a.out)
        write_tls(a.out)
    write_demand(a.out, a.seed, a.veh_per_hour, a.ped_per_hour, a.end, a.cv_share, a.aggr_share)
    print("wrote network/demand to", a.out)
