"""Run SUMO for one prepared run directory and log ground truth at 10 Hz.

Logged: every vehicle (position of the front bumper, angle, speed, acceleration,
signal bits, lane), every pedestrian (position, edge), and the controller's
internal timing (phase, time spent, planned switch). emulate_v2x.py turns this
into BSM / SDSM / SPaT streams.

Usage:
    python sim/run_sim.py --run data_out/runs/base_s1
"""
import argparse
import json
import os
import sys

import numpy as np

try:
    import sumo
    os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
except ImportError:
    pass
import traci
import traci.constants as tc

VEH_VARS = [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED, tc.VAR_ACCELERATION,
            tc.VAR_SIGNALS, tc.VAR_LANE_ID]
PER_VARS = [tc.VAR_POSITION, tc.VAR_ANGLE, tc.VAR_SPEED, tc.VAR_ROAD_ID]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory made by sim/scenario.py")
    ap.add_argument("--tls-id", default="C")
    a = ap.parse_args()
    sc = json.load(open(os.path.join(a.run, "scenario.json")))
    warmup, end = sc["warmup_s"], sc["warmup_s"] + sc["logged_s"]
    sumo_bin = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
    traci.start([sumo_bin, "-n", os.path.join(a.run, "x.net.xml"), "-a", os.path.join(a.run, "tls.add.xml"),
                 "-r", os.path.join(a.run, "demand.rou.xml"), "--step-length", "0.1", "--seed", str(sc["seed"]),
                 "--no-step-log", "true", "--no-warnings", "true", "--time-to-teleport", "300",
                 "--pedestrian.model", "striping"])
    tls = a.tls_id
    logic = [l for l in traci.trafficlight.getAllProgramLogics(tls)
             if l.programID == traci.trafficlight.getProgram(tls)][0]
    phases = [{"state": p.state, "duration": p.duration, "minDur": p.minDur, "maxDur": p.maxDur, "name": p.name}
              for p in logic.phases]
    vrows, prows, trows = [], [], []
    vmeta, pmeta = {}, {}
    while traci.simulation.getTime() < end:
        traci.simulationStep()
        t = round(traci.simulation.getTime(), 1)
        for vid in traci.simulation.getDepartedIDList():
            traci.vehicle.subscribe(vid, VEH_VARS)
            vmeta[vid] = {"type": traci.vehicle.getTypeID(vid), "length": traci.vehicle.getLength(vid),
                          "width": traci.vehicle.getWidth(vid), "route": list(traci.vehicle.getRoute(vid)),
                          "depart": t}
        for pid in traci.simulation.getDepartedPersonIDList():
            traci.person.subscribe(pid, PER_VARS)
            pmeta[pid] = {"depart": t, "speed_max": traci.person.getMaxSpeed(pid)}
        for vid in traci.simulation.getArrivedIDList():
            if vid in vmeta:
                vmeta[vid]["arrive"] = t
        if t < warmup:
            continue
        for vid, r in traci.vehicle.getAllSubscriptionResults().items():
            x, y = r[tc.VAR_POSITION]
            vrows.append((t, vid, x, y, r[tc.VAR_ANGLE], r[tc.VAR_SPEED], r[tc.VAR_ACCELERATION],
                          r[tc.VAR_SIGNALS], r[tc.VAR_LANE_ID]))
        for pid, r in traci.person.getAllSubscriptionResults().items():
            x, y = r[tc.VAR_POSITION]
            prows.append((t, pid, x, y, r[tc.VAR_ANGLE], r[tc.VAR_SPEED], r[tc.VAR_ROAD_ID]))
        trows.append((t, traci.trafficlight.getPhase(tls), traci.trafficlight.getSpentDuration(tls),
                      traci.trafficlight.getNextSwitch(tls)))
    traci.close()

    def cols(rows, names, dtypes):
        c = list(zip(*rows)) if rows else [[] for _ in names]
        return {n: np.asarray(v, dtype=d) for n, v, d in zip(names, c, dtypes)}

    np.savez_compressed(os.path.join(a.run, "gt_vehicles.npz"), **cols(
        vrows, ["t", "id", "x", "y", "angle", "speed", "accel", "signals", "lane"],
        [np.float64, object, np.float32, np.float32, np.float32, np.float32, np.float32, np.int32, object]))
    np.savez_compressed(os.path.join(a.run, "gt_persons.npz"), **cols(
        prows, ["t", "id", "x", "y", "angle", "speed", "edge"],
        [np.float64, object, np.float32, np.float32, np.float32, np.float32, object]))
    np.savez_compressed(os.path.join(a.run, "gt_tls.npz"), **cols(
        trows, ["t", "phase", "spent", "next_switch"], [np.float64, np.int32, np.float32, np.float64]))
    json.dump({"tls_id": tls, "phases": phases, "vehicles": vmeta, "persons": pmeta, "warmup": warmup,
               "end": end, "step_s": 0.1}, open(os.path.join(a.run, "meta.json"), "w"))
    print(f"{a.run}: {len(vrows)} vehicle rows, {len(prows)} person rows")


if __name__ == "__main__":
    sys.exit(main())
