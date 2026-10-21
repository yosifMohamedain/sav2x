"""Stitch SDSM temporary object IDs (and rotating BSM IDs) into persistent tracks.

J3224 object IDs are local to the sender's tracker and change whenever it
re-initializes; BSM temporary IDs rotate for privacy. A predictor needs
continuous histories, so detections are linked across ID changes with a
constant-velocity gate on position, plus type, heading and size checks.

This is the geometry-only association discussed for cross-RSU handover, in its
simplest single-RSU form. Two modes are produced so association error can be
separated from prediction error:
    ground_truth : track = simulation identity (oracle; sim only)
    estimated    : track = output of this module (what the field gets)
"""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def stitch(t_obs, obj_id, x, y, speed, heading, typ, length, max_gap_s=2.0, gate_m=2.5, gate_per_s=1.5):
    """Return an array of track ids (ints) for detections given in any order.

    Inputs are 1-D arrays of equal length. Detections that keep the same
    obj_id with small gaps stay on one track; a new or reused obj_id is matched
    to tracks that went quiet within max_gap_s, by gated nearest neighbor with
    a Hungarian assignment per frame."""
    n = len(t_obs)
    order = np.lexsort((obj_id, t_obs))
    track = np.full(n, -1, dtype=np.int64)
    tracks = {}          # tid -> dict(last_t, x, y, vx, vy, typ, heading, speed, length)
    obj2trk = {}         # temp obj id -> tid
    next_tid = 0
    # frames: identical t_obs within 1 ms
    ts = t_obs[order]
    bounds = np.flatnonzero(np.diff(ts) > 1e-3) + 1
    for frame in np.split(order, bounds):
        t = float(t_obs[frame[0]])
        updated = set()
        unmatched = []
        for i in frame:
            oid = int(obj_id[i])
            tid = obj2trk.get(oid)
            if tid is not None:
                tr = tracks[tid]
                dt = t - tr["last_t"]
                px, py = tr["x"] + tr["vx"] * dt, tr["y"] + tr["vy"] * dt
                if 0 < dt <= max_gap_s and math.hypot(x[i] - px, y[i] - py) < gate_m + gate_per_s * dt + 2.0 \
                        and tr["typ"] == typ[i] and tid not in updated:
                    _update(tr, t, x[i], y[i], speed[i], heading[i], length[i])
                    track[i] = tid
                    updated.add(tid)
                    continue
            unmatched.append(i)
        if unmatched:
            cands = [tid for tid, tr in tracks.items()
                     if tid not in updated and 0 < t - tr["last_t"] <= max_gap_s]
            cost = np.full((len(unmatched), len(cands)), 1e6)
            for a, i in enumerate(unmatched):
                for b, tid in enumerate(cands):
                    tr = tracks[tid]
                    if tr["typ"] != typ[i]:
                        continue
                    dt = t - tr["last_t"]
                    px, py = tr["x"] + tr["vx"] * dt, tr["y"] + tr["vy"] * dt
                    d = math.hypot(x[i] - px, y[i] - py)
                    gate = gate_m + gate_per_s * dt
                    if d > gate:
                        continue
                    dh = abs((heading[i] - tr["heading"] + 180) % 360 - 180)
                    if tr["speed"] > 1.0 and speed[i] > 1.0 and dh > 45:
                        continue
                    if typ[i] == "vehicle" and abs(length[i] - tr["length"]) > 2.5:
                        continue
                    cost[a, b] = d / gate + 0.2 * abs(speed[i] - tr["speed"]) + 0.3 * dt
            rows, cols = linear_sum_assignment(cost) if len(cands) else ([], [])
            assigned = {}
            for r, c in zip(rows, cols):
                if cost[r, c] < 1e5:
                    assigned[r] = cands[c]
            for a, i in enumerate(unmatched):
                tid = assigned.get(a)
                if tid is None:
                    tid = next_tid
                    next_tid += 1
                    tracks[tid] = {"last_t": t, "x": x[i], "y": y[i], "vx": 0.0, "vy": 0.0,
                                   "typ": typ[i], "heading": heading[i], "speed": speed[i], "length": length[i]}
                else:
                    _update(tracks[tid], t, x[i], y[i], speed[i], heading[i], length[i])
                obj2trk[int(obj_id[i])] = tid
                track[i] = tid
                updated.add(tid)
        # drop long-dead tracks
        if len(tracks) > 400:
            for tid in [k for k, tr in tracks.items() if t - tr["last_t"] > 10 * max_gap_s]:
                del tracks[tid]
    return track


def _update(tr, t, x, y, speed, heading, length):
    dt = t - tr["last_t"]
    if dt > 0:
        a = math.radians(90 - heading)
        # blend measured heading/speed with the finite-difference velocity
        vx_m, vy_m = speed * math.cos(a), speed * math.sin(a)
        vx_d, vy_d = (x - tr["x"]) / dt, (y - tr["y"]) / dt
        tr["vx"], tr["vy"] = 0.7 * vx_m + 0.3 * vx_d, 0.7 * vy_m + 0.3 * vy_d
    tr.update(last_t=t, x=x, y=y, speed=speed, heading=heading, length=0.9 * tr["length"] + 0.1 * length)


def association_quality(track, truth):
    """Purity and identity-switch counts of estimated tracks against truth ids."""
    track = np.asarray(track)
    truth = np.asarray(truth)
    pure, total, frag = 0, 0, 0
    for tid in np.unique(track):
        m = truth[track == tid]
        vals, cnt = np.unique(m, return_counts=True)
        pure += cnt.max()
        total += len(m)
    for g in np.unique(truth):
        frag += len(np.unique(track[truth == g])) - 1
    return {"purity": pure / max(total, 1), "fragmentations": int(frag),
            "n_tracks": int(len(np.unique(track))), "n_truth": int(len(np.unique(truth)))}
