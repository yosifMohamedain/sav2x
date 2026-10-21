"""Markdown tables from evaluate.py results.

    python scripts/summarize.py results/test/results.json results/heldout/results.json > results/summary.md
"""
import json
import sys

ROWS = [
    ("trajectory", ("all", "minADE@8s"), "minADE, 6 modes, 8 s (m)", 2),
    ("trajectory", ("all", "minFDE@8s"), "minFDE, 6 modes, 8 s (m)", 2),
    ("trajectory", ("all", "FDE1@8s"), "Most-likely-mode FDE, 8 s (m)", 2),
    ("trajectory", ("ingress_red", "minFDE@8s"), "  approaching on red, minFDE 8 s (m)", 2),
    ("trajectory", ("ingress_green", "minFDE@8s"), "  approaching on green, minFDE 8 s (m)", 2),
    ("trajectory", ("all", "ISS_ADE3@5s"), "ISS score: top-3 weighted ADE, 5 s (m)", 2),
    ("trajectory", ("all", "MR2@8s"), "Miss rate at 2 m, 8 s", 3),
    ("maneuver", ("all", "man_acc"), "Maneuver accuracy (before stop line)", 3),
    ("maneuver", ("turning", "man_acc"), "  turning vehicles", 3),
    ("arrival", "crossing/detect", "Crossing vehicles predicted to enter (P >= 0.5)", 3),
    ("arrival", "crossing/mae_all", "Crossing vehicles: crosswalk entry-time MAE (s)", 2),
    ("arrival", "crossing_red/mae_all", "  approaching on red (s)", 2),
    ("arrival", "crossing_green/mae_all", "  approaching on green (s)", 2),
    ("arrival", "crossing/cover80", "  80% interval coverage", 3),
    ("arrival", "all/brier@5s", "All vehicles: Brier, enters within 5 s", 3),
    ("conflict", ("risk_cal", "auroc"), "Conflict AUROC", 3),
    ("conflict", ("risk_cal", "auprc"), "Conflict AUPRC", 3),
    ("conflict", ("risk_raw", "ece"), "Calibration error, raw", 3),
    ("conflict", ("risk_cal", "ece"), "Calibration error, calibrated on val", 3),
    ("conflict", ("alerts", "events"), "Conflict vehicles", 0),
    ("conflict", ("alerts", "recall"), "Alert recall", 2),
    ("conflict", ("alerts", "timely_recall"), "Timely recall (lead >= TTA_min)", 2),
    ("conflict", ("alerts", "precision"), "Alert precision", 2),
    ("conflict", ("alerts", "F2"), "Alert F2", 2),
    ("conflict", ("alerts", "median_lead_s"), "Median lead time (s)", 1),
    ("conflict", ("alerts", "false_alert_vehicles_per_hour"), "False-alert vehicles per hour", 0),
    ("conflict", ("alerts", "false_alert_share"), "Share of non-conflict vehicles alerted", 3),
    ("conflict", ("alerts", "p_star"), "Threshold p* (from val)", 2),
]


def get(r, sec, key):
    v = r.get(sec, {})
    for k in (key if isinstance(key, tuple) else (key,)):
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return v


def table(res, names, labels=None):
    labels = labels or names
    out = ["| Metric | " + " | ".join(labels) + " |", "| --- |" + " --- |" * len(names)]
    for sec, key, lab, dp in ROWS:
        vals = []
        for n in names:
            v = get(res.get(n, {}), sec, key)
            vals.append("" if v is None else (f"{v:,.{dp}f}" if isinstance(v, (int, float)) else str(v)))
        if any(vals):
            out.append(f"| {lab} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def model_extras(r, names):
    out = ["| Model | Ped occupancy AUROC 0-5 / 5-10 / 10-20 s | Ped occupancy Brier 0-5 s | Conflict AUPRC, direct head | Conflict AUPRC, composed from arrival x occupancy |",
           "| --- | --- | --- | --- | --- |"]
    for n in names:
        m = r[n]
        if "occupancy" not in m:
            continue
        o = m["occupancy"]
        au = " / ".join(f"{o[h].get('auroc', float('nan')):.3f}" for h in ("0-5s", "5-10s", "10-20s"))
        out.append(f"| {n} | {au} | {o['0-5s'].get('brier', float('nan')):.3f} | "
                   f"{m['conflict']['risk_cal'].get('auprc', float('nan')):.3f} | "
                   f"{m['conflict_composed']['risk_cal'].get('auprc', float('nan')):.3f} |")
    return "\n".join(out)


def by_scenario(r, names):
    scs = sorted(next(iter(r[n]["conflict"]["alerts_by_scenario"] for n in names)))
    out = ["| Scenario | " + " | ".join(f"{n}: timely recall, false alerts/h" for n in names) + " |",
           "| --- |" + " --- |" * len(names)]
    for sc in scs:
        cells = []
        for n in names:
            a = r[n]["conflict"]["alerts_by_scenario"].get(sc, {})
            cells.append(f"{a.get('timely_recall', 0):.2f}, {a.get('false_alert_vehicles_per_hour', 0):,.0f} ({a.get('events', 0)} events)")
        out.append(f"| {sc} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(paths):
    for p in paths:
        r = json.load(open(p))
        names = [k for k, v in r.items() if isinstance(v, dict) and "trajectory" in v]
        print(f"## {r['split']} ({r['n']:,} decisions, {r['hours']:.1f} h, runs: {', '.join(r['runs'])})\n")
        print(table(r, names) + "\n")
        print(model_extras(r, names) + "\n")
        print(by_scenario(r, names) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
