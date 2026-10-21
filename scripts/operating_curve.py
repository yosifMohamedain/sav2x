"""Alert operating curve (timely recall vs false-alert vehicles per hour) from saved decisions,
sweeping a threshold on raw risk. Usage: python scripts/operating_curve.py hours name=path ..."""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from evaluate import alert_events  # noqa: E402


def curve(decs, hours, n=40):
    thr = np.unique(np.concatenate([np.geomspace(1e-3, 0.95, n), [0.4]]))
    out = []
    for t in thr:
        a = alert_events(decs, None, float(t), hours)
        out.append({"thr": float(t), "fa_per_h": a["false_alert_vehicles_per_hour"], "timely": a["timely_recall"],
                    "recall": a["recall"], "precision": a["precision"]})
    return out


if __name__ == "__main__":
    hours = float(sys.argv[1])
    res = {}
    for spec in sys.argv[2:]:
        name, path = spec.split("=", 1)
        res[name] = curve(json.load(open(path)), hours)
    json.dump(res, open("results/operating_curves.json", "w"), indent=1)
    for k, v in res.items():
        best = {}
        for fa_cap in (10, 30, 60, 120, 250, 500, 1000):
            ok = [c for c in v if c["fa_per_h"] <= fa_cap]
            best[fa_cap] = max((c["timely"] for c in ok), default=0.0)
        print(k, {f"<= {c}/h": round(t, 2) for c, t in best.items()})
