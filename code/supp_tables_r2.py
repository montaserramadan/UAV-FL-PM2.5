"""
Corrected supplementary Tables S3/S4 for the second revision: same windows and
fleet sizes as v4, but with the 13 B LoRaWAN MAC overhead in every frame and a
serialised synchronous round (gateway delivers the model, then the node
uploads), instead of v4's min(uplink, downlink).  Writes results_supp_r2.json.
"""
import json, math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fl_experiment_v3 import lorawan_airtime

MAC, HDR = 13, 2
WIN = {"uplink SF7": (7, 222, 0.01), "RX1 SF7": (7, 222, 0.01),
       "RX2 SF9": (9, 115, 0.10), "RX2 SF12": (12, 51, 0.10)}
PAY = {"Dense f32": 5956, "Quantised": 1513, "Top-40% + q": 1809,
       "Top-10% + q + EF": 468}


def side(pb, sf, mx, duty):
    fr = max(1, math.ceil(pb / (mx - HDR)))
    t = fr * lorawan_airtime(mx + MAC, sf=sf)
    return fr, t, t / duty          # frames, airtime, duty-limited time


out = {"S3": {}, "S4": {}}
for name, pb in PAY.items():
    rows = {}
    for w, (sf, mx, d) in WIN.items():
        fr, t, tl = side(pb, sf, mx, d)
        n = 1 if w == "uplink SF7" else 3
        rows[w] = dict(frames=fr, per_day=86400.0 / (tl * n))
    out["S3"][name] = rows
    _, _, t_up = side(pb, 7, 222, 0.01)
    _, _, t_dn = side(pb, 9, 115, 0.10)
    sc = {str(N): 86400.0 / (t_up + N * t_dn) for N in (3, 5, 10, 20, 50)}
    sc["multicast"] = 86400.0 / (t_up + t_dn)
    sc["gateway_exceeds_node_at_N"] = t_up / t_dn
    out["S4"][name] = sc
    print(name, {w: (r["frames"], round(r["per_day"], 1)) for w, r in rows.items()})
    print("   ", {k: round(v, 1) for k, v in sc.items()})
json.dump(out, open(os.path.join(HERE, "results_supp_r2.json"), "w"), indent=2)
