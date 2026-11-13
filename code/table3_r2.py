"""
Table 3 for the second revision: every compression configuration with
  * corrected analytical capacities (13 B LoRaWAN MAC overhead now included,
    which v4 omitted), uplink cap, gateway cap at N=3 (RX2 SF9 unicast), and
    the sustainable rate for a SERIALISED synchronous round (downlink, then
    uplink), replacing v4's min(uplink, downlink);
  * simulated rounds/day at N=3 from lorawan_sim.py, on an ideal link and on a
    lossy one (PER 10 %, background load 0.05).
Writes results_table3_r2.json.
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lorawan_sim as L

v3 = json.load(open(os.path.join(HERE, "results_fl_v3.json")))["summary"]["models"]["TinyLSTM"]
v5 = json.load(open(os.path.join(HERE, "results_fl_v5.json")))["compression_baselines"]["configs"]
fac = v3["sparse_quant_factorial"]

ROWS = [  # label, payload, rmse mean, rmse std, source
    ("dense f32", fac["dense_f32"]),
    ("int8", fac["quant_only"]),
    ("top-40% f32", fac["sparse_only_k=0.4"]),
    ("top-40% int8", fac["both_k=0.4"]),
    ("top-20% f32", fac["sparse_only_k=0.2"]),
    ("top-20% int8", fac["both_k=0.2"]),
]
rows = [dict(label=l, payload=r["uplink_bytes_per_round"], rmse=r["rmse_mean"],
             rmse_std=r["rmse_std"], source="v3 factorial") for l, r in ROWS]
for l, k in (("top-20% int8 + EF", "top-k int8+EF k=0.2"),
             ("top-10% int8 + EF", "top-k int8+EF k=0.1"),
             ("EF-signSGD", "EF-signSGD")):
    c = v5[k]
    rows.append(dict(label=l, payload=c["bytes_per_round_mean"], rmse=c["final_rmse"]["mean"],
                     rmse_std=c["final_rmse"]["std"], source="v5 compression"))

SCEN = {"ideal": (0.0, 0.0), "lossy": (0.10, 0.05)}
N = 3
for r in rows:
    pb = r["payload"]
    U = L.nfrag(pb, L.UL["max_app"]); D = L.nfrag(pb, L.DL["max_app"])
    r["ul_frames"], r["dl_frames"] = U, D
    r["uplink_cap"] = 86400 * L.UL["duty"] / (U * L.T_UL_FULL)
    r["gateway_cap_N3"] = 86400 * L.DL["duty"] / (N * D * L.T_DL_FULL)
    r["v4_style_min"] = min(r["uplink_cap"], r["gateway_cap_N3"])
    r["serial_ceiling"] = L.ideal_serial(pb, N)
    for s, (per, bg) in SCEN.items():
        sims = [L.simulate(pb, N, per, bg, False, rounds=10, seed=i) for i in range(5)]
        r[f"sim_{s}"] = float(np.mean([x["rounds_per_day"] for x in sims]))
        r[f"sim_{s}_std"] = float(np.std([x["rounds_per_day"] for x in sims]))
        r[f"sim_{s}_full"] = float(np.mean([x["full_participation"] for x in sims]))
        r[f"sim_{s}_node_airtime_s"] = float(np.mean([x["node_airtime_s_round"] for x in sims]))
    print(f"{r['label']:20s} {pb:6.0f} B  RMSE {r['rmse']:.2f}  UL {U:2d}fr  cap {r['uplink_cap']:6.1f}  "
          f"gw {r['gateway_cap_N3']:6.1f}  min {r['v4_style_min']:6.1f}  serial {r['serial_ceiling']:6.1f}  "
          f"sim {r['sim_ideal']:6.1f} / lossy {r['sim_lossy']:6.1f}  airtime {r['sim_lossy_node_airtime_s']:.1f}s")
json.dump(rows, open(os.path.join(HERE, "results_table3_r2.json"), "w"), indent=2)
