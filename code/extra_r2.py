"""Two follow-ups from the pre-submission review (second revision):
  1. centralised upper bound under UAV-domain standardisation (is the 5.66 T-FL
     gain matched by a lower centralised bound?);
  2. a non-linear time-series baseline at every horizon: per-zone gradient-
     boosted trees on the same flattened 60 s window (deterministic).
Writes results_extra_r2.json."""
import json, math, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fl_experiment_v5 as v5
from fl_experiment_v3 import TinyLSTM, ROUNDS, LOCAL_EPOCHS, train_full, predict, metrics, SEEDS, FEATS
from sklearn.ensemble import HistGradientBoostingRegressor

out = {"centralised": {}, "gbt": {}}
for variant in ("orig", "domainstd"):
    vals = []
    for s in SEEDS:
        d = v5.build(H=1, variant=variant, seed=s)
        pre = v5.pretrain(TinyLSTM, d, s)
        Xc = np.concatenate([c[0] for c in d["clients_tr"].values()])
        yc = np.concatenate([c[1] for c in d["clients_tr"].values()])
        torch.manual_seed(s)
        m = train_full(TinyLSTM, len(FEATS), pre, Xc, yc, ROUNDS * LOCAL_EPOCHS)
        vals.append(v5.eval_global(m, d)["Global"]["rmse"])
    out["centralised"][variant] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), per_seed=vals)
    print("centralised", variant, round(np.mean(vals), 2), round(np.std(vals), 2), flush=True)

for H in v5.HORIZONS:
    d = v5.build(H=H)
    preds, ys, per = [], [], {}
    for z, (Xtr, ytr) in d["clients_tr"].items():
        Xte, yte = d["per_test"][z]
        g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0)
        g.fit(Xtr.reshape(len(Xtr), -1), ytr)
        p = v5.unscale(d, g.predict(Xte.reshape(len(Xte), -1))); yy = v5.unscale(d, yte)
        per[z] = math.sqrt(np.mean((p - yy) ** 2)); preds.append(p); ys.append(yy)
    out["gbt"][f"H={H}"] = dict(Global=v5.pooled_rmse_of(preds, ys)["rmse"], per_zone=per)
    print("GBT", H, round(out["gbt"][f"H={H}"]["Global"], 2), flush=True)
json.dump(out, open(os.path.join(HERE, "results_extra_r2.json"), "w"), indent=2)
