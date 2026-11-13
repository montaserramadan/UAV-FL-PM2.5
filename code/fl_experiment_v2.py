"""
Extended FL experiment for the UAV industrial PM2.5 paper (v2, post-review).

Adds, on the SAME data (no new flights), the analyses the review panel asked for:
  A1  baselines: local-only + centralized (upper bound) + per-zone persistence
  A2  APPLIED communication compression inside FedAvg (quantise+sparsify the
      per-client update each round) with a compression-ratio SWEEP, so the
      accuracy cost of compression is MEASURED, not just byte-counted
  A3  multi-seed -> mean +/- std on every headline metric
  A4  FedProx (non-IID-aware baseline) vs vanilla FedAvg
  A6  honest LoRaWAN duty-cycle budget: frames-per-update + rounds-per-day

The base pipeline (data loading, scaling, split, models) is byte-identical to
fl_experiment.py so seed=42 reproduces the published numbers
(TinyLSTM T-FL RMSE 6.34 / R2 0.92). Writes results_fl_v2.json.
"""
import csv, os, json, math, copy
import numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

SEQ = 12
FEATS = ["pm25", "pm10", "co2", "ch2o", "voc"]
TARGET = "pm25"
HORIZON = 1
SEEDS = [42, 1, 2, 3, 4]                 # 42 first so it reproduces the paper number
ROUNDS, LOCAL_EPOCHS, LR = 30, 2, 1e-3
COMPRESS_KS = [1.0, 0.4, 0.2, 0.1]       # 1.0 = uncompressed (dense) reference
PROX_MU = 0.01

FACTORY = {"PCBA": "data/pretraining/PCBA_5month.csv",
           "CNC":  "data/pretraining/CNC_5month.csv",
           "plastic": "data/pretraining/plastic_5month.csv"}
UAV = {"Zone1": "data/uav/zone1_vehicle_maintenance.csv",
       "Zone2": "data/uav/zone2_galvanization.csv",
       "Zone3": "data/uav/zone3_recycling.csv"}
LAT = (40.80, 41.30); LON = (28.50, 29.40)


# ----------------------------- data loading (identical) -----------------------------
def load_factory(path):
    rows = list(csv.reader(open(os.path.join(ROOT, path), encoding="utf-8-sig")))
    idx = {"pm25": 1, "pm10": 2, "co2": 3, "voc": 4, "ch2o": 5}
    out = []
    for r in rows[1:]:
        try:
            out.append({k: float(r[i]) for k, i in idx.items()})
        except (ValueError, IndexError):
            continue
    return out


def load_uav(path):
    rows = list(csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8-sig")))
    out = []
    for r in rows:
        try:
            lat = float(r["Latitude"]); lon = float(r["Longitude"])
            alt = float(r["Altitude_m"]); spd = float(r["Speed_mps"])
            rec = dict(pm25=float(r["PM2.5"]), pm10=float(r["PM10"]),
                       co2=float(r["CO2"]), ch2o=float(r["CH2O"]), voc=float(r["TVOC"]))
        except (ValueError, KeyError):
            continue
        if not (LAT[0] < lat < LAT[1] and LON[0] < lon < LON[1]):
            continue
        if alt > 2000 or spd > 20:
            continue
        out.append(rec)
    return out


def make_sequences(records, mu, sd, tgt_mu, tgt_sd):
    arr = np.array([[rec[f] for f in FEATS] for rec in records], dtype=np.float32)
    arrs = (arr - mu) / sd
    tgt = np.array([rec[TARGET] for rec in records], dtype=np.float32)
    X, y = [], []
    for i in range(len(records) - SEQ - HORIZON + 1):
        X.append(arrs[i:i + SEQ]); y.append((tgt[i + SEQ + HORIZON - 1] - tgt_mu) / tgt_sd)
    if not X:
        return np.empty((0, SEQ, len(FEATS)), np.float32), np.empty((0,), np.float32)
    return np.stack(X), np.array(y, np.float32)


# ----------------------------- models (identical) -----------------------------
class TinyLSTM(nn.Module):
    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, n_feat, ch=16, k=3):
        super().__init__()
        self.conv = nn.Conv1d(n_feat, ch, k, padding=0)
        self.act = nn.ReLU()
        self.fc = nn.Linear(ch * (SEQ - k + 1), 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.act(self.conv(x))
        return self.fc(x.flatten(1)).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------- training -----------------------------
def train_local(model, X, y, epochs, lr, bs=8, global_state=None, prox_mu=0.0):
    """Local SGD. If prox_mu>0 and global_state given, adds FedProx proximal term."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    gref = None
    if prox_mu > 0 and global_state is not None:
        gref = {k: v.clone() for k, v in global_state.items()}
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            if gref is not None:                       # FedProx proximal penalty
                prox = 0.0
                for name, p in model.named_parameters():
                    prox = prox + ((p - gref[name]) ** 2).sum()
                loss = loss + (prox_mu / 2.0) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


@torch.no_grad()
def predict(model, X):
    model.eval()
    if len(X) == 0:
        return np.empty((0,))
    return model(torch.tensor(X)).numpy()


def metrics(pred_scaled, y_scaled, tgt_mu, tgt_sd):
    pred = pred_scaled * tgt_sd + tgt_mu
    y = y_scaled * tgt_sd + tgt_mu
    err = pred - y
    rmse = math.sqrt(np.mean(err ** 2)); mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def fedavg(states, weights):
    w = np.array(weights, float); w = w / w.sum()
    avg = copy.deepcopy(states[0])
    for k in avg:
        avg[k] = sum(states[i][k] * float(w[i]) for i in range(len(states)))
    return avg


# ----------------------------- A2: applied update compression -----------------------------
def compress_delta(delta, k, bits=8):
    """Quantise (int8, per-tensor scale) + top-k sparsify a per-client update.
    Returns (reconstructed delta dict, uplink payload bytes). k=1.0 -> dense f32."""
    if k >= 1.0:
        f32_bytes = sum(v.numel() for v in delta.values()) * 4
        return {kk: vv.clone() for kk, vv in delta.items()}, f32_bytes
    # global top-k across all params by magnitude
    flat = torch.cat([v.flatten().abs() for v in delta.values()])
    total = flat.numel()
    keep = max(1, int(k * total))
    thresh = torch.kthvalue(flat, total - keep + 1).values if keep < total else torch.tensor(-1.0)
    recon, kept_total, n_tensors = {}, 0, 0
    for name, v in delta.items():
        mask = v.abs() >= thresh
        vals = v[mask]
        if vals.numel() == 0:
            recon[name] = torch.zeros_like(v); n_tensors += 1; continue
        scale = vals.abs().max() / 127.0 + 1e-12       # int8 quantisation
        q = torch.round(vals / scale).clamp(-127, 127)
        deq = q * scale
        out = torch.zeros_like(v); out[mask] = deq
        recon[name] = out
        kept_total += int(vals.numel()); n_tensors += 1
    # payload: 1 value byte + 2 index bytes per kept weight + 4-byte per-tensor scale
    payload = kept_total * 3 + n_tensors * 4
    return recon, payload


def federated(model_cls, nf, init_state, clients, test, tgt_mu, tgt_sd,
              compress_k=1.0, prox_mu=0.0):
    """FedAvg. Clients upload a (optionally compressed) update = local - global.
    Server aggregates updates (weighted) and applies to global."""
    glob = model_cls(nf)
    if init_state is not None:
        glob.load_state_dict(copy.deepcopy(init_state))
    history, uplink_bytes = [], 0.0
    for _ in range(ROUNDS):
        gstate = copy.deepcopy(glob.state_dict())
        deltas, sizes, round_bytes = [], [], []
        for z, (Xtr, ytr) in clients.items():
            local = model_cls(nf)
            local.load_state_dict(copy.deepcopy(gstate))
            train_local(local, Xtr, ytr, LOCAL_EPOCHS, LR,
                        global_state=gstate, prox_mu=prox_mu)
            delta = {k: local.state_dict()[k] - gstate[k] for k in gstate}
            cdelta, pbytes = compress_delta(delta, compress_k)
            deltas.append(cdelta); sizes.append(len(Xtr)); round_bytes.append(pbytes)
        w = np.array(sizes, float); w = w / w.sum()
        newstate = {k: gstate[k] + sum(deltas[i][k] * float(w[i]) for i in range(len(deltas)))
                    for k in gstate}
        glob.load_state_dict(newstate)
        history.append(metrics(predict(glob, test[0]), test[1], tgt_mu, tgt_sd)[0])
        uplink_bytes += float(np.mean(round_bytes))     # avg per-client uplink this round
    return glob, history, uplink_bytes / ROUNDS         # avg per-client uplink / round


def train_full(model_cls, nf, init_state, X, y, epochs):
    """Non-federated training (local-only or centralized). epochs = ROUNDS*LOCAL_EPOCHS."""
    m = model_cls(nf)
    if init_state is not None:
        m.load_state_dict(copy.deepcopy(init_state))
    train_local(m, X, y, epochs, LR)
    return m


# ----------------------------- LoRaWAN budget (honest) -----------------------------
def lorawan_airtime(payload_bytes, sf=7, bw=125000, cr=1, preamble=8, header=True):
    Tsym = (2 ** sf) / bw
    de = 1 if sf >= 11 else 0
    ih = 0 if header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + 16 - 20 * ih
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    return (preamble + 4.25) * Tsym + n_payload * Tsym


def lorawan_budget(payload_bytes, sf=7, max_app_payload=222, duty=0.01):
    """Frames-per-update + duty-cycle-limited rounds/day for one node, EU868."""
    frames = max(1, math.ceil(payload_bytes / max_app_payload))
    per_frame_bytes = min(payload_bytes, max_app_payload)
    airtime_frame = lorawan_airtime(per_frame_bytes, sf=sf)
    airtime_update = frames * airtime_frame
    # 1% duty cycle: each TX forces (1/duty - 1) * airtime silence on the sub-band
    off_time_update = airtime_update * (1.0 / duty - 1.0)
    cycle_s = airtime_update + off_time_update           # min seconds between whole updates
    updates_per_hour = 3600.0 / cycle_s
    return dict(payload_bytes=payload_bytes, frames=frames,
                airtime_per_frame_s=round(airtime_frame, 3),
                airtime_update_s=round(airtime_update, 3),
                min_cycle_s=round(cycle_s, 1),
                rounds_per_hour=round(updates_per_hour, 2),
                rounds_per_day=round(updates_per_hour * 24, 1))


# ----------------------------- one seed -----------------------------
def run_seed(seed, data):
    np.random.seed(seed); torch.manual_seed(seed)
    mu, sd, tgt_mu, tgt_sd = data["scaler"]
    Xpre, ypre = data["pretrain"]
    idx = np.random.permutation(len(Xpre)); Xpre, ypre = Xpre[idx], ypre[idx]
    cut = int(0.9 * len(Xpre)); Xtr_p, ytr_p = Xpre[:cut], ypre[:cut]
    clients_tr = data["clients_tr"]; per_test = data["per_test"]; pooled = data["pooled_test"]

    out = {}
    for mname, mcls in [("TinyLSTM", TinyLSTM), ("1D-CNN", CNN1D)]:
        nf = len(FEATS)
        torch.manual_seed(seed)
        pre = mcls(nf); train_local(pre, Xtr_p, ytr_p, epochs=5, lr=1e-3, bs=256)
        pre_state = copy.deepcopy(pre.state_dict())
        E = ROUNDS * LOCAL_EPOCHS
        block = {"params": n_params(pre)}

        def glob_row(model):
            r = {z: dict(zip(("rmse", "mae", "r2"), metrics(predict(model, X), y, tgt_mu, tgt_sd)))
                 for z, (X, y) in per_test.items()}
            r["Global"] = dict(zip(("rmse", "mae", "r2"),
                                   metrics(predict(model, pooled[0]), pooled[1], tgt_mu, tgt_sd)))
            return r

        # --- A1 baselines ---
        # local-only (transfer init): each zone trains on its own data, tested on its own
        lo = {}
        for z, (Xtr, ytr) in clients_tr.items():
            torch.manual_seed(seed)
            m = train_full(mcls, nf, pre_state, Xtr, ytr, E)
            Xte, yte = per_test[z]
            lo[z] = dict(zip(("rmse", "mae", "r2"), metrics(predict(m, Xte), yte, tgt_mu, tgt_sd)))
        # pooled local-only = concat each zone's own-model predictions
        preds, ys = [], []
        for z, (Xtr, ytr) in clients_tr.items():
            torch.manual_seed(seed)
            m = train_full(mcls, nf, pre_state, Xtr, ytr, E)
            Xte, yte = per_test[z]
            preds.append(predict(m, Xte) * tgt_sd + tgt_mu); ys.append(yte * tgt_sd + tgt_mu)
        preds = np.concatenate(preds); ys = np.concatenate(ys); err = preds - ys
        lo["Global"] = dict(rmse=math.sqrt(np.mean(err ** 2)), mae=float(np.mean(np.abs(err))),
                            r2=float(1 - np.sum(err ** 2) / np.sum((ys - ys.mean()) ** 2)))
        block["local_only"] = lo

        # centralized upper bound (transfer init): pool all zone train data
        Xc = np.concatenate([c[0] for c in clients_tr.values()])
        yc = np.concatenate([c[1] for c in clients_tr.values()])
        torch.manual_seed(seed)
        cen = train_full(mcls, nf, pre_state, Xc, yc, E)
        block["centralized"] = glob_row(cen)

        # --- FL / T-FL (dense) ---
        torch.manual_seed(seed)
        gfl, hfl, _ = federated(mcls, nf, None, clients_tr, pooled, tgt_mu, tgt_sd)
        block["FL"] = glob_row(gfl); block["FL"]["convergence_rmse"] = hfl
        torch.manual_seed(seed)
        gtfl, htfl, _ = federated(mcls, nf, pre_state, clients_tr, pooled, tgt_mu, tgt_sd)
        block["T-FL"] = glob_row(gtfl); block["T-FL"]["convergence_rmse"] = htfl

        # --- A4 FedProx (transfer init) ---
        torch.manual_seed(seed)
        gpx, hpx, _ = federated(mcls, nf, pre_state, clients_tr, pooled, tgt_mu, tgt_sd,
                                prox_mu=PROX_MU)
        block["T-FedProx"] = glob_row(gpx); block["T-FedProx"]["convergence_rmse"] = hpx

        # --- A2 compression sweep on the deployed setting (T-FL) ---
        sweep = {}
        for k in COMPRESS_KS:
            torch.manual_seed(seed)
            gk, _, upl = federated(mcls, nf, pre_state, clients_tr, pooled, tgt_mu, tgt_sd,
                                   compress_k=k)
            g = metrics(predict(gk, pooled[0]), pooled[1], tgt_mu, tgt_sd)
            sweep[f"k={k}"] = dict(rmse=g[0], mae=g[1], r2=g[2],
                                   uplink_bytes_per_round=round(upl, 1),
                                   lorawan=lorawan_budget(upl))
        block["compression_sweep"] = sweep
        out[mname] = block
    return out


def main():
    print("Loading data (once) ...")
    fac = []
    for name, p in FACTORY.items():
        fac += load_factory(p)
    fa = np.array([[r[f] for f in FEATS] for r in fac], np.float32)
    mu, sd = fa.mean(0), fa.std(0) + 1e-8
    tgt_mu = float(fa[:, 0].mean()); tgt_sd = float(fa[:, 0].std() + 1e-8)
    Xpre, ypre = make_sequences(fac, mu, sd, tgt_mu, tgt_sd)

    uav = {z: load_uav(p) for z, p in UAV.items()}
    clients_tr, per_test, tX, tY = {}, {}, [], []
    for z, recs in uav.items():
        X, y = make_sequences(recs, mu, sd, tgt_mu, tgt_sd)
        ctr = int(0.7 * len(X))
        clients_tr[z] = (X[:ctr], y[:ctr]); per_test[z] = (X[ctr:], y[ctr:])
        tX.append(X[ctr:]); tY.append(y[ctr:])
    pooled_test = (np.concatenate(tX), np.concatenate(tY))

    # per-zone + pooled persistence (seed-independent)
    persist = {}
    pp, py = [], []
    for z, (X, y) in per_test.items():
        last = X[:, -1, 0] * sd[0] + mu[0]; yy = y * tgt_sd + tgt_mu; e = last - yy
        persist[z] = dict(rmse=math.sqrt(np.mean(e ** 2)), mae=float(np.mean(np.abs(e))),
                          r2=float(1 - np.sum(e ** 2) / np.sum((yy - yy.mean()) ** 2)))
        pp.append(last); py.append(yy)
    pp = np.concatenate(pp); py = np.concatenate(py); e = pp - py
    persist["Global"] = dict(rmse=math.sqrt(np.mean(e ** 2)), mae=float(np.mean(np.abs(e))),
                             r2=float(1 - np.sum(e ** 2) / np.sum((py - py.mean()) ** 2)))

    data = dict(scaler=(mu, sd, tgt_mu, tgt_sd), pretrain=(Xpre, ypre),
                clients_tr=clients_tr, per_test=per_test, pooled_test=pooled_test)

    all_seeds = []
    for s in SEEDS:
        print(f"  seed {s} ...")
        all_seeds.append(run_seed(s, data))

    # aggregate mean/std of Global RMSE/MAE/R2 per model/method
    def agg(model, method, field):
        vals = [sd_[model][method]["Global"][field] for sd_ in all_seeds
                if method in sd_[model]]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {"seeds": SEEDS, "persistence": persist, "models": {}}
    methods = ["local_only", "centralized", "FL", "T-FL", "T-FedProx"]
    for model in ["TinyLSTM", "1D-CNN"]:
        msum = {}
        for meth in methods:
            m_r, s_r = agg(model, meth, "rmse")
            m_m, s_m = agg(model, meth, "mae")
            m_2, s_2 = agg(model, meth, "r2")
            msum[meth] = dict(rmse_mean=m_r, rmse_std=s_r, mae_mean=m_m, mae_std=s_m,
                              r2_mean=m_2, r2_std=s_2)
        # compression sweep: mean over seeds
        sweep = {}
        for k in COMPRESS_KS:
            key = f"k={k}"
            rs = [sd_[model]["compression_sweep"][key]["rmse"] for sd_ in all_seeds]
            r2s = [sd_[model]["compression_sweep"][key]["r2"] for sd_ in all_seeds]
            ex = all_seeds[0][model]["compression_sweep"][key]
            sweep[key] = dict(rmse_mean=float(np.mean(rs)), rmse_std=float(np.std(rs)),
                              r2_mean=float(np.mean(r2s)),
                              uplink_bytes_per_round=ex["uplink_bytes_per_round"],
                              lorawan=ex["lorawan"])
        msum["compression_sweep"] = sweep
        msum["params"] = all_seeds[0][model]["params"]
        # keep seed-42 per-zone detail (index 0) for the Zone-3 discussion
        msum["seed42_perzone"] = {m: all_seeds[0][model][m] for m in
                                  ["local_only", "T-FL"] if m in all_seeds[0][model]}
        summary["models"][model] = msum

    json.dump({"summary": summary, "per_seed": all_seeds},
              open(os.path.join(HERE, "results_fl_v2.json"), "w"), indent=2)

    # ---- console report ----
    print("\n" + "=" * 66)
    print("PERSISTENCE  per-zone:", {z: round(persist[z]["rmse"], 2) for z in persist})
    for model in ["TinyLSTM", "1D-CNN"]:
        m = summary["models"][model]
        print(f"\n=== {model} ({m['params']} params) — Global RMSE mean±std over {len(SEEDS)} seeds ===")
        for meth in methods:
            d = m[meth]
            print(f"  {meth:12s} RMSE {d['rmse_mean']:5.2f}±{d['rmse_std']:.2f}   "
                  f"R2 {d['r2_mean']:5.2f}")
        print(f"  -- compression sweep (T-FL) --")
        for k in COMPRESS_KS:
            s = m["compression_sweep"][f"k={k}"]; lw = s["lorawan"]
            print(f"  k={k:<4} RMSE {s['rmse_mean']:5.2f}  uplink {s['uplink_bytes_per_round']:6.0f}B/rnd  "
                  f"frames {lw['frames']}  rounds/day {lw['rounds_per_day']}")
    print("\nSaved -> results_fl_v2.json")


if __name__ == "__main__":
    main()
