"""
FL experiment v3 for the UAV industrial PM2.5 paper (dean-review revision).

Builds on fl_experiment_v2.py and adds, on the SAME data (no new flights), the
three rigour items the dean asked for:

  R1  Temporal-leakage-safe validation.  v2 windowed the full per-zone record and
      then split X at 0.7, which lets ~SEQ windows straddle the train/test cut
      (a test window's input rows overlap train rows).  v3 splits each zone's
      records chronologically FIRST, then windows train and test independently,
      so test windows use only test rows.  Zero row overlap across the cut.

  R2  init x federation 2x2 grid.  v2 already ran a TRANSFER-init local baseline
      (mislabelled "local-only") and a random-init FL.  v3 adds the missing
      random-init local cell so the four combinations {random,transfer} x
      {local,federated} are all present, cleanly isolating what transfer buys and
      what federation buys on top of it.

  R3  Sparsity vs quantisation, separated.  v2 folded top-k sparsification and
      int8 quantisation into one knob.  v3 runs the 2x2 factorial
      {sparsify?} x {quantise?} at the operating points so each effect's accuracy
      cost and byte saving is attributed, not just the combination.

Everything else (models, FedAvg/FedProx, multi-seed, LoRaWAN duty-cycle budget)
is identical to v2.  Writes results_fl_v3.json.
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
SEEDS = [42, 1, 2, 3, 4]
ROUNDS, LOCAL_EPOCHS, LR = 30, 2, 1e-3
COMPRESS_KS = [1.0, 0.4, 0.2, 0.1]
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
    """Window a contiguous record list.  Callers pass ALREADY-SPLIT records
    (R1): train and test are windowed separately so no window crosses the cut."""
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
            if gref is not None:
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


# ----------------------------- R3: compression with separable knobs -----------------------------
def compress_delta(delta, k, sparsify=True, quantise=True, bits=8):
    """Per-client update compression with INDEPENDENT sparsify / quantise knobs.

      sparsify=True  -> keep top-k magnitude weight changes, zero the rest (indexed)
      quantise=True  -> int8 per-tensor scale on the transmitted values
    Returns (reconstructed delta dict, uplink payload bytes).

    Payload accounting (per kept weight): value byte(s) + index bytes if sparse.
      dense f32   : total*4                          + n_tensors*0
      quant only  : total*1                          + n_tensors*4   (per-tensor scale)
      sparse f32  : kept*4 (value) + kept*2 (index)  + n_tensors*0
      both        : kept*1 (value) + kept*2 (index)  + n_tensors*4
    """
    val_bytes = 1 if quantise else 4
    n_tensors = len(delta)

    # --- select which entries to transmit ---
    if sparsify and k < 1.0:
        flat = torch.cat([v.flatten().abs() for v in delta.values()])
        total = flat.numel()
        keep = max(1, int(k * total))
        thresh = torch.kthvalue(flat, total - keep + 1).values if keep < total else torch.tensor(-1.0)
    else:
        thresh = None  # dense: transmit everything

    recon, kept_total = {}, 0
    for name, v in delta.items():
        if thresh is not None:
            mask = v.abs() >= thresh
        else:
            mask = torch.ones_like(v, dtype=torch.bool)
        vals = v[mask]
        if vals.numel() == 0:
            recon[name] = torch.zeros_like(v); continue
        if quantise:
            scale = vals.abs().max() / 127.0 + 1e-12
            q = torch.round(vals / scale).clamp(-127, 127)
            vals = q * scale
        out = torch.zeros_like(v); out[mask] = vals
        recon[name] = out
        kept_total += int(mask.sum().item())

    idx_bytes = 2 if (sparsify and k < 1.0) else 0
    scale_bytes = 4 if quantise else 0
    payload = kept_total * (val_bytes + idx_bytes) + n_tensors * scale_bytes
    return recon, payload


def federated(model_cls, nf, init_state, clients, test, tgt_mu, tgt_sd,
              compress_k=1.0, sparsify=True, quantise=True, prox_mu=0.0):
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
            cdelta, pbytes = compress_delta(delta, compress_k,
                                            sparsify=sparsify, quantise=quantise)
            deltas.append(cdelta); sizes.append(len(Xtr)); round_bytes.append(pbytes)
        w = np.array(sizes, float); w = w / w.sum()
        newstate = {k: gstate[k] + sum(deltas[i][k] * float(w[i]) for i in range(len(deltas)))
                    for k in gstate}
        glob.load_state_dict(newstate)
        history.append(metrics(predict(glob, test[0]), test[1], tgt_mu, tgt_sd)[0])
        uplink_bytes += float(np.mean(round_bytes))
    return glob, history, uplink_bytes / ROUNDS


def train_full(model_cls, nf, init_state, X, y, epochs):
    m = model_cls(nf)
    if init_state is not None:
        m.load_state_dict(copy.deepcopy(init_state))
    train_local(m, X, y, epochs, LR)
    return m


# ----------------------------- LoRaWAN budget (identical) -----------------------------
def lorawan_airtime(payload_bytes, sf=7, bw=125000, cr=1, preamble=8, header=True):
    Tsym = (2 ** sf) / bw
    de = 1 if sf >= 11 else 0
    ih = 0 if header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + 16 - 20 * ih
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    return (preamble + 4.25) * Tsym + n_payload * Tsym


def lorawan_budget(payload_bytes, sf=7, max_app_payload=222, duty=0.01):
    frames = max(1, math.ceil(payload_bytes / max_app_payload))
    per_frame_bytes = min(payload_bytes, max_app_payload)
    airtime_frame = lorawan_airtime(per_frame_bytes, sf=sf)
    airtime_update = frames * airtime_frame
    off_time_update = airtime_update * (1.0 / duty - 1.0)
    cycle_s = airtime_update + off_time_update
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

        # --- R2: init x federation grid -----------------------------------
        # local baselines, both initialisations.  init=None -> random; pre_state -> transfer.
        def local_grid(init_state):
            lo = {}
            for z, (Xtr, ytr) in clients_tr.items():
                torch.manual_seed(seed)
                m = train_full(mcls, nf, init_state, Xtr, ytr, E)
                Xte, yte = per_test[z]
                lo[z] = dict(zip(("rmse", "mae", "r2"), metrics(predict(m, Xte), yte, tgt_mu, tgt_sd)))
            preds, ys = [], []
            for z, (Xtr, ytr) in clients_tr.items():
                torch.manual_seed(seed)
                m = train_full(mcls, nf, init_state, Xtr, ytr, E)
                Xte, yte = per_test[z]
                preds.append(predict(m, Xte) * tgt_sd + tgt_mu); ys.append(yte * tgt_sd + tgt_mu)
            preds = np.concatenate(preds); ys = np.concatenate(ys); err = preds - ys
            lo["Global"] = dict(rmse=math.sqrt(np.mean(err ** 2)), mae=float(np.mean(np.abs(err))),
                                r2=float(1 - np.sum(err ** 2) / np.sum((ys - ys.mean()) ** 2)))
            return lo

        block["local_random"] = local_grid(None)        # random-init local (NEW cell)
        block["local_only"] = local_grid(pre_state)     # transfer-init local = T-Local

        # centralized upper bound (transfer init)
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

        # --- FedProx (transfer init) ---
        torch.manual_seed(seed)
        gpx, hpx, _ = federated(mcls, nf, pre_state, clients_tr, pooled, tgt_mu, tgt_sd,
                                prox_mu=PROX_MU)
        block["T-FedProx"] = glob_row(gpx); block["T-FedProx"]["convergence_rmse"] = hpx

        # --- A2 combined compression sweep on T-FL (sparsify + quantise) ---
        sweep = {}
        for k in COMPRESS_KS:
            torch.manual_seed(seed)
            gk, _, upl = federated(mcls, nf, pre_state, clients_tr, pooled, tgt_mu, tgt_sd,
                                   compress_k=k, sparsify=True, quantise=True)
            g = metrics(predict(gk, pooled[0]), pooled[1], tgt_mu, tgt_sd)
            sweep[f"k={k}"] = dict(rmse=g[0], mae=g[1], r2=g[2],
                                   uplink_bytes_per_round=round(upl, 1),
                                   lorawan=lorawan_budget(upl))
        block["compression_sweep"] = sweep

        # --- R3: sparsity vs quantisation factorial on T-FL ---
        # dense f32 (neither), quant-only (dense int8), sparse-only f32 at each k, both at each k.
        factorial = {}

        def run_cfg(kk, sp, qz):
            torch.manual_seed(seed)
            gk, _, upl = federated(mcls, nf, pre_state, clients_tr, pooled, tgt_mu, tgt_sd,
                                   compress_k=kk, sparsify=sp, quantise=qz)
            g = metrics(predict(gk, pooled[0]), pooled[1], tgt_mu, tgt_sd)
            return dict(rmse=g[0], r2=g[2], uplink_bytes_per_round=round(upl, 1),
                        lorawan=lorawan_budget(upl))

        factorial["dense_f32"] = run_cfg(1.0, False, False)
        factorial["quant_only"] = run_cfg(1.0, False, True)      # dense int8
        for k in [0.4, 0.2]:
            factorial[f"sparse_only_k={k}"] = run_cfg(k, True, False)   # top-k f32
            factorial[f"both_k={k}"] = run_cfg(k, True, True)          # top-k int8
        block["sparse_quant_factorial"] = factorial

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

    # R1: split each zone's RECORDS chronologically, THEN window train/test apart.
    uav = {z: load_uav(p) for z, p in UAV.items()}
    clients_tr, per_test, tX, tY = {}, {}, [], []
    split_report = {}
    for z, recs in uav.items():
        crec = int(0.7 * len(recs))
        train_recs, test_recs = recs[:crec], recs[crec:]
        Xtr, ytr = make_sequences(train_recs, mu, sd, tgt_mu, tgt_sd)
        Xte, yte = make_sequences(test_recs, mu, sd, tgt_mu, tgt_sd)
        clients_tr[z] = (Xtr, ytr); per_test[z] = (Xte, yte)
        tX.append(Xte); tY.append(yte)
        split_report[z] = dict(records=len(recs), train_records=len(train_recs),
                               test_records=len(test_recs),
                               train_windows=int(len(Xtr)), test_windows=int(len(Xte)))
    pooled_test = (np.concatenate(tX), np.concatenate(tY))
    print("  split (R1, leakage-safe):", json.dumps(split_report))

    # per-zone + pooled persistence on the leakage-safe test windows
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

    def agg(model, method, field):
        vals = [sd_[model][method]["Global"][field] for sd_ in all_seeds
                if method in sd_[model]]
        return float(np.mean(vals)), float(np.std(vals))

    summary = {"seeds": SEEDS, "split_report": split_report, "persistence": persist, "models": {}}
    methods = ["local_random", "local_only", "centralized", "FL", "T-FL", "T-FedProx"]
    for model in ["TinyLSTM", "1D-CNN"]:
        msum = {}
        for meth in methods:
            m_r, s_r = agg(model, meth, "rmse")
            m_m, s_m = agg(model, meth, "mae")
            m_2, s_2 = agg(model, meth, "r2")
            msum[meth] = dict(rmse_mean=m_r, rmse_std=s_r, mae_mean=m_m, mae_std=s_m,
                              r2_mean=m_2, r2_std=s_2)
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
        # R3 factorial: mean RMSE over seeds, payload/lorawan from seed-42 (deterministic)
        fac_keys = list(all_seeds[0][model]["sparse_quant_factorial"].keys())
        factorial = {}
        for fk in fac_keys:
            rs = [sd_[model]["sparse_quant_factorial"][fk]["rmse"] for sd_ in all_seeds]
            ex = all_seeds[0][model]["sparse_quant_factorial"][fk]
            factorial[fk] = dict(rmse_mean=float(np.mean(rs)), rmse_std=float(np.std(rs)),
                                 uplink_bytes_per_round=ex["uplink_bytes_per_round"],
                                 lorawan=ex["lorawan"])
        msum["sparse_quant_factorial"] = factorial
        msum["params"] = all_seeds[0][model]["params"]
        msum["seed42_perzone"] = {m: all_seeds[0][model][m] for m in
                                  ["local_random", "local_only", "T-FL"] if m in all_seeds[0][model]}
        summary["models"][model] = msum

    json.dump({"summary": summary, "per_seed": all_seeds},
              open(os.path.join(HERE, "results_fl_v3.json"), "w"), indent=2)

    # ---- console report ----
    print("\n" + "=" * 70)
    print("PERSISTENCE per-zone:", {z: round(persist[z]["rmse"], 2) for z in persist})
    for model in ["TinyLSTM", "1D-CNN"]:
        m = summary["models"][model]
        print(f"\n=== {model} ({m['params']} params) — Global RMSE mean±std over {len(SEEDS)} seeds ===")
        for meth in methods:
            d = m[meth]
            print(f"  {meth:13s} RMSE {d['rmse_mean']:5.2f}±{d['rmse_std']:.2f}   R2 {d['r2_mean']:5.2f}")
        print("  -- init x federation 2x2 (RMSE) --")
        print(f"     {'':10s} local            federated")
        print(f"     random   {m['local_random']['rmse_mean']:5.2f}            {m['FL']['rmse_mean']:5.2f}")
        print(f"     transfer {m['local_only']['rmse_mean']:5.2f}            {m['T-FL']['rmse_mean']:5.2f}")
        print("  -- combined compression sweep (T-FL) --")
        for k in COMPRESS_KS:
            s = m["compression_sweep"][f"k={k}"]; lw = s["lorawan"]
            print(f"     k={k:<4} RMSE {s['rmse_mean']:5.2f}  {s['uplink_bytes_per_round']:6.0f}B  "
                  f"frames {lw['frames']}  rounds/day {lw['rounds_per_day']}")
        print("  -- R3 sparsity vs quantisation factorial --")
        for fk, s in m["sparse_quant_factorial"].items():
            lw = s["lorawan"]
            print(f"     {fk:18s} RMSE {s['rmse_mean']:5.2f}  {s['uplink_bytes_per_round']:6.0f}B  frames {lw['frames']}")
    print("\nSaved -> results_fl_v3.json")


if __name__ == "__main__":
    main()
