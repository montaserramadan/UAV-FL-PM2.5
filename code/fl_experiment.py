"""
Transfer Federated Learning experiment for the UAV industrial PM2.5 paper.

Pipeline
--------
1. Pretrain TinyLSTM and 1D-CNN on three real factory datasets
   (PCBA, CNC, plastic; ~43k 5-min readings each) for next-step PM2.5
   forecasting  ->  initial weights theta_0.
2. Federated fine-tuning across 3 UAV zones (real cleaned flight logs) with
   FedAvg, for two settings that differ ONLY in initialization:
       FL    : random init
       T-FL  : pretrained init (theta_0)
3. Report per-client and global RMSE / MAE / R2 (in ug/m3), a persistence
   baseline (honesty check), and convergence-vs-rounds curves.
4. Model-compression accounting (8-bit quant + top-k sparsification) with
   REAL parameter counts, payload sizes and EU868 LoRaWAN airtime.

Everything is seeded and reproducible. Numbers printed here are the numbers
that go in the paper -- no hand-editing.
"""
import csv, os, json, math, copy
import numpy as np
import torch, torch.nn as nn

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

SEQ = 12                         # input sequence length (paper)
FEATS = ["pm25", "pm10", "co2", "ch2o", "voc"]   # features shared by both datasets
TARGET = "pm25"
HORIZON = 1                      # next-step

FACTORY = {
    "PCBA": "data/pretraining/PCBA_5month.csv",
    "CNC":  "data/pretraining/CNC_5month.csv",
    "plastic": "data/pretraining/plastic_5month.csv",
}
UAV = {
    "Zone1": "data/uav/zone1_vehicle_maintenance.csv",
    "Zone2": "data/uav/zone2_galvanization.csv",
    "Zone3": "data/uav/zone3_recycling.csv",
}
LAT = (40.80, 41.30); LON = (28.50, 29.40)


# ----------------------------- data loading -----------------------------
def load_factory(path):
    rows = list(csv.reader(open(os.path.join(ROOT, path), encoding="utf-8-sig")))
    hdr = rows[0]
    idx = {"pm25": 1, "pm10": 2, "co2": 3, "voc": 4, "ch2o": 5}  # by column order
    out = []
    for r in rows[1:]:
        try:
            rec = {k: float(r[i]) for k, i in idx.items()}
        except (ValueError, IndexError):
            continue
        out.append(rec)
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


def make_sequences(records, scaler_mu, scaler_sd, tgt_mu, tgt_sd):
    """Return X (n, SEQ, F) scaled, y (n,) scaled target = PM2.5 HORIZON ahead."""
    arr = np.array([[rec[f] for f in FEATS] for rec in records], dtype=np.float32)
    arrs = (arr - scaler_mu) / scaler_sd
    tgt = np.array([rec[TARGET] for rec in records], dtype=np.float32)
    X, y = [], []
    for i in range(len(records) - SEQ - HORIZON + 1):
        X.append(arrs[i:i + SEQ])
        y.append((tgt[i + SEQ + HORIZON - 1] - tgt_mu) / tgt_sd)
    if not X:
        return np.empty((0, SEQ, len(FEATS)), np.float32), np.empty((0,), np.float32)
    return np.stack(X), np.array(y, np.float32)


# ----------------------------- models -----------------------------
class TinyLSTM(nn.Module):
    def __init__(self, n_feat, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class CNN1D(nn.Module):
    """Compact 1D-CNN for local temporal feature extraction. Keeps the temporal
    dimension (no global pooling) so the most-recent observations -- which
    dominate next-step forecasting -- are preserved before the dense head."""
    def __init__(self, n_feat, ch=16, k=3):
        super().__init__()
        self.conv = nn.Conv1d(n_feat, ch, k, padding=0)   # SEQ -> SEQ-k+1
        self.act = nn.ReLU()
        self.fc = nn.Linear(ch * (SEQ - k + 1), 1)

    def forward(self, x):                 # x: (B, SEQ, F)
        x = x.transpose(1, 2)             # -> (B, F, SEQ)
        x = self.act(self.conv(x))        # -> (B, ch, SEQ-k+1)
        x = x.flatten(1)
        return self.fc(x).squeeze(-1)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ----------------------------- training helpers -----------------------------
def train_local(model, X, y, epochs, lr, bs=8):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt, yt = torch.tensor(X), torch.tensor(y)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, yt[idx])
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
    rmse = math.sqrt(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    ss_res = np.sum(err ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def fedavg(states, weights):
    w = np.array(weights, float); w = w / w.sum()
    avg = copy.deepcopy(states[0])
    for k in avg:
        avg[k] = sum(states[i][k] * float(w[i]) for i in range(len(states)))
    return avg


def federated(model_cls, n_feat, init_state, clients, test, tgt_mu, tgt_sd,
              rounds=30, epochs=2, lr=1e-3):
    """clients: {zone:(Xtr,ytr)} ; test: pooled (X,y). Returns global model + history."""
    glob = model_cls(n_feat)
    if init_state is not None:
        glob.load_state_dict(copy.deepcopy(init_state))
    history = []
    for rnd in range(rounds):
        states, sizes = [], []
        for z, (Xtr, ytr) in clients.items():
            local = model_cls(n_feat)
            local.load_state_dict(copy.deepcopy(glob.state_dict()))
            train_local(local, Xtr, ytr, epochs, lr)
            states.append(local.state_dict()); sizes.append(len(Xtr))
        glob.load_state_dict(fedavg(states, sizes))
        rmse, mae, r2 = metrics(predict(glob, test[0]), test[1], tgt_mu, tgt_sd)
        history.append(rmse)
    return glob, history


# ----------------------------- compression -----------------------------
def compression_report(model):
    P = n_params(model)
    f32 = P * 4
    q8 = P * 1                              # 8-bit weights
    k = 0.20
    kept = int(P * k)
    # sparse payload: value byte + ~2-byte index per kept weight
    sparse = kept * (1 + 2)
    return dict(params=P, float32_bytes=f32, int8_bytes=q8,
                topk_kept=kept, sparse_bytes=sparse,
                compression_ratio=round(f32 / sparse, 1))


def lorawan_airtime(payload_bytes, sf=7, bw=125000, cr=1, preamble=8, header=True):
    """EU868 LoRaWAN time-on-air (s) per LoRa modem spec (Semtech)."""
    Tsym = (2 ** sf) / bw
    de = 1 if sf >= 11 else 0
    ih = 0 if header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + 16 - 20 * ih
    den = 4 * (sf - 2 * de)
    n_payload = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    return (preamble + 4.25) * Tsym + n_payload * Tsym


# ----------------------------- main -----------------------------
def main():
    print("Loading factory pretraining data ...")
    fac = []
    for name, p in FACTORY.items():
        d = load_factory(p); fac += d
        print(f"  {name}: {len(d)} rows")
    # scaler from factory data (shared transferable pipeline)
    fa = np.array([[r[f] for f in FEATS] for r in fac], np.float32)
    mu, sd = fa.mean(0), fa.std(0) + 1e-8
    tgt_mu = float(fa[:, 0].mean()); tgt_sd = float(fa[:, 0].std() + 1e-8)
    Xpre, ypre = make_sequences(fac, mu, sd, tgt_mu, tgt_sd)
    # shuffle + split pretrain
    idx = np.random.permutation(len(Xpre)); Xpre, ypre = Xpre[idx], ypre[idx]
    cut = int(0.9 * len(Xpre))
    Xtr_p, ytr_p = Xpre[:cut], ypre[:cut]
    print(f"  pretrain sequences: {len(Xpre)} (train {len(Xtr_p)})")

    print("\nLoading UAV data ...")
    uav = {z: load_uav(p) for z, p in UAV.items()}
    clients_tr, test_X, test_y = {}, [], []
    per_client_test = {}
    for z, recs in uav.items():
        X, y = make_sequences(recs, mu, sd, tgt_mu, tgt_sd)
        n = len(X); ctr = int(0.7 * n)            # temporal split along flight path
        clients_tr[z] = (X[:ctr], y[:ctr])
        per_client_test[z] = (X[ctr:], y[ctr:])
        test_X.append(X[ctr:]); test_y.append(y[ctr:])
        print(f"  {z}: {len(recs)} pts -> {n} seq (train {ctr}, test {n-ctr})")
    pooled_test = (np.concatenate(test_X), np.concatenate(test_y))

    # persistence baseline (predict next PM2.5 = current PM2.5) on pooled test
    def persistence(per_client_test):
        preds, ys = [], []
        for z, (X, y) in per_client_test.items():
            last = X[:, -1, 0]                      # scaled current pm2.5 feature
            # convert: feature scaled with mu/sd of pm25 -> need raw; invert
            preds.append(last * sd[0] + mu[0]); ys.append(y * tgt_sd + tgt_mu)
        preds = np.concatenate(preds); ys = np.concatenate(ys)
        err = preds - ys
        return math.sqrt(np.mean(err**2)), float(np.mean(np.abs(err))), \
               float(1 - np.sum(err**2)/np.sum((ys-ys.mean())**2))
    pe = persistence(per_client_test)
    print(f"\nPersistence baseline (pooled test): RMSE={pe[0]:.2f} MAE={pe[1]:.2f} R2={pe[2]:.2f}")

    results = {"persistence": dict(rmse=pe[0], mae=pe[1], r2=pe[2]), "models": {}}

    for mname, mcls in [("TinyLSTM", TinyLSTM), ("1D-CNN", CNN1D)]:
        print("\n" + "=" * 60 + f"\nMODEL: {mname}\n" + "=" * 60)
        nf = len(FEATS)
        # --- pretrain ---
        torch.manual_seed(SEED)
        pre = mcls(nf)
        train_local(pre, Xtr_p, ytr_p, epochs=5, lr=1e-3, bs=256)
        pre_state = copy.deepcopy(pre.state_dict())
        print(f"  params = {n_params(pre)}")

        block = {}
        for setting, init in [("FL", None), ("T-FL", pre_state)]:
            torch.manual_seed(SEED)
            glob, hist = federated(mcls, nf, init, clients_tr, pooled_test,
                                   tgt_mu, tgt_sd, rounds=30, epochs=2, lr=1e-3)
            row = {}
            for z, (X, y) in per_client_test.items():
                row[z] = dict(zip(("rmse", "mae", "r2"),
                                  metrics(predict(glob, X), y, tgt_mu, tgt_sd)))
            row["Global"] = dict(zip(("rmse", "mae", "r2"),
                                     metrics(predict(glob, pooled_test[0]), pooled_test[1], tgt_mu, tgt_sd)))
            row["convergence_rmse"] = hist
            block[setting] = row
            g = row["Global"]
            print(f"  {setting:5s} Global: RMSE={g['rmse']:.2f} MAE={g['mae']:.2f} R2={g['r2']:.2f}")
        # compression on this model
        comp = compression_report(pre)
        comp["airtime_uncompressed_s"] = round(lorawan_airtime(comp["float32_bytes"]), 2)
        comp["airtime_compressed_s"] = round(lorawan_airtime(comp["sparse_bytes"]), 2)
        block["compression"] = comp
        print(f"  compression: {comp['params']} params, {comp['float32_bytes']}B f32 -> "
              f"{comp['sparse_bytes']}B sparse-int8 (CR {comp['compression_ratio']}x); "
              f"airtime {comp['airtime_uncompressed_s']}s -> {comp['airtime_compressed_s']}s")
        results["models"][mname] = block

    json.dump(results, open(os.path.join(HERE, "results_fl.json"), "w"), indent=2)
    print("\nSaved -> results_fl.json")
    return results


if __name__ == "__main__":
    main()
