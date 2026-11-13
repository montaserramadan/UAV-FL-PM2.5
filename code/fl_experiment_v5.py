"""
FL experiment v5 for the UAV industrial PM2.5 paper (IEEE IoT-J second revision).

Imports fl_experiment_v3 and fl_experiment_v4 UNCHANGED (same data loading,
leakage-safe split, models, training loop, FedAvg, seeds).  Every number in the
first-revision manuscript therefore still reproduces from v3/v4; v5 only adds
the analyses the second-round reviewers asked for.

  H   Forecast horizon (Reviewer 1, comment 2).  Horizons of 1, 6, 12 and 24
      steps (5 s, 30 s, 60 s, 120 s).  Pretraining is redone at each horizon.
      Baselines: persistence, a per-zone ridge autoregression on the same
      60 s window, transfer-initialised local training and T-FL.

  F   What federation adds over transfer-initialised local training
      (Reviewer 1, comment 3).  Ten seeds, paired per seed, for random-init
      local, transfer-init local (T-Local), T-FL and int8 T-FL.  Also answers
      Reviewer 2, comment 3 (paired dense-vs-int8 difference).  Plus the
      new-site question that local training cannot answer: a held-out zone is
      served either by the two-zone federated model or by the best single other
      zone's local model.

  P   Pretraining-source ablation (Reviewer 1, comment 5).  The source is
      5-minute factory data and the target is 5-second UAV data.  Variants:
        orig       as published
        shuffled   factory rows permuted before windowing: destroys the source's
                   temporal dynamics, keeps its marginal and cross-channel
                   statistics
        coarse15   factory data block-averaged to 15 min: a WIDER resolution gap
        domainstd  UAV inputs/target standardised with the UAV training
                   statistics rather than the factory scaler: removes the
                   source-domain level/scale bias at the input
        none       no pretraining (random init), for reference

  C   Communication-efficient FL baselines (Reviewer 1, comment 6), all on T-FL:
        dense f32; int8; top-k int8 (k=0.2, as published); top-k int8 + error
        feedback (k=0.2, 0.1); random-k int8 + error feedback (k=0.2, shared
        PRNG so no indices are sent); adaptive top-k with error feedback (DGC-
        style exponential warm-up of k from 1.0 to 0.1); EF-signSGD (1 bit per
        weight + one fp32 scale per tensor).
      Reports final RMSE, and cumulative uplink bytes and LoRaWAN frames per
      client to first reach a target global RMSE.

Writes results_fl_v5.json (sections are written as they finish).
"""
import copy, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fl_experiment_v3 as v3
import fl_experiment_v4 as v4
from fl_experiment_v3 import (FEATS, SEQ, ROUNDS, LOCAL_EPOCHS, LR, SEEDS, UAV,
                              TinyLSTM, CNN1D, metrics, n_params, predict,
                              train_local, train_full)

torch.set_num_threads(4)
OUT = os.path.join(HERE, "results_fl_v5.json")
SEEDS10 = SEEDS + [5, 6, 7, 8, 9]
HORIZONS = [1, 6, 12, 24]
# NOTE: v3.federated() defaults to quantise=True, so v3's "T-FL" rows were int8
# updates.  v5 always states the operator explicitly.
DENSE = dict(compress_k=1.0, sparsify=False, quantise=False)
USABLE = 220                      # 222 B SF7 app payload minus 2 B fragment header


def save(section, obj):
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res[section] = obj
    json.dump(res, open(OUT, "w"), indent=2, default=float)


def mstd(v):
    v = np.asarray(v, float)
    return dict(mean=float(np.mean(v)), std=float(np.std(v)), n=int(len(v)))


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def seqs(records, mu, sd, tmu, tsd, H):
    arr = (np.array([[r[f] for f in FEATS] for r in records], np.float32) - mu) / sd
    tgt = np.array([r["pm25"] for r in records], np.float32)
    X, y = [], []
    for i in range(len(records) - SEQ - H + 1):
        X.append(arr[i:i + SEQ]); y.append((tgt[i + SEQ + H - 1] - tmu) / tsd)
    if not X:
        return np.empty((0, SEQ, len(FEATS)), np.float32), np.empty((0,), np.float32)
    return np.stack(X), np.array(y, np.float32)


def factory_records(variant="orig", seed=0):
    recs = []
    for _, p in v3.FACTORY.items():
        r = v3.load_factory(p)
        if variant == "coarse15":
            r = [{f: float(np.mean([x[f] for x in r[i:i + 3]])) for f in r[0]}
                 for i in range(0, len(r) - 2, 3)]
        recs += r
    if variant == "shuffled":
        rng = np.random.default_rng(1000 + seed)
        recs = [recs[i] for i in rng.permutation(len(recs))]
    return recs


def build(H=1, variant="orig", seed=0):
    """v3's loading and leakage-safe 70/30 per-zone chronological split, at
    horizon H, with the chosen pretraining-source variant."""
    fac = factory_records(variant, seed)
    fa = np.array([[r[f] for f in FEATS] for r in fac], np.float32)
    mu, sd = fa.mean(0), fa.std(0) + 1e-8
    tmu, tsd = float(fa[:, 0].mean()), float(fa[:, 0].std() + 1e-8)
    Xpre, ypre = seqs(fac, mu, sd, tmu, tsd, H)

    uav = {z: v3.load_uav(p) for z, p in UAV.items()}
    split = {z: (r[:int(0.7 * len(r))], r[int(0.7 * len(r)):]) for z, r in uav.items()}
    umu, usd, utmu, utsd = mu, sd, tmu, tsd
    if variant == "domainstd":
        tr = [x for z in split for x in split[z][0]]
        ua = np.array([[r[f] for f in FEATS] for r in tr], np.float32)
        umu, usd = ua.mean(0), ua.std(0) + 1e-8
        utmu, utsd = float(ua[:, 0].mean()), float(ua[:, 0].std() + 1e-8)
    ctr, pte, tX, tY = {}, {}, [], []
    for z, (a, b) in split.items():
        ctr[z] = seqs(a, umu, usd, utmu, utsd, H)
        pte[z] = seqs(b, umu, usd, utmu, utsd, H)
        tX.append(pte[z][0]); tY.append(pte[z][1])
    return dict(scaler=(umu, usd, utmu, utsd), pretrain=(Xpre, ypre),
                clients_tr=ctr, per_test=pte, pooled=(np.concatenate(tX), np.concatenate(tY)),
                raw_test={z: b for z, (a, b) in split.items()},
                raw_train={z: a for z, (a, b) in split.items()})


def pretrain(mcls, data, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    Xp, yp = data["pretrain"]
    idx = np.random.permutation(len(Xp)); Xp, yp = Xp[idx], yp[idx]
    cut = int(0.9 * len(Xp))
    torch.manual_seed(seed)
    m = mcls(len(FEATS)); train_local(m, Xp[:cut], yp[:cut], epochs=5, lr=1e-3, bs=256)
    return copy.deepcopy(m.state_dict())


def unscale(data, p):
    return p * data["scaler"][3] + data["scaler"][2]


def pooled_rmse_of(preds, ys):
    p = np.concatenate(preds); y = np.concatenate(ys); e = p - y
    return dict(rmse=math.sqrt(np.mean(e ** 2)), mae=float(np.mean(np.abs(e))),
                r2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))


def eval_global(model, data):
    out = {z: dict(zip(("rmse", "mae", "r2"),
                       metrics(predict(model, X), y, data["scaler"][2], data["scaler"][3])))
           for z, (X, y) in data["per_test"].items()}
    out["Global"] = dict(zip(("rmse", "mae", "r2"),
                             metrics(predict(model, data["pooled"][0]), data["pooled"][1],
                                     data["scaler"][2], data["scaler"][3])))
    return out


def local_models(mcls, init, data, seed):
    E = ROUNDS * LOCAL_EPOCHS
    models, preds, ys, per = {}, [], [], {}
    for z, (Xtr, ytr) in data["clients_tr"].items():
        torch.manual_seed(seed)
        m = train_full(mcls, len(FEATS), init, Xtr, ytr, E)
        models[z] = m
        Xte, yte = data["per_test"][z]
        per[z] = dict(zip(("rmse", "mae", "r2"),
                          metrics(predict(m, Xte), yte, data["scaler"][2], data["scaler"][3])))
        preds.append(unscale(data, predict(m, Xte))); ys.append(unscale(data, yte))
    per["Global"] = pooled_rmse_of(preds, ys)
    return models, per


def persistence(data, H):
    """Last observed PM2.5 in the window, held H steps ahead."""
    preds, ys, per = [], [], {}
    mu, sd = data["scaler"][0], data["scaler"][1]
    for z, (X, y) in data["per_test"].items():
        last = X[:, -1, 0] * sd[0] + mu[0]; yy = unscale(data, y)
        e = last - yy
        per[z] = dict(rmse=math.sqrt(np.mean(e ** 2)))
        preds.append(last); ys.append(yy)
    per["Global"] = pooled_rmse_of(preds, ys)
    return per


def ridge_ar(data, alpha=1.0):
    """Per-zone linear autoregression on the same flattened 12 x 5 window."""
    from sklearn.linear_model import Ridge
    preds, ys, per = [], [], {}
    for z, (Xtr, ytr) in data["clients_tr"].items():
        Xte, yte = data["per_test"][z]
        r = Ridge(alpha=alpha).fit(Xtr.reshape(len(Xtr), -1), ytr)
        p = unscale(data, r.predict(Xte.reshape(len(Xte), -1))); yy = unscale(data, yte)
        per[z] = dict(rmse=math.sqrt(np.mean((p - yy) ** 2)))
        preds.append(p); ys.append(yy)
    per["Global"] = pooled_rmse_of(preds, ys)
    return per


# ----------------------------------------------------------------------------
# compression operators with per-client state (error feedback, round index)
# ----------------------------------------------------------------------------
def frames_of(nbytes):
    return max(1, math.ceil(nbytes / USABLE))


class Compressor:
    """name in: dense, int8, topk, topk_ef, randk_ef, adapt_ef, sign_ef."""

    def __init__(self, name, k=1.0, k_final=0.1, warm=0.7, seed=0):
        self.name, self.k, self.k_final, self.warm = name, k, k_final, warm
        self.mem = {}            # client -> residual
        self.rng = np.random.default_rng(seed)

    def k_at(self, rnd):
        if self.name == "adapt_ef":
            return max(self.k_final, self.warm ** rnd)
        return self.k

    def __call__(self, cid, delta, rnd):
        ef = self.name.endswith("_ef")
        if ef:
            mem = self.mem.get(cid) or {k: torch.zeros_like(v) for k, v in delta.items()}
            delta = {k: delta[k] + mem[k] for k in delta}
        nt = len(delta)
        total = sum(v.numel() for v in delta.values())

        if self.name == "dense":
            out, nbytes = delta, total * 4
        elif self.name == "int8":
            out, nbytes = v3.compress_delta(delta, 1.0, sparsify=False, quantise=True)
        elif self.name in ("topk", "topk_ef", "adapt_ef"):
            out, nbytes = v3.compress_delta(delta, self.k_at(rnd), sparsify=True, quantise=True)
        elif self.name == "randk_ef":
            keep = max(1, int(self.k * total))
            sel = torch.zeros(total, dtype=torch.bool)
            sel[torch.as_tensor(self.rng.choice(total, keep, replace=False))] = True
            out, pos = {}, 0
            for name, v in delta.items():
                m = sel[pos:pos + v.numel()].view_as(v); pos += v.numel()
                vals = v[m]
                o = torch.zeros_like(v)
                if vals.numel():
                    s = vals.abs().max() / 127.0 + 1e-12
                    o[m] = torch.round(vals / s).clamp(-127, 127) * s
                out[name] = o
            nbytes = keep * 1 + nt * 4 + 4          # values + scales + shared PRNG seed
        elif self.name == "sign_ef":
            out = {k: torch.sign(v) * v.abs().mean() for k, v in delta.items()}
            nbytes = math.ceil(total / 8) + nt * 4
        else:
            raise ValueError(self.name)

        if ef:
            self.mem[cid] = {k: delta[k] - out[k] for k in delta}
        return out, nbytes


def federated_c(mcls, init, data, comp, seed):
    torch.manual_seed(seed)
    glob = mcls(len(FEATS)); glob.load_state_dict(copy.deepcopy(init))
    hist, bytes_r, frames_r = [], [], []
    for rnd in range(ROUNDS):
        g = copy.deepcopy(glob.state_dict())
        deltas, sizes, b, f = [], [], [], []
        for cid, (Xtr, ytr) in data["clients_tr"].items():
            loc = mcls(len(FEATS)); loc.load_state_dict(copy.deepcopy(g))
            train_local(loc, Xtr, ytr, LOCAL_EPOCHS, LR)
            d = {k: loc.state_dict()[k] - g[k] for k in g}
            cd, nb = comp(cid, d, rnd)
            deltas.append(cd); sizes.append(len(Xtr)); b.append(nb); f.append(frames_of(nb))
        w = np.array(sizes, float); w /= w.sum()
        glob.load_state_dict({k: g[k] + sum(deltas[i][k] * float(w[i]) for i in range(len(deltas)))
                              for k in g})
        hist.append(eval_global(glob, data)["Global"]["rmse"])
        bytes_r.append(float(np.mean(b))); frames_r.append(float(np.mean(f)))
    return glob, hist, bytes_r, frames_r


def cost_to_target(hist, bytes_r, frames_r, target):
    for i, h in enumerate(hist):
        if h <= target:
            return dict(round=i + 1, bytes=float(np.sum(bytes_r[:i + 1])),
                        frames=float(np.sum(frames_r[:i + 1])))
    return dict(round=None, bytes=None, frames=None)


# ----------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------
def section_H():
    out = {}
    for H in HORIZONS:
        t0 = time.time()
        data = build(H=H)
        rows = dict(persistence=persistence(data, H), ridge_ar=ridge_ar(data),
                    windows=dict(train={z: int(len(v[0])) for z, v in data["clients_tr"].items()},
                                 test={z: int(len(v[0])) for z, v in data["per_test"].items()}))
        acc = {k: [] for k in ("T-Local", "T-FL", "Local-random")}
        zon = {k: {z: [] for z in UAV} for k in acc}
        for s in SEEDS:
            pre = pretrain(TinyLSTM, data, s)
            _, tl = local_models(TinyLSTM, pre, data, s)
            _, lr_ = local_models(TinyLSTM, None, data, s)
            torch.manual_seed(s)
            g, _, _ = v3.federated(TinyLSTM, len(FEATS), pre, data["clients_tr"], data["pooled"],
                                   data["scaler"][2], data["scaler"][3], **DENSE)
            tf = eval_global(g, data)
            for k, r in (("T-Local", tl), ("T-FL", tf), ("Local-random", lr_)):
                acc[k].append(r["Global"]["rmse"])
                for z in UAV:
                    zon[k][z].append(r[z]["rmse"])
        for k in acc:
            rows[k] = dict(Global=mstd(acc[k]), per_zone={z: mstd(zon[k][z]) for z in UAV},
                           per_seed=acc[k])
        out[f"H={H}"] = rows
        print(f"  H={H:2d} ({5*H:3d} s)  persist {rows['persistence']['Global']['rmse']:.2f}  "
              f"ridge {rows['ridge_ar']['Global']['rmse']:.2f}  "
              f"T-Local {rows['T-Local']['Global']['mean']:.2f}  "
              f"T-FL {rows['T-FL']['Global']['mean']:.2f}+-{rows['T-FL']['Global']['std']:.2f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        save("horizon", out)
    return out


def section_F():
    from scipy import stats
    data = build(H=1)
    cells = {k: [] for k in ("Local-random", "T-Local", "T-FL", "T-FL-int8")}
    zones = {k: {z: [] for z in UAV} for k in cells}
    newsite = {z: dict(fed2=[], best_single=[], mean_single=[]) for z in UAV}
    upl = None
    for s in SEEDS10:
        t0 = time.time()
        pre = pretrain(TinyLSTM, data, s)
        _, lr_ = local_models(TinyLSTM, None, data, s)
        tl_models, tl = local_models(TinyLSTM, pre, data, s)
        torch.manual_seed(s)
        gd, _, upl = v3.federated(TinyLSTM, len(FEATS), pre, data["clients_tr"], data["pooled"],
                                  data["scaler"][2], data["scaler"][3], **DENSE)
        torch.manual_seed(s)
        gq, _, _ = v3.federated(TinyLSTM, len(FEATS), pre, data["clients_tr"], data["pooled"],
                                data["scaler"][2], data["scaler"][3],
                                compress_k=1.0, sparsify=False, quantise=True)
        for k, r in (("Local-random", lr_), ("T-Local", tl), ("T-FL", eval_global(gd, data)),
                     ("T-FL-int8", eval_global(gq, data))):
            cells[k].append(r["Global"]["rmse"])
            for z in UAV:
                zones[k][z].append(r[z]["rmse"])
        # new site: two-zone federation vs a single other zone's local model
        for held in UAV:
            seen = {z: v for z, v in data["clients_tr"].items() if z != held}
            Xte, yte = data["per_test"][held]
            torch.manual_seed(s)
            g2, _, _ = v3.federated(TinyLSTM, len(FEATS), pre, seen, (Xte, yte),
                                    data["scaler"][2], data["scaler"][3], **DENSE)
            r_fed = metrics(predict(g2, Xte), yte, data["scaler"][2], data["scaler"][3])[0]
            r_single = [metrics(predict(tl_models[z], Xte), yte, data["scaler"][2],
                                data["scaler"][3])[0] for z in seen]
            newsite[held]["fed2"].append(r_fed)
            newsite[held]["best_single"].append(min(r_single))
            newsite[held]["mean_single"].append(float(np.mean(r_single)))
        print(f"  seed {s}: T-Local {cells['T-Local'][-1]:.2f}  T-FL {cells['T-FL'][-1]:.2f}  "
              f"int8 {cells['T-FL-int8'][-1]:.3f}  [{time.time()-t0:.0f}s]", flush=True)

    def paired(a, b):
        a, b = np.array(a), np.array(b); d = a - b
        w = stats.wilcoxon(a, b) if np.any(d != 0) else None
        t = stats.ttest_rel(a, b)
        ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d)) \
            if d.std() > 0 else (float(d.mean()), float(d.mean()))
        return dict(diff_mean=float(d.mean()), diff_std=float(d.std(ddof=1)),
                    ci95=[float(ci[0]), float(ci[1])], wins=int(np.sum(d > 0)), n=len(d),
                    t_p=float(t.pvalue), wilcoxon_p=(float(w.pvalue) if w else None),
                    max_abs_diff=float(np.max(np.abs(d))))

    out = dict(seeds=SEEDS10,
               cells={k: dict(Global=mstd(v), per_zone={z: mstd(zones[k][z]) for z in UAV},
                              per_seed=v) for k, v in cells.items()},
               paired=dict(
                   transfer_gain_local=paired(cells["Local-random"], cells["T-Local"]),
                   federation_gain=paired(cells["T-Local"], cells["T-FL"]),
                   federation_gain_per_zone={z: paired(zones["T-Local"][z], zones["T-FL"][z])
                                             for z in UAV},
                   int8_cost=paired(cells["T-FL-int8"], cells["T-FL"])),
               uplink_bytes_per_round_dense=upl,
               newsite={z: {k: mstd(v) for k, v in d.items()} | {"wins_fed": int(
                   np.sum(np.array(d["fed2"]) < np.array(d["best_single"])))}
                        for z, d in newsite.items()})
    save("federation_value", out)
    return out


def section_P():
    out = {}
    for variant in ("orig", "shuffled", "coarse15", "domainstd", "none"):
        t0 = time.time()
        tl, tf = [], []
        for s in SEEDS:
            data = build(H=1, variant=("orig" if variant == "none" else variant), seed=s)
            pre = None if variant == "none" else pretrain(TinyLSTM, data, s)
            _, l = local_models(TinyLSTM, pre, data, s)
            torch.manual_seed(s)
            g, _, _ = v3.federated(TinyLSTM, len(FEATS), pre, data["clients_tr"], data["pooled"],
                                   data["scaler"][2], data["scaler"][3], **DENSE)
            tl.append(l["Global"]["rmse"]); tf.append(eval_global(g, data)["Global"]["rmse"])
        out[variant] = dict(local=mstd(tl), federated=mstd(tf), local_per_seed=tl, fed_per_seed=tf)
        print(f"  {variant:10s} local {np.mean(tl):.2f}+-{np.std(tl):.2f}  "
              f"fed {np.mean(tf):.2f}+-{np.std(tf):.2f}  [{time.time()-t0:.0f}s]", flush=True)
        save("pretrain_ablation", out)
    return out


COMP_CFGS = [
    ("dense f32", dict(name="dense")),
    ("int8", dict(name="int8")),
    ("top-k int8 k=0.2", dict(name="topk", k=0.2)),
    ("top-k int8 k=0.1", dict(name="topk", k=0.1)),
    ("top-k int8+EF k=0.2", dict(name="topk_ef", k=0.2)),
    ("top-k int8+EF k=0.1", dict(name="topk_ef", k=0.1)),
    ("rand-k int8+EF k=0.2", dict(name="randk_ef", k=0.2)),
    ("adaptive top-k+EF 1.0->0.1", dict(name="adapt_ef", k_final=0.1, warm=0.7)),
    ("EF-signSGD", dict(name="sign_ef")),
]
TARGETS = [7.5, 7.0, 6.8]


def section_C():
    data = build(H=1)
    res = {lab: dict(final=[], hist=[], cost={str(t): [] for t in TARGETS}, bytes_round=None)
           for lab, _ in COMP_CFGS}
    for s in SEEDS:
        t0 = time.time()
        pre = pretrain(TinyLSTM, data, s)
        for lab, cfg in COMP_CFGS:
            comp = Compressor(seed=s, **cfg)
            _, hist, b, f = federated_c(TinyLSTM, pre, data, comp, s)
            r = res[lab]
            r["final"].append(hist[-1]); r["hist"].append(hist)
            r["bytes_round"] = b; r["frames_round"] = f
            for t in TARGETS:
                r["cost"][str(t)].append(cost_to_target(hist, b, f, t))
        print(f"  seed {s} [{time.time()-t0:.0f}s]", flush=True)

    out = {}
    for lab, r in res.items():
        cost = {}
        for t, lst in r["cost"].items():
            hit = [c for c in lst if c["round"] is not None]
            cost[t] = dict(reached=f"{len(hit)}/{len(lst)}",
                           rounds=mstd([c["round"] for c in hit]) if hit else None,
                           bytes=mstd([c["bytes"] for c in hit]) if hit else None,
                           frames=mstd([c["frames"] for c in hit]) if hit else None)
        out[lab] = dict(final_rmse=mstd(r["final"]),
                        mean_curve=np.mean(np.array(r["hist"]), 0).tolist(),
                        bytes_per_round_mean=float(np.mean(r["bytes_round"])),
                        frames_per_round_mean=float(np.mean(r["frames_round"])),
                        total_bytes_30=float(np.sum(r["bytes_round"])),
                        total_frames_30=float(np.sum(r["frames_round"])),
                        to_target=cost, final_per_seed=r["final"])
        c = out[lab]["to_target"]["7.0"]
        print(f"  {lab:28s} RMSE {np.mean(r['final']):.2f}+-{np.std(r['final']):.2f}  "
              f"{out[lab]['bytes_per_round_mean']:7.0f} B/rnd  "
              f"{out[lab]['frames_per_round_mean']:5.1f} fr/rnd  "
              f"to 7.0: {c['reached']} frames {c['frames']['mean'] if c['frames'] else '-'}",
              flush=True)
    save("compression_baselines", dict(targets=TARGETS, configs=out))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["F", "C", "P", "H"]
    for w in which:
        print(f"=== section {w} ===", flush=True)
        dict(H=section_H, F=section_F, P=section_P, C=section_C)[w]()
    print("Saved ->", OUT)
