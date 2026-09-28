"""
FL experiment v4 for the UAV industrial PM2.5 paper (IEEE IoT-J R1 revision).

Imports fl_experiment_v3 unchanged (same data loading, models, training, FedAvg,
seeds) and adds ONLY what the IoT-J reviewers asked for.  Nothing in v3 is
modified, so every headline number in the submitted manuscript stays
reproducible from v3; v4 either refines a communication figure with a stated
extra term or adds a new measurement.

Reviewer 1
  A1  Frame count with application-level fragmentation metadata.  v3 divided the
      payload by the full 222 B SF7 application payload.  A real fragmented
      transfer must also carry a round ID and a fragment sequence number in each
      frame, so only 220 B per frame are usable for model data.  Changes the
      dense f32 TinyLSTM update from 27 to 28 frames (all compressed rows are
      unchanged) and lowers its uplink capacity accordingly.

  A2  Downlink / gateway duty-cycle bound.  v3 reported only single-node uplink
      capacity.  A federated round also needs the gateway to send the new global
      model to every node.  Adds an analytical downlink bound for both Class A
      receive windows -- RX1 (uplink channel, SF7, ETSI g1 sub-band, 1 % duty
      cycle) and RX2 (869.525 MHz, ETSI g3 sub-band, 10 % duty cycle, at the
      SF9/DR3 rate public networks use and at the SF12/DR0 regional default) --
      and reports the end-to-end sustainable rate as the minimum of the two
      sides, which falls as 1/N in the number of nodes.

Reviewer 2
  B1  Heterogeneity at the model level.  The clients share one architecture
      because FedAvg averages like-shaped tensors, so heterogeneity can only
      appear in the updates.  Measures it: pairwise cosine similarity and
      relative L2 divergence between the three raw client updates in every
      round, against an IID control that re-partitions the identical pooled
      windows at random into three clients of the same sizes.

  B2  A model-level heterogeneity MECHANISM.  Adds T-FL-Rep: the recurrent /
      convolutional representation is federated while each client keeps a private
      read-out head, so the deployed models differ per node.  Also shrinks the
      uplink because the head is never transmitted.

  B3  Adaptability to an unseen environment.  Leave-one-zone-out: train T-FL on
      two zones and evaluate on the held-out zone's test windows, zero-shot, then
      after 1 and 5 rounds' worth of local adaptation on that zone's training
      windows.  Bracketed by persistence and by the all-three-zones T-FL.

  B4  Environment parameter characterisation.  Per-zone campaign parameters
      (clock time, duration, samples, PM2.5 level/spread, temperature, humidity)
      dumped straight from the quality-controlled logs for the parameter table.

Writes results_fl_v4.json.  New learning experiments run on the TinyLSTM (the
model every detailed result in the paper uses) and, where cheap, the 1D-CNN.
"""
import copy, csv, json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import fl_experiment_v3 as v3
from fl_experiment_v3 import (FEATS, ROOT, ROUNDS, LOCAL_EPOCHS, LR, SEEDS, UAV,
                              TinyLSTM, CNN1D, lorawan_airtime, make_sequences,
                              metrics, n_params, predict, train_local, train_full)

# ----------------------------------------------------------------------------
# A1/A2  LoRaWAN accounting
# ----------------------------------------------------------------------------
FRAG_HDR_B = 2          # round ID + fragment sequence number, per frame
N_NODES = 3             # clients in this study

# EU868 Class A windows.  max_app = application payload ceiling at that data
# rate; duty = ETSI duty cycle of the sub-band the gateway transmits in.
DOWNLINK_WINDOWS = {
    "RX1_SF7":  dict(sf=7,  max_app=222, duty=0.01,
                     note="uplink channel, DR5/SF7, ETSI g1 sub-band"),
    "RX2_SF9":  dict(sf=9,  max_app=115, duty=0.10,
                     note="869.525 MHz, DR3/SF9 as used by public networks, ETSI g3"),
    "RX2_SF12": dict(sf=12, max_app=51,  duty=0.10,
                     note="869.525 MHz, DR0/SF12 regional-parameter default, ETSI g3"),
}


def frames_and_airtime(payload_bytes, sf, max_app, frag_hdr=FRAG_HDR_B):
    """Fragment a payload into frames that each carry frag_hdr bytes of
    application control metadata, and total the time-on-air.

    Keeps v3's conservative convention that a frame belonging to a multi-frame
    transfer is charged at the full max_app airtime; only a transfer that fits
    in one frame is charged for its own length."""
    usable = max_app - frag_hdr
    frames = max(1, math.ceil(payload_bytes / usable))
    on_wire = min(payload_bytes + frames * frag_hdr, max_app)
    toa = lorawan_airtime(on_wire, sf=sf)
    return frames, frames * toa, toa


def uplink_budget(payload_bytes, sf=7, max_app=222, duty=0.01, frag_hdr=FRAG_HDR_B):
    """Maximum single-node uplink capacity: what one node's duty cycle allows.
    This is the quantity v3 called rounds_per_day; it is an upper bound on one
    node's uplink and NOT a count of sustainable complete federated rounds."""
    frames, airtime, toa = frames_and_airtime(payload_bytes, sf, max_app, frag_hdr)
    return dict(payload_bytes=round(payload_bytes, 1), frames=frames,
                airtime_per_frame_s=round(toa, 3),
                airtime_update_s=round(airtime, 3),
                min_cycle_s=round(airtime / duty, 1),
                max_uplink_rounds_per_day=round(86400.0 * duty / airtime, 1))


def downlink_budget(payload_bytes, n_nodes=N_NODES, window="RX2_SF9",
                    frag_hdr=FRAG_HDR_B, multicast=False):
    """Gateway-side bound.  Every round the server must deliver the new global
    model to each node, so unicast airtime scales with n_nodes; a Class B/C
    multicast group would make it independent of n_nodes."""
    w = DOWNLINK_WINDOWS[window]
    frames, airtime_node, toa = frames_and_airtime(payload_bytes, w["sf"],
                                                   w["max_app"], frag_hdr)
    airtime_round = airtime_node if multicast else airtime_node * n_nodes
    return dict(window=window, note=w["note"], sf=w["sf"], max_app_payload=w["max_app"],
                duty=w["duty"], n_nodes=(1 if multicast else n_nodes),
                multicast=multicast, frames=frames,
                airtime_per_frame_s=round(toa, 3),
                airtime_per_node_s=round(airtime_node, 3),
                airtime_per_round_s=round(airtime_round, 3),
                max_downlink_rounds_per_day=round(86400.0 * w["duty"] / airtime_round, 1))


def end_to_end_budget(payload_bytes, n_nodes=N_NODES, window="RX2_SF9",
                      multicast=False):
    """Sustainable complete federated rounds per day = the tighter of the two
    sides.  Uplink is per node and independent of n_nodes; unicast downlink
    falls as 1/n_nodes."""
    up = uplink_budget(payload_bytes)
    dn = downlink_budget(payload_bytes, n_nodes=n_nodes, window=window,
                         multicast=multicast)
    rounds = min(up["max_uplink_rounds_per_day"], dn["max_downlink_rounds_per_day"])
    return dict(uplink=up, downlink=dn,
                sustainable_rounds_per_day=round(rounds, 1),
                binding_side=("uplink"
                              if up["max_uplink_rounds_per_day"] <= dn["max_downlink_rounds_per_day"]
                              else "downlink"))


def crossover_nodes(payload_bytes, window="RX2_SF9"):
    """Fleet size at which the unicast gateway downlink overtakes the single-node
    uplink as the binding constraint."""
    up = uplink_budget(payload_bytes)["max_uplink_rounds_per_day"]
    one = downlink_budget(payload_bytes, n_nodes=1, window=window)
    return round(one["max_downlink_rounds_per_day"] / up, 2)


# ----------------------------------------------------------------------------
# B1  client drift, and B2 the per-client-head variant
# ----------------------------------------------------------------------------
def _flat(delta):
    return torch.cat([v.flatten() for v in delta.values()]).double()


def _drift_stats(deltas, w):
    """Pairwise cosine similarity between raw client updates, and each client's
    relative L2 distance from the aggregate."""
    vs = [_flat(d) for d in deltas]
    mean = sum(float(w[i]) * vs[i] for i in range(len(vs)))
    cos = []
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            denom = vs[i].norm() * vs[j].norm()
            cos.append(float(torch.dot(vs[i], vs[j]) / denom) if denom > 0 else float("nan"))
    mnorm = float(mean.norm())
    drift = [float((vs[i] - mean).norm()) / mnorm if mnorm > 0 else float("nan")
             for i in range(len(vs))]
    return cos, drift


def federated_instrumented(model_cls, nf, init_state, clients, test, tgt_mu, tgt_sd,
                           prox_mu=0.0):
    """v3.federated with no compression, plus per-round client-drift telemetry."""
    glob = model_cls(nf)
    if init_state is not None:
        glob.load_state_dict(copy.deepcopy(init_state))
    cos_rounds, drift_rounds, history = [], [], []
    for _ in range(ROUNDS):
        gstate = copy.deepcopy(glob.state_dict())
        deltas, sizes = [], []
        for z, (Xtr, ytr) in clients.items():
            local = model_cls(nf)
            local.load_state_dict(copy.deepcopy(gstate))
            train_local(local, Xtr, ytr, LOCAL_EPOCHS, LR,
                        global_state=gstate, prox_mu=prox_mu)
            deltas.append({k: local.state_dict()[k] - gstate[k] for k in gstate})
            sizes.append(len(Xtr))
        w = np.array(sizes, float); w = w / w.sum()
        cos, drift = _drift_stats(deltas, w)
        cos_rounds.append(cos); drift_rounds.append(drift)
        newstate = {k: gstate[k] + sum(deltas[i][k] * float(w[i]) for i in range(len(deltas)))
                    for k in gstate}
        glob.load_state_dict(newstate)
        history.append(metrics(predict(glob, test[0]), test[1], tgt_mu, tgt_sd)[0])
    cos_all = np.array(cos_rounds, float); drift_all = np.array(drift_rounds, float)
    return glob, history, dict(
        cos_mean=float(np.nanmean(cos_all)), cos_std=float(np.nanstd(cos_all)),
        cos_first_round=float(np.nanmean(cos_all[0])),
        cos_last_round=float(np.nanmean(cos_all[-1])),
        drift_mean=float(np.nanmean(drift_all)), drift_std=float(np.nanstd(drift_all)),
        drift_per_client_mean=[float(np.nanmean(drift_all[:, i]))
                               for i in range(drift_all.shape[1])])


def iid_repartition(clients_tr, seed):
    """Control: pool every training window and deal them back out at random into
    three clients holding exactly the same counts.  Identical data, identical
    client sizes, heterogeneity removed."""
    rng = np.random.default_rng(seed)
    names = list(clients_tr.keys())
    X = np.concatenate([clients_tr[z][0] for z in names])
    y = np.concatenate([clients_tr[z][1] for z in names])
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    out, start = {}, 0
    for z in names:
        n = len(clients_tr[z][0])
        out[z] = (X[start:start + n], y[start:start + n])
        start += n
    return out


def _is_head(key):
    return key.startswith("fc.")


def federated_rep(model_cls, nf, init_state, clients, per_test, tgt_mu, tgt_sd):
    """B2: federate the representation, keep a private read-out head on every
    client.  The deployed models therefore differ per node, which is
    heterogeneity expressed at the model level rather than only in the data.
    Uplink carries the shared tensors alone."""
    glob = model_cls(nf)
    if init_state is not None:
        glob.load_state_dict(copy.deepcopy(init_state))
    g0 = glob.state_dict()
    heads = {z: {k: v.clone() for k, v in g0.items() if _is_head(k)} for z in clients}
    shared_keys = [k for k in g0 if not _is_head(k)]
    uplink_vals = sum(int(g0[k].numel()) for k in shared_keys)
    for _ in range(ROUNDS):
        gstate = copy.deepcopy(glob.state_dict())
        deltas, sizes = [], []
        for z, (Xtr, ytr) in clients.items():
            local = model_cls(nf)
            st = copy.deepcopy(gstate); st.update(heads[z])
            local.load_state_dict(st)
            train_local(local, Xtr, ytr, LOCAL_EPOCHS, LR)
            ls = local.state_dict()
            heads[z] = {k: ls[k].detach().clone() for k in heads[z]}
            deltas.append({k: ls[k] - gstate[k] for k in shared_keys})
            sizes.append(len(Xtr))
        w = np.array(sizes, float); w = w / w.sum()
        newstate = copy.deepcopy(gstate)
        for k in shared_keys:
            newstate[k] = gstate[k] + sum(deltas[i][k] * float(w[i]) for i in range(len(deltas)))
        glob.load_state_dict(newstate)

    # evaluate each client with its own head; pool the predictions so the number
    # is directly comparable with the single-model pooled RMSE used elsewhere
    per_zone, preds, ys = {}, [], []
    for z, (Xte, yte) in per_test.items():
        local = model_cls(nf)
        st = copy.deepcopy(glob.state_dict()); st.update(heads[z])
        local.load_state_dict(st)
        p = predict(local, Xte)
        per_zone[z] = dict(zip(("rmse", "mae", "r2"), metrics(p, yte, tgt_mu, tgt_sd)))
        preds.append(p * tgt_sd + tgt_mu); ys.append(yte * tgt_sd + tgt_mu)
    preds = np.concatenate(preds); ys = np.concatenate(ys); err = preds - ys
    per_zone["Global"] = dict(rmse=math.sqrt(np.mean(err ** 2)),
                              mae=float(np.mean(np.abs(err))),
                              r2=float(1 - np.sum(err ** 2) / np.sum((ys - ys.mean()) ** 2)))
    per_zone["uplink_bytes_per_round_f32"] = uplink_vals * 4
    per_zone["uplink_bytes_per_round_int8"] = uplink_vals + 4 * len(shared_keys)
    return per_zone


# ----------------------------------------------------------------------------
# B3  leave-one-zone-out adaptability
# ----------------------------------------------------------------------------
ADAPT_ROUNDS = [1, 5]


def leave_one_zone_out(model_cls, nf, pre_state, clients_tr, per_test,
                       tgt_mu, tgt_sd, seed):
    """Hold one zone out of the federation entirely, then ask the resulting
    global model to forecast it.  Zero-shot measures transfer to an unseen
    industrial environment; the adaptation rows measure how quickly it closes
    the gap using only that zone's own training window."""
    out = {}
    for held in clients_tr:
        seen = {z: v for z, v in clients_tr.items() if z != held}
        Xte, yte = per_test[held]
        torch.manual_seed(seed)
        g, _, _ = v3.federated(model_cls, nf, pre_state, seen,
                               (Xte, yte), tgt_mu, tgt_sd)
        row = {"zero_shot": dict(zip(("rmse", "mae", "r2"),
                                     metrics(predict(g, Xte), yte, tgt_mu, tgt_sd)))}
        gstate = copy.deepcopy(g.state_dict())
        Xtr, ytr = clients_tr[held]
        for r in ADAPT_ROUNDS:
            m = model_cls(nf); m.load_state_dict(copy.deepcopy(gstate))
            torch.manual_seed(seed)
            train_local(m, Xtr, ytr, LOCAL_EPOCHS * r, LR)
            row[f"adapt_{r}_rounds"] = dict(zip(("rmse", "mae", "r2"),
                                                metrics(predict(m, Xte), yte, tgt_mu, tgt_sd)))
        out[held] = row
    return out


# ----------------------------------------------------------------------------
# B4  per-zone environment parameters, from the quality-controlled logs
# ----------------------------------------------------------------------------
def zone_parameters():
    rows = {}
    for z, path in UAV.items():
        recs = list(csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8-sig")))
        keep = []
        for r in recs:
            try:
                lat = float(r["Latitude"]); lon = float(r["Longitude"])
                alt = float(r["Altitude_m"]); spd = float(r["Speed_mps"])
                rec = dict(pm25=float(r["PM2.5"]), pm10=float(r["PM10"]),
                           co2=float(r["CO2"]), temp=float(r["Temp_C"]),
                           rh=float(r["Humidity_%"]), date=r["Date"], time=r["Time"])
            except (ValueError, KeyError):
                continue
            if not (v3.LAT[0] < lat < v3.LAT[1] and v3.LON[0] < lon < v3.LON[1]):
                continue
            if alt > 2000 or spd > 20:
                continue
            keep.append(rec)
        pm = np.array([r["pm25"] for r in keep], float)
        tt = np.array([r["temp"] for r in keep], float)
        rh = np.array([r["rh"] for r in keep], float)

        def secs(t):
            h, m, s = (int(x) for x in t.split(":"))
            return h * 3600 + m * 60 + s
        t0, t1 = keep[0]["time"], keep[-1]["time"]
        rows[z] = dict(date=keep[0]["date"], start=t0, end=t1,
                       duration_min=round((secs(t1) - secs(t0)) / 60.0, 1),
                       samples=len(keep),
                       pm25_mean=round(float(pm.mean()), 1),
                       pm25_median=round(float(np.median(pm)), 1),
                       pm25_cv=round(float(pm.std() / pm.mean()), 2),
                       pm25_min=int(pm.min()), pm25_max=int(pm.max()),
                       temp_mean=round(float(tt.mean()), 1),
                       temp_min=round(float(tt.min()), 1), temp_max=round(float(tt.max()), 1),
                       rh_mean=round(float(rh.mean()), 1),
                       rh_min=round(float(rh.min()), 1), rh_max=round(float(rh.max()), 1))
    return rows


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def build_data():
    """Byte-for-byte the loading and leakage-safe split of v3.main()."""
    fac = []
    for name, p in v3.FACTORY.items():
        fac += v3.load_factory(p)
    fa = np.array([[r[f] for f in FEATS] for r in fac], np.float32)
    mu, sd = fa.mean(0), fa.std(0) + 1e-8
    tgt_mu = float(fa[:, 0].mean()); tgt_sd = float(fa[:, 0].std() + 1e-8)
    Xpre, ypre = make_sequences(fac, mu, sd, tgt_mu, tgt_sd)

    uav = {z: v3.load_uav(p) for z, p in UAV.items()}
    clients_tr, per_test, tX, tY = {}, {}, [], []
    for z, recs in uav.items():
        crec = int(0.7 * len(recs))
        Xtr, ytr = make_sequences(recs[:crec], mu, sd, tgt_mu, tgt_sd)
        Xte, yte = make_sequences(recs[crec:], mu, sd, tgt_mu, tgt_sd)
        clients_tr[z] = (Xtr, ytr); per_test[z] = (Xte, yte)
        tX.append(Xte); tY.append(yte)
    return dict(scaler=(mu, sd, tgt_mu, tgt_sd), pretrain=(Xpre, ypre),
                clients_tr=clients_tr, per_test=per_test,
                pooled_test=(np.concatenate(tX), np.concatenate(tY)))


def persistence_per_zone(data):
    mu, sd, tgt_mu, tgt_sd = data["scaler"]
    out = {}
    for z, (X, y) in data["per_test"].items():
        last = X[:, -1, 0] * sd[0] + mu[0]; yy = y * tgt_sd + tgt_mu; e = last - yy
        out[z] = dict(rmse=math.sqrt(np.mean(e ** 2)), mae=float(np.mean(np.abs(e))),
                      r2=float(1 - np.sum(e ** 2) / np.sum((yy - yy.mean()) ** 2)))
    return out


def run_learning(data, models=("TinyLSTM", "1D-CNN")):
    mu, sd, tgt_mu, tgt_sd = data["scaler"]
    Xpre, ypre = data["pretrain"]
    clients_tr, per_test, pooled = data["clients_tr"], data["per_test"], data["pooled_test"]
    nf = len(FEATS)
    cls = {"TinyLSTM": TinyLSTM, "1D-CNN": CNN1D}
    per_seed = []

    for seed in SEEDS:
        print(f"  seed {seed} ...", flush=True)
        np.random.seed(seed); torch.manual_seed(seed)
        idx = np.random.permutation(len(Xpre))
        Xp, yp = Xpre[idx], ypre[idx]
        cut = int(0.9 * len(Xp)); Xtr_p, ytr_p = Xp[:cut], yp[:cut]

        seed_out = {}
        for mname in models:
            mcls = cls[mname]
            torch.manual_seed(seed)
            pre = mcls(nf); train_local(pre, Xtr_p, ytr_p, epochs=5, lr=1e-3, bs=256)
            pre_state = copy.deepcopy(pre.state_dict())
            block = {"params": n_params(pre)}

            # B1 -- drift on the real non-IID partition, and on the IID control
            torch.manual_seed(seed)
            _, _, dr_noniid = federated_instrumented(mcls, nf, pre_state, clients_tr,
                                                     pooled, tgt_mu, tgt_sd)
            iid = iid_repartition(clients_tr, seed)
            torch.manual_seed(seed)
            _, _, dr_iid = federated_instrumented(mcls, nf, pre_state, iid,
                                                  pooled, tgt_mu, tgt_sd)
            block["drift_noniid"] = dr_noniid
            block["drift_iid"] = dr_iid

            # B2 -- shared representation + private per-client head
            torch.manual_seed(seed)
            block["T-FL-Rep"] = federated_rep(mcls, nf, pre_state, clients_tr,
                                              per_test, tgt_mu, tgt_sd)

            # per-zone T-FL on all three zones, for the LOZO comparison
            torch.manual_seed(seed)
            gt, _, _ = v3.federated(mcls, nf, pre_state, clients_tr, pooled, tgt_mu, tgt_sd)
            block["T-FL_seen"] = {z: dict(zip(("rmse", "mae", "r2"),
                                              metrics(predict(gt, X), y, tgt_mu, tgt_sd)))
                                  for z, (X, y) in per_test.items()}

            # B3 -- leave-one-zone-out
            block["lozo"] = leave_one_zone_out(mcls, nf, pre_state, clients_tr,
                                               per_test, tgt_mu, tgt_sd, seed)
            seed_out[mname] = block
        per_seed.append(seed_out)
    return per_seed


def aggregate(per_seed, models):
    def mstd(vals):
        return dict(mean=float(np.mean(vals)), std=float(np.std(vals)))
    summary = {}
    for mname in models:
        blocks = [s[mname] for s in per_seed]
        m = {"params": blocks[0]["params"]}
        for key in ("drift_noniid", "drift_iid"):
            m[key] = {f: mstd([b[key][f] for b in blocks])
                      for f in ("cos_mean", "cos_first_round", "cos_last_round", "drift_mean")}
        m["T-FL-Rep"] = {z: {f: mstd([b["T-FL-Rep"][z][f] for b in blocks])
                             for f in ("rmse", "mae", "r2")}
                         for z in list(blocks[0]["T-FL-Rep"]) if z.startswith(("Zone", "Global"))}
        m["T-FL-Rep"]["uplink_bytes_per_round_f32"] = blocks[0]["T-FL-Rep"]["uplink_bytes_per_round_f32"]
        m["T-FL-Rep"]["uplink_bytes_per_round_int8"] = blocks[0]["T-FL-Rep"]["uplink_bytes_per_round_int8"]
        m["T-FL_seen"] = {z: mstd([b["T-FL_seen"][z]["rmse"] for b in blocks])
                          for z in blocks[0]["T-FL_seen"]}
        zones = list(blocks[0]["lozo"].keys())
        rowkeys = ["zero_shot"] + [f"adapt_{r}_rounds" for r in ADAPT_ROUNDS]
        m["lozo"] = {z: {rk: {f: mstd([b["lozo"][z][rk][f] for b in blocks])
                              for f in ("rmse", "mae", "r2")}
                         for rk in rowkeys} for z in zones}
        summary[mname] = m
    return summary


def comms_tables():
    """Payloads are taken from the v3 factorial so the accounting change is the
    only difference from the submitted Table 3."""
    v3res = json.load(open(os.path.join(HERE, "results_fl_v3.json")))
    fac = v3res["summary"]["models"]["TinyLSTM"]["sparse_quant_factorial"]
    rows = {}
    for key, rec in fac.items():
        pb = rec["uplink_bytes_per_round"]
        rows[key] = dict(
            rmse_mean=rec["rmse_mean"],
            payload_bytes=pb,
            v3_frames=rec["lorawan"]["frames"],
            v3_rounds_per_day=rec["lorawan"]["rounds_per_day"],
            uplink=uplink_budget(pb),
            downlink={w: downlink_budget(pb, window=w) for w in DOWNLINK_WINDOWS},
            end_to_end_rx2_sf9=end_to_end_budget(pb, window="RX2_SF9"),
            crossover_nodes_rx2_sf9=crossover_nodes(pb, "RX2_SF9"),
        )
    scaling = {}
    for key in ("dense_f32", "quant_only", "both_k=0.4"):
        pb = fac[key]["uplink_bytes_per_round"]
        scaling[key] = {str(n): end_to_end_budget(pb, n_nodes=n, window="RX2_SF9")
                        ["sustainable_rounds_per_day"] for n in (3, 5, 10, 20, 50)}
        scaling[key]["multicast"] = end_to_end_budget(
            pb, window="RX2_SF9", multicast=True)["sustainable_rounds_per_day"]
    return dict(rows=rows, fleet_scaling_rx2_sf9=scaling, frag_hdr_bytes=FRAG_HDR_B)


def main():
    models = ("TinyLSTM", "1D-CNN")
    print("v4: loading data ...", flush=True)
    data = build_data()
    print("v4: LoRaWAN accounting ...", flush=True)
    comms = comms_tables()
    print("v4: zone parameters ...", flush=True)
    zones = zone_parameters()
    print("v4: learning experiments ...", flush=True)
    per_seed = run_learning(data, models)
    summary = aggregate(per_seed, models)

    out = dict(seeds=SEEDS, frag_hdr_bytes=FRAG_HDR_B, n_nodes=N_NODES,
               zone_parameters=zones, persistence_per_zone=persistence_per_zone(data),
               comms=comms, models=summary)
    json.dump(out, open(os.path.join(HERE, "results_fl_v4.json"), "w"), indent=2,
              default=float)

    # ---------------- console report ----------------
    print("\n" + "=" * 74)
    print("A1/A2  LoRaWAN accounting  (frag header %d B/frame -> 220 B usable at SF7)"
          % FRAG_HDR_B)
    print(f"{'config':18s} {'bytes':>7s} {'v3 fr':>6s} {'fr':>4s} {'up/day':>8s} "
          f"{'dn/day':>8s} {'e2e':>7s} {'bind':>8s}")
    for k, r in comms["rows"].items():
        print(f"{k:18s} {r['payload_bytes']:7.0f} {r['v3_frames']:6d} "
              f"{r['uplink']['frames']:4d} {r['uplink']['max_uplink_rounds_per_day']:8.1f} "
              f"{r['downlink']['RX2_SF9']['max_downlink_rounds_per_day']:8.1f} "
              f"{r['end_to_end_rx2_sf9']['sustainable_rounds_per_day']:7.1f} "
              f"{r['end_to_end_rx2_sf9']['binding_side']:>8s}")
    print("\n  fleet scaling, sustainable rounds/day (RX2 SF9):")
    for k, v in comms["fleet_scaling_rx2_sf9"].items():
        print(f"    {k:14s} " + "  ".join(f"N={n}:{v[n]}" for n in ("3", "5", "10", "20", "50"))
              + f"   multicast:{v['multicast']}")
    print("\n  downlink windows, quant_only payload:")
    q = comms["rows"]["quant_only"]
    for w, d in q["downlink"].items():
        print(f"    {w:9s} SF{d['sf']:<3d} maxapp {d['max_app_payload']:3d} duty "
              f"{d['duty']:.2f}  {d['frames']:3d} frames  {d['max_downlink_rounds_per_day']:7.1f} rounds/day")

    print("\n" + "=" * 74)
    print("B4  zone parameters")
    for z, r in zones.items():
        print(f"  {z}: {r['date']} {r['start']}-{r['end']} ({r['duration_min']} min) "
              f"n={r['samples']}  PM2.5 {r['pm25_mean']} (CV {r['pm25_cv']}, "
              f"{r['pm25_min']}-{r['pm25_max']})  T {r['temp_mean']}  RH {r['rh_mean']}")

    for mname in models:
        m = summary[mname]
        print("\n" + "=" * 74)
        print(f"{mname} ({m['params']} params)")
        print("B1  client drift over 30 rounds")
        for key, lab in (("drift_noniid", "non-IID (real zones)"), ("drift_iid", "IID control")):
            d = m[key]
            print(f"    {lab:22s} cos {d['cos_mean']['mean']:+.3f}+-{d['cos_mean']['std']:.3f}"
                  f"   (r1 {d['cos_first_round']['mean']:+.3f} -> r30 {d['cos_last_round']['mean']:+.3f})"
                  f"   rel.L2 drift {d['drift_mean']['mean']:.2f}")
        print("B2  T-FL-Rep (federated representation + private head)")
        rep = m["T-FL-Rep"]
        print(f"    Global RMSE {rep['Global']['rmse']['mean']:.2f}+-{rep['Global']['rmse']['std']:.2f}"
              f"   R2 {rep['Global']['r2']['mean']:.2f}"
              f"   uplink f32 {rep['uplink_bytes_per_round_f32']} B / int8 {rep['uplink_bytes_per_round_int8']} B")
        for z in [k for k in rep if k.startswith("Zone")]:
            print(f"      {z}: RMSE {rep[z]['rmse']['mean']:.2f}  R2 {rep[z]['r2']['mean']:+.2f}")
        print("B3  leave-one-zone-out (RMSE on the held-out zone's test windows)")
        print(f"      {'held-out':8s} {'persist':>8s} {'T-FL seen':>10s} {'zero-shot':>10s} "
              + " ".join(f"{'+'+str(r)+' rnd':>9s}" for r in ADAPT_ROUNDS))
        for z, row in m["lozo"].items():
            pz = out["persistence_per_zone"][z]["rmse"]
            print(f"      {z:8s} {pz:8.2f} {m['T-FL_seen'][z]['mean']:10.2f} "
                  f"{row['zero_shot']['rmse']['mean']:10.2f} "
                  + " ".join(f"{row[f'adapt_{r}_rounds']['rmse']['mean']:9.2f}" for r in ADAPT_ROUNDS))
    print("\nSaved -> results_fl_v4.json")


if __name__ == "__main__":
    main()
