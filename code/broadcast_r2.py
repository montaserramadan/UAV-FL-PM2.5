"""Downlink realism check (second revision, pre-submission audit).

The LoRaWAN accounting must size the DOWNLINK as the global model the gateway
broadcasts, not as one client's compressed update.  We take the broadcast to
be the global model quantised to 8 bits per tensor (1,513 B for the TinyLSTM),
and test here whether clients that start each round from that quantised model,
instead of the exact float32 one, lose accuracy.  Same seeds, same compressors
as fl_experiment_v5 section C.  Writes results_broadcast_r2.json."""
import copy, json, os, sys
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fl_experiment_v5 as v5
from fl_experiment_v3 import TinyLSTM, ROUNDS, LOCAL_EPOCHS, LR, SEEDS, FEATS, train_local


def q8(state):
    out = {}
    for k, v in state.items():
        s = v.abs().max() / 127.0 + 1e-12
        out[k] = torch.round(v / s).clamp(-127, 127) * s
    return out


def federated_q(init, data, comp, seed, quant_broadcast):
    torch.manual_seed(seed)
    glob = TinyLSTM(len(FEATS)); glob.load_state_dict(copy.deepcopy(init))
    hist = []
    for rnd in range(ROUNDS):
        g = copy.deepcopy(glob.state_dict())
        start = q8(g) if quant_broadcast else g          # what the clients receive
        deltas, sizes = [], []
        for cid, (Xtr, ytr) in data["clients_tr"].items():
            loc = TinyLSTM(len(FEATS)); loc.load_state_dict(copy.deepcopy(start))
            train_local(loc, Xtr, ytr, LOCAL_EPOCHS, LR)
            d = {k: loc.state_dict()[k] - start[k] for k in start}
            cd, _ = comp(cid, d, rnd)
            deltas.append(cd); sizes.append(len(Xtr))
        w = np.array(sizes, float); w /= w.sum()
        glob.load_state_dict({k: g[k] + sum(deltas[i][k] * float(w[i]) for i in range(len(deltas))) for k in g})
        hist.append(v5.eval_global(glob, data)["Global"]["rmse"])
    return hist


CFGS = [("dense f32", dict(name="dense")), ("int8", dict(name="int8")),
        ("top-k int8+EF k=0.2", dict(name="topk_ef", k=0.2)),
        ("top-k int8+EF k=0.1", dict(name="topk_ef", k=0.1)),
        ("EF-signSGD", dict(name="sign_ef"))]
data = v5.build(H=1)
res = {lab: {"exact": [], "q8": []} for lab, _ in CFGS}
for s in SEEDS:
    pre = v5.pretrain(TinyLSTM, data, s)
    for lab, cfg in CFGS:
        for mode in ("exact", "q8"):
            h = federated_q(pre, data, v5.Compressor(seed=s, **cfg), s, mode == "q8")
            res[lab][mode].append(h[-1])
    print("seed", s, flush=True)
out = {}
for lab, r in res.items():
    e, q = np.array(r["exact"]), np.array(r["q8"])
    out[lab] = dict(exact_mean=float(e.mean()), q8_mean=float(q.mean()), q8_std=float(q.std()),
                    diff_mean=float((q - e).mean()), diff_max_abs=float(np.abs(q - e).max()),
                    exact=e.tolist(), q8=q.tolist())
    print(f"{lab:22s} exact {e.mean():.3f}  8-bit broadcast {q.mean():.3f}+-{q.std():.2f}  diff {(q-e).mean():+.3f} (max {np.abs(q-e).max():.3f})")
json.dump(out, open(os.path.join(HERE, "results_broadcast_r2.json"), "w"), indent=2)
