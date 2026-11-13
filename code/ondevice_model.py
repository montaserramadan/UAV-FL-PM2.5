"""
On-device cost model for one federated round on the UAV node
(IEEE IoT-J second revision, Reviewer 1 comment 7).

No microcontroller was available, so this is a counted-operations model, not a
measurement.  What is EXACT: the floating-point operation count of forward and
backward passes (torch.utils.flop_counter on the real models, real batch size
and window length), the Adam update cost, and every memory term (parameters,
gradients, optimiser state, back-propagation activations, training buffer).
What is ASSUMED, and swept: the effective FP32 throughput a hand-written C
implementation attains on each MCU class, and the active power.  The output is
therefore a bracket, reported with the assumption that produced each end.

Writes results_ondevice.json.
"""
import json, math, os, sys
import torch
from torch.utils.flop_counter import FlopCounterMode

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fl_experiment_v3 as v3
from fl_experiment_v3 import TinyLSTM, CNN1D, SEQ, FEATS, LOCAL_EPOCHS

BS = 8
ADAM_FLOPS_PER_PARAM = 12          # m, v updates, bias correction, sqrt, divide, step
F32 = 4

# MCU classes.  sram_kB from the vendor datasheets; eff_mflops is an ASSUMED
# bracket for scalar FP32 C code on one core (low = 0.1 FLOP/cycle,
# high = 0.5 FLOP/cycle); active_mW is an ASSUMED conservative bracket for the
# MCU core running flat out (radio off).
MCUS = {
    "Cortex-M4F @ 80 MHz (STM32L4-class)": dict(mhz=80, sram_kB=128, active_mW=(15, 40)),
    "Cortex-M4F @ 64 MHz (nRF52840-class)": dict(mhz=64, sram_kB=256, active_mW=(15, 40)),
    "Xtensa LX7 @ 240 MHz (ESP32-S3-class)": dict(mhz=240, sram_kB=512, active_mW=(150, 330)),
}
FLOP_PER_CYCLE = (0.1, 0.5)
# UAV propulsion power, small quadcopter, ASSUMED bracket (W)
FLIGHT_W = (100.0, 250.0)


def count_flops(mcls):
    m = mcls(len(FEATS))
    x = torch.randn(BS, SEQ, len(FEATS)); y = torch.randn(BS)
    with FlopCounterMode(display=False) as fc:
        out = m(x)
    fwd = fc.get_total_flops()
    with FlopCounterMode(display=False) as fc:
        loss = ((m(x) - y) ** 2).mean(); loss.backward()
    fwd_bwd = fc.get_total_flops()
    return m, fwd, fwd_bwd


def lstm_manual_flops(hidden=16, nin=len(FEATS)):
    """Cross-check: LSTM cell MACs per step = 4*h*(nin+h) (+ biases); 2 FLOPs/MAC;
    plus element-wise gate maths ~ 10*h per step."""
    mac = 4 * hidden * (nin + hidden)
    return SEQ * (2 * mac + 4 * hidden * 2 + 10 * hidden) + 2 * hidden  # + read-out


def memory(m, hidden=16):
    P = sum(p.numel() for p in m.parameters())
    params = P * F32; grads = P * F32; adam = 2 * P * F32
    if isinstance(m, TinyLSTM):
        # per step, kept for BPTT: input, 4 gates, cell, hidden, tanh(cell)
        per_step = len(FEATS) + 4 * hidden + 3 * hidden
        acts = BS * SEQ * per_step * F32
    else:
        ch = m.conv.out_channels; L = SEQ - m.conv.kernel_size[0] + 1
        acts = BS * (SEQ * len(FEATS) + 2 * ch * L) * F32
    return dict(params_B=params, grads_B=grads, adam_B=adam, activations_B=acts,
                total_training_B=params + grads + adam + acts)


def main():
    data = v3  # zone window counts from the leakage-safe split
    import fl_experiment_v4 as v4
    d = v4.build_data()
    windows = {z: int(len(v[0])) for z, v in d["clients_tr"].items()}
    raw_rows = {z: len(v3.load_uav(p)) for z, p in v3.UAV.items()}
    out = dict(batch=BS, seq=SEQ, local_epochs=LOCAL_EPOCHS, train_windows=windows,
               assumptions=dict(flop_per_cycle=FLOP_PER_CYCLE, flight_W=FLIGHT_W,
                                adam_flops_per_param=ADAM_FLOPS_PER_PARAM), models={})
    for name, mcls in (("TinyLSTM", TinyLSTM), ("1D-CNN", CNN1D)):
        m, fwd, fb = count_flops(mcls)
        if name == "TinyLSTM":
            # FlopCounterMode does not see inside the fused LSTM kernel (it counts
            # only the read-out); use the analytic cell count, backward = 2x forward
            fwd = BS * lstm_manual_flops(); fb = 3 * fwd
        P = sum(p.numel() for p in m.parameters())
        per_step_flops = fb + P * ADAM_FLOPS_PER_PARAM          # one minibatch of 8
        mem = memory(m)
        # training buffer: raw records of the zone's 70 % split, 5 channels f32;
        # windows are cut on the fly, never materialised
        buf = {z: int(0.7 * r) * len(FEATS) * F32 for z, r in raw_rows.items()}
        per_round = {}
        for z, nw in windows.items():
            steps = LOCAL_EPOCHS * math.ceil(nw / BS)
            per_round[z] = dict(minibatch_steps=steps, flops=steps * per_step_flops)
        worst = max(v["flops"] for v in per_round.values())
        mcu = {}
        for mn, spec in MCUS.items():
            lo_t = worst / (spec["mhz"] * 1e6 * FLOP_PER_CYCLE[1])
            hi_t = worst / (spec["mhz"] * 1e6 * FLOP_PER_CYCLE[0])
            e_lo = lo_t * spec["active_mW"][0] / 1e3
            e_hi = hi_t * spec["active_mW"][1] / 1e3
            ram = mem["total_training_B"] + max(buf.values())
            mcu[mn] = dict(train_s_per_round=[lo_t, hi_t], energy_J_per_round=[e_lo, e_hi],
                           ram_needed_kB=ram / 1024, sram_kB=spec["sram_kB"],
                           ram_fraction=ram / 1024 / spec["sram_kB"],
                           flight_seconds_equivalent=[e_lo / FLIGHT_W[1], e_hi / FLIGHT_W[0]])
        out["models"][name] = dict(
            params=P, fwd_flops_batch=fwd, fwd_bwd_flops_batch=fb,
            manual_fwd_flops_window=(lstm_manual_flops() if name == "TinyLSTM" else None),
            flops_per_minibatch_step=per_step_flops, memory=mem, train_buffer_B=buf,
            per_round=per_round, worst_round_flops=worst, mcu=mcu)
        print(f"\n{name}: {P} params  fwd/batch {fwd:,} FLOPs  fwd+bwd/batch {fb:,}  "
              f"(manual fwd/window {lstm_manual_flops() if name=='TinyLSTM' else '-'})")
        print(f"  memory: params {mem['params_B']} B, grads {mem['grads_B']} B, adam {mem['adam_B']} B, "
              f"acts {mem['activations_B']} B -> {mem['total_training_B']/1024:.1f} kB "
              f"(+ buffer {max(buf.values())/1024:.1f} kB)")
        print(f"  worst-zone round: {worst/1e6:.2f} MFLOP")
        for mn, r in mcu.items():
            print(f"  {mn:40s} {r['train_s_per_round'][0]:6.2f}-{r['train_s_per_round'][1]:6.2f} s  "
                  f"{r['energy_J_per_round'][0]:6.3f}-{r['energy_J_per_round'][1]:6.3f} J  "
                  f"RAM {r['ram_needed_kB']:.0f}/{r['sram_kB']} kB  "
                  f"= {r['flight_seconds_equivalent'][0]*1e3:.1f}-{r['flight_seconds_equivalent'][1]*1e3:.1f} ms of flight")
    json.dump(out, open(os.path.join(HERE, "results_ondevice.json"), "w"), indent=2)
    print("\nSaved -> results_ondevice.json")


if __name__ == "__main__":
    main()
