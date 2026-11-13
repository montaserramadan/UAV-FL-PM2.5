"""
Monte Carlo protocol simulation of federated rounds over EU868 LoRaWAN
(IEEE IoT-J second revision, Reviewer 1 comment 4).

The first-revision analysis (fl_experiment_v4.py) gives an analytical ceiling
on rounds/day from duty cycles alone.  This simulator replaces the ideal link
with a protocol-level model and reports what survives:

  * frame loss on the link (packet error rate, swept), uplink and downlink;
  * collisions, pure-ALOHA: explicit between fleet frames on the same channel,
    and against Poisson background traffic from other devices in the cell;
  * gateway half-duplex: an uplink overlapping any gateway transmission is lost;
  * Class A downlink: every downlink frame must be solicited by an uplink (poll)
    and is sent in RX2 (869.525 MHz, SF9, ETSI g3, 10 % duty cycle) two seconds
    after it; optionally a Class C multicast of the global model followed by
    unicast repair of missing fragments;
  * fragment-level selective repeat: after a node's update fragments, the
    gateway returns an ACK bitmap; missing fragments are re-sent, up to
    MAX_TRIES attempts per fragment, after which the node is dropped from that
    round (FedAvg then aggregates the nodes that did arrive);
  * duty cycles enforced per device: node uplinks in g1 (1 %), gateway RX2
    downlinks in g3 (10 %);
  * full PHY payload: application bytes + 2 B fragment header + 13 B LoRaWAN
    MAC overhead (MHDR, FHDR, FPort, MIC).

It is a simulation, not a measurement: it omits capture effect (pessimistic),
inter-SF interference, ADR, and gateway receive-path limits (optimistic).
Writes results_lorawan_sim.json.
"""
import heapq, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fl_experiment_v3 import lorawan_airtime

MAC_OH = 13          # MHDR 1 + FHDR 7 + FPort 1 + MIC 4
FRAG_HDR = 2
UL = dict(sf=7, max_app=222, duty=0.01, channels=3)
DL = dict(sf=9, max_app=115, duty=0.10)
RX2_DELAY = 2.0
POLL_APP = 2         # poll / ack-request carries a 2 B status
ACK_APP = 6          # ACK bitmap downlink
MAX_TRIES = 8
UL_JITTER = 2.0      # s, per-frame transmit jitter
START_SPREAD = 10.0  # s, random start of the update upload after training
BG_APP = 20          # typical background sensor uplink


def toa(app, sf):
    return lorawan_airtime(app + MAC_OH, sf=sf)


T_UL_FULL = toa(UL["max_app"], UL["sf"])
T_POLL = toa(POLL_APP, UL["sf"])
T_DL_FULL = toa(DL["max_app"], DL["sf"])
T_ACK = toa(ACK_APP, DL["sf"])
T_BG = toa(BG_APP, UL["sf"])
# back-off after an unserved poll: up to one gateway off-period for a full DL frame
POLL_BACKOFF = T_DL_FULL * (1.0 / DL["duty"] - 1.0)


def nfrag(payload, max_app):
    return max(1, math.ceil(payload / (max_app - FRAG_HDR)))


class Sim:
    def __init__(self, payload, n_nodes, per, bg_load, multicast, t_train, seed, dl_payload=None):
        self.rng = np.random.default_rng(seed)
        self.N, self.per, self.mc, self.t_train = n_nodes, per, multicast, t_train
        self.U = nfrag(payload, UL["max_app"])
        self.D = nfrag(payload if dl_payload is None else dl_payload, DL["max_app"])
        # background: Poisson per channel with normalised offered load bg_load
        self.lam_bg = bg_load / T_BG
        self.node_free = np.zeros(n_nodes)      # duty-cycle release time, g1
        self.gw_free = 0.0                      # duty-cycle release time, g3
        self.gw_tx = []                         # (start, end) gateway transmissions
        self.fleet_tx = []                      # (start, end, channel, node)
        self.node_airtime = np.zeros(n_nodes)
        self.gw_airtime = 0.0
        self.ul_frames = np.zeros(n_nodes)

    # --------------------------------------------------------------- radio
    def _ul(self, node, t, dur):
        """Node transmits an uplink at the earliest duty-allowed time >= t.
        Returns (start, end, channel)."""
        # transmit jitter, as LoRaWAN stacks randomise uplink timing; without it
        # nodes released together stay phase-locked by the common duty period
        s = max(t, self.node_free[node]) + self.rng.uniform(0.0, UL_JITTER)
        e = s + dur
        self.node_free[node] = s + dur / UL["duty"]
        ch = int(self.rng.integers(UL["channels"]))
        self.fleet_tx.append((s, e, ch, node))
        self.node_airtime[node] += dur
        self.ul_frames[node] += 1
        return s, e, ch

    def _ul_ok(self, s, e, ch, node):
        """Resolved after the fact: link loss, background, fleet collision,
        gateway half-duplex."""
        if self.rng.random() < self.per:
            return False
        if self.rng.random() > math.exp(-self.lam_bg * ((e - s) + T_BG)):
            return False
        for (a, b, c, n) in self.fleet_tx:
            if n != node and c == ch and a < e and b > s:
                return False
        for (a, b) in self.gw_tx:
            if a < e and b > s:
                return False
        return True

    def _gw_send(self, t, dur):
        """Gateway transmits at exactly t if its g3 duty cycle allows, else
        returns None (a Class A RX window cannot be postponed)."""
        if t < self.gw_free:
            return None
        for (a, b) in self.gw_tx:
            if a < t + dur and b > t:
                return None
        self.gw_tx.append((t, t + dur))
        self.gw_free = t + dur / DL["duty"]
        self.gw_airtime += dur
        return t + dur

    def _gw_send_asap(self, t, dur):
        """Class C: gateway may transmit whenever free."""
        s = max(t, self.gw_free)
        while any(a < s + dur and b > s for (a, b) in self.gw_tx):
            s = max(b for (a, b) in self.gw_tx if a < s + dur and b > s)
        self.gw_tx.append((s, s + dur))
        self.gw_free = s + dur / DL["duty"]
        self.gw_airtime += dur
        return s + dur

    # --------------------------------------------------------------- round
    def round(self, t0):
        """Run one synchronous FedAvg round from t0.  Returns (t_end, n_arrived).

        Nodes act concurrently; a discrete-event queue keeps the fleet's frames
        interleaved in time so collisions and half-duplex are resolved against
        what the other nodes were actually doing."""
        need_dl = [set(range(self.D)) for _ in range(self.N)]
        if self.mc:
            t = t0
            for f in range(self.D):
                t = self._gw_send_asap(t, T_DL_FULL)
                for n in range(self.N):
                    if self.rng.random() >= self.per:
                        need_dl[n].discard(f)
        tries = [np.zeros(self.U, int) for _ in range(self.N)]
        have_ul = [set() for _ in range(self.N)]
        done_t = [None] * self.N
        dropped = [False] * self.N
        pend = []   # (resolve_time, kind, node, payload)
        q = []      # (time, seq, node, state)
        seq = 0
        start = t0 if not self.mc else max(t0, self.gw_tx[-1][1] if self.gw_tx else t0)
        for n in range(self.N):
            heapq.heappush(q, (start + self.rng.random(), seq, n, "dl")); seq += 1

        while q:
            t, _, n, st = heapq.heappop(q)
            if st == "dl":
                if not need_dl[n]:
                    heapq.heappush(q, (t + self.t_train + self.rng.uniform(0.0, START_SPREAD),
                                       seq, n, "ul")); seq += 1
                    continue
                s, e, ch = self._ul(n, t, T_POLL)
                # gateway learns of the poll only if it was received; decide at e
                # (approximate: resolve against fleet frames already scheduled)
                ok = self._ul_ok(s, e, ch, n)
                served = False
                if ok:
                    end = self._gw_send(e + RX2_DELAY, T_DL_FULL)
                    served = end is not None
                    if served and self.rng.random() >= self.per:
                        need_dl[n].discard(min(need_dl[n]))
                # unserved poll (lost, or gateway duty-blocked): random back-off
                nxt = e + RX2_DELAY + T_DL_FULL + 0.01
                if not served:
                    nxt += self.rng.uniform(0.0, POLL_BACKOFF)
                heapq.heappush(q, (nxt, seq, n, "dl")); seq += 1
            elif st == "ul":
                missing = [f for f in range(self.U) if f not in have_ul[n]]
                if any(tries[n][f] >= MAX_TRIES for f in missing):
                    dropped[n] = True; done_t[n] = t; continue
                last_e = t
                for f in missing:
                    s, e, ch = self._ul(n, last_e, T_UL_FULL)
                    tries[n][f] += 1
                    pend.append((s, e, ch, n, f))
                    last_e = e
                heapq.heappush(q, (last_e, seq, n, "ack")); seq += 1
            elif st == "ack":
                # resolve this node's outstanding fragments now that the fleet's
                # concurrent frames up to t have been scheduled
                keep = []
                for (s, e, ch, nn, f) in pend:
                    if nn == n:
                        if self._ul_ok(s, e, ch, n):
                            have_ul[n].add(f)
                    else:
                        keep.append((s, e, ch, nn, f))
                pend = keep
                end = self._gw_send(t + RX2_DELAY, T_ACK)
                ack_ok = end is not None and self.rng.random() >= self.per
                if len(have_ul[n]) == self.U and ack_ok:
                    done_t[n] = end; continue
                if len(have_ul[n]) == self.U:
                    # complete at gateway but ACK lost: one ack-request poll
                    s, e, ch = self._ul(n, t + RX2_DELAY + 1.0, T_POLL)
                    heapq.heappush(q, (e, seq, n, "ackonly")); seq += 1
                    continue
                heapq.heappush(q, (t + RX2_DELAY + T_ACK + 0.01, seq, n, "ul")); seq += 1
            elif st == "ackonly":
                end = self._gw_send(t + RX2_DELAY, T_ACK)
                if end is not None and self.rng.random() >= self.per:
                    done_t[n] = end
                else:
                    s, e, ch = self._ul(n, t + RX2_DELAY + 1.0, T_POLL)
                    heapq.heappush(q, (e, seq, n, "ackonly")); seq += 1
        # prune history older than this round to keep scans short
        tend = max(done_t)
        self.fleet_tx = [x for x in self.fleet_tx if x[1] > tend - 60]
        self.gw_tx = [x for x in self.gw_tx if x[1] > tend - 60]
        return tend, self.N - sum(dropped)


def simulate(payload, n_nodes, per, bg_load, multicast=False, t_train=2.0, dl_payload=None,
             rounds=30, seed=0):
    sim = Sim(payload, n_nodes, per, bg_load, multicast, t_train, seed, dl_payload)
    t, durs, arrived = 0.0, [], []
    for _ in range(rounds):
        t1, a = sim.round(t)
        durs.append(t1 - t); arrived.append(a); t = t1
    durs = np.array(durs)
    return dict(rounds_per_day=86400.0 / durs.mean(),
                round_s_mean=float(durs.mean()), round_s_p95=float(np.percentile(durs, 95)),
                full_participation=float(np.mean(np.array(arrived) == n_nodes)),
                ul_frames_per_node_round=float(sim.ul_frames.mean() / rounds),
                ul_frames_ideal=sim.U, dl_frames_ideal=sim.D,
                node_airtime_s_round=float(sim.node_airtime.mean() / rounds),
                gw_airtime_s_round=float(sim.gw_airtime / rounds))


# (uplink update, downlink broadcast) in bytes.  The gateway sends the global
# model: float32 (5,956 B) for the float row, 8-bit (1,513 B) otherwise.
PAYLOADS = {
    "dense f32": (5956, 5956),
    "int8": (1513, 1513),
    "top-k int8 k=0.2": (915, 1513),
    "top-k+EF k=0.1": (468, 1513),
    "EF-signSGD": (211, 1513),
}


def ideal(payload, n, dl_payload=None):
    """Analytical ceiling with the same full PHY payload, for comparison."""
    U = nfrag(payload, UL["max_app"])
    D = nfrag(payload if dl_payload is None else dl_payload, DL["max_app"])
    up = 86400 * UL["duty"] / (U * T_UL_FULL)
    dn = 86400 * DL["duty"] / (n * D * T_DL_FULL)
    return min(up, dn)


def ideal_serial(payload, n, dl_payload=None):
    """Same airtime, but downlink and uplink phases of a synchronous round run
    one after the other instead of overlapping."""
    U = nfrag(payload, UL["max_app"])
    D = nfrag(payload if dl_payload is None else dl_payload, DL["max_app"])
    t_up = U * T_UL_FULL / UL["duty"]
    t_dn = n * D * T_DL_FULL / DL["duty"]
    return 86400.0 / (t_up + t_dn)


def main():
    reps = 3
    out = dict(assumptions=dict(
        mac_overhead_B=MAC_OH, frag_hdr_B=FRAG_HDR, uplink=UL, downlink_rx2=DL,
        max_tries=MAX_TRIES, ul_jitter_s=UL_JITTER, start_spread_s=START_SPREAD, poll_backoff_s=POLL_BACKOFF, rx2_delay_s=RX2_DELAY, bg_frame_app_B=BG_APP,
        toa_s=dict(ul_full=T_UL_FULL, poll=T_POLL, dl_full=T_DL_FULL, ack=T_ACK, bg=T_BG),
        reps=reps), grid=[])
    print(f"ToA: UL222 {T_UL_FULL*1e3:.1f} ms  poll {T_POLL*1e3:.1f} ms  "
          f"DL115@SF9 {T_DL_FULL*1e3:.1f} ms  ack {T_ACK*1e3:.1f} ms", flush=True)
    pick = sys.argv[1:] or list(PAYLOADS)
    for name, (pb, pd) in [(k, PAYLOADS[k]) for k in pick]:
        for n in (3, 10, 20):
            for per in (0.0, 0.10, 0.20):
                for bg in (0.0, 0.05):
                    for mc in (False, True):
                        if mc and (per, bg) not in ((0.0, 0.0), (0.10, 0.05)):
                            continue
                        rs = [simulate(pb, n, per, bg, mc, rounds=10, seed=r, dl_payload=pd) for r in range(reps)]
                        row = dict(config=name, payload=pb, n=n, per=per, bg_load=bg,
                                   multicast=mc, dl_payload=pd, ideal_rounds_per_day=ideal(pb, n, pd),
                                   ideal_serial_rounds_per_day=ideal_serial(pb, n, pd))
                        for k in rs[0]:
                            v = [r[k] for r in rs]
                            row[k] = float(np.mean(v))
                            if k == "rounds_per_day":
                                row["rounds_per_day_std"] = float(np.std(v))
                        out["grid"].append(row)
                        print(f"{name:17s} N={n:2d} PER={per:.2f} bg={bg:.2f} mc={int(mc)}  "
                              f"ideal {row['ideal_rounds_per_day']:7.1f}  serial {row['ideal_serial_rounds_per_day']:7.1f}  sim {row['rounds_per_day']:7.1f}"
                              f"  full {row['full_participation']:.2f}  "
                              f"ULfr {row['ul_frames_per_node_round']:.1f}/{row['ul_frames_ideal']}",
                              flush=True)
        tag = "" if len(pick) == len(PAYLOADS) else "_" + "_".join(p.split()[0] for p in pick)
        json.dump(out, open(os.path.join(HERE, f"results_lorawan_sim{tag}.json"), "w"), indent=2)
    print("Saved -> results_lorawan_sim.json")


if __name__ == "__main__":
    main()
