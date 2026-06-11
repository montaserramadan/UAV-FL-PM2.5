"""
Reproducible analysis pipeline for the UAV industrial PM2.5 paper.

Inputs : the raw per-flight UAV logs (3 industrial zones + 1 test flight).
Outputs: cleaned data summary, per-zone descriptive stats, spatial
         heterogeneity statistics (Moran's I, Getis-Ord Gi*), IDW heatmap
         leave-one-out cross-validation, and non-IID quantification
         (KS test + Jensen-Shannon divergence) across zones.

No external geostats package required: Moran's I and Gi* are implemented
directly so the results are fully transparent and reproducible.
"""
import csv, json, math, os
import numpy as np
from scipy import stats

np.random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

RAW = {
    "test_2007": "data/uav/test_flight.csv",
    "Zone1":     "data/uav/zone1_vehicle_maintenance.csv",
    "Zone2":     "data/uav/zone2_galvanization.csv",
    "Zone3":     "data/uav/zone3_recycling.csv",
}
ZONE_LABEL = {
    "Zone1": "Vehicle maintenance & repair",
    "Zone2": "Chemical / metal galvanization",
    "Zone3": "Recycling & material recovery",
}

# Istanbul bounding box for GPS sanity filtering
LAT_MIN, LAT_MAX = 40.80, 41.30
LON_MIN, LON_MAX = 28.50, 29.40
ALT_MAX = 2000.0        # m MSL, anything above is GPS garbage for a UAV
SPEED_MAX = 20.0        # m/s, 72 km/h; spikes above are GPS noise


def load_clean(path):
    rows = list(csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8-sig")))
    raw_n = len(rows)
    clean = []
    dropped = {"gps_bbox": 0, "altitude": 0, "speed": 0, "nan": 0}
    for r in rows:
        try:
            lat = float(r["Latitude"]); lon = float(r["Longitude"])
            alt = float(r["Altitude_m"]); spd = float(r["Speed_mps"])
            pm25 = float(r["PM2.5"]); pm10 = float(r["PM10"])
            co2 = float(r["CO2"]); ch2o = float(r["CH2O"]); tvoc = float(r["TVOC"])
            temp = float(r["Temp_C"]); hum = float(r["Humidity_%"])
        except (ValueError, KeyError):
            dropped["nan"] += 1
            continue
        if not (LAT_MIN < lat < LAT_MAX and LON_MIN < lon < LON_MAX):
            dropped["gps_bbox"] += 1; continue
        if alt > ALT_MAX:
            dropped["altitude"] += 1; continue
        if spd > SPEED_MAX:
            dropped["speed"] += 1; continue
        clean.append(dict(lat=lat, lon=lon, alt=alt, spd=spd, pm25=pm25,
                          pm10=pm10, co2=co2, ch2o=ch2o, tvoc=tvoc,
                          temp=temp, hum=hum, time=r["Time"], date=r["Date"]))
    return raw_n, clean, dropped


def to_meters(lat, lon):
    """Equirectangular projection to local meters around the data centroid."""
    lat0 = np.mean(lat)
    x = (lon - np.mean(lon)) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110540.0
    return x, y


def knn_weights(x, y, k=8, row_standardize=True, include_self=False):
    n = len(x)
    coords = np.column_stack([x, y])
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    W = np.zeros((n, n))
    for i in range(n):
        order = np.argsort(d[i])
        order = order[order != i]            # exclude self for neighbour pick
        nn = order[:k]
        W[i, nn] = 1.0
        if include_self:
            W[i, i] = 1.0
    if row_standardize:
        rs = W.sum(1, keepdims=True)
        rs[rs == 0] = 1
        W = W / rs
    return W


def morans_i(z_vals, W, n_perm=999):
    z = z_vals - z_vals.mean()
    n = len(z)
    Wsum = W.sum()
    num = z @ W @ z
    den = (z ** 2).sum()
    I = (n / Wsum) * (num / den)
    # permutation inference
    perms = np.empty(n_perm)
    for p in range(n_perm):
        zp = np.random.permutation(z)
        perms[p] = (n / Wsum) * ((zp @ W @ zp) / den)
    p_val = (np.sum(np.abs(perms) >= abs(I)) + 1) / (n_perm + 1)
    z_score = (I - perms.mean()) / perms.std()
    return float(I), float(z_score), float(p_val)


def getis_ord_gistar(vals, x, y, k=8):
    """Gi* with binary k-nearest weights including self; returns counts of
    statistically significant hot/cold spots at the 95% level."""
    n = len(vals)
    W = knn_weights(x, y, k=k, row_standardize=False, include_self=True)
    Xbar = vals.mean()
    S = math.sqrt((vals ** 2).sum() / n - Xbar ** 2)
    hot = cold = 0
    zmax = 0.0
    for i in range(n):
        w = W[i]
        Wi = w.sum()
        num = (w * vals).sum() - Xbar * Wi
        denom = S * math.sqrt((n * (w ** 2).sum() - Wi ** 2) / (n - 1))
        if denom == 0:
            continue
        gi = num / denom
        zmax = max(zmax, abs(gi))
        if gi > 1.96: hot += 1
        elif gi < -1.96: cold += 1
    return hot, cold, float(zmax)


def idw_loocv(vals, x, y, power=2, k=12):
    """Leave-one-out cross-validation of inverse-distance-weighting maps."""
    n = len(vals)
    coords = np.column_stack([x, y])
    preds = np.empty(n)
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(1))
        d[i] = np.inf
        order = np.argsort(d)[:k]
        dd = d[order]
        w = 1.0 / np.power(dd, power)
        preds[i] = np.sum(w * vals[order]) / np.sum(w)
    err = preds - vals
    rmse = math.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((vals - vals.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(rmse=rmse, mae=mae, r2=r2, preds=preds.tolist())


def js_divergence(a, b, bins, rng):
    pa, _ = np.histogram(a, bins=bins, range=rng, density=True)
    pb, _ = np.histogram(b, bins=bins, range=rng, density=True)
    pa = pa + 1e-12; pb = pb + 1e-12
    pa /= pa.sum(); pb /= pb.sum()
    m = 0.5 * (pa + pb)
    kl = lambda p, q: np.sum(p * np.log2(p / q))
    return float(0.5 * kl(pa, m) + 0.5 * kl(pb, m))


def descstats(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), min=float(v.min()), max=float(v.max()),
                mean=float(v.mean()), median=float(np.median(v)),
                std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                cv=float(v.std(ddof=1) / v.mean()) if v.mean() else float("nan"),
                skew=float(stats.skew(v)), n_unique=int(np.unique(v).size))


def main():
    report = {"cleaning": {}, "descriptive": {}, "spatial": {}, "noniid": {}}
    zones = {}
    for key, path in RAW.items():
        raw_n, clean, dropped = load_clean(path)
        report["cleaning"][key] = dict(raw=raw_n, kept=len(clean), dropped=dropped)
        zones[key] = clean

    print("=" * 70)
    print("DATA CLEANING")
    print("=" * 70)
    total_kept = 0
    for k, c in report["cleaning"].items():
        print(f"{k:10s} raw={c['raw']:4d}  kept={c['kept']:4d}  dropped={c['dropped']}")
        if k != "test_2007":
            total_kept += c['kept']
    print(f"\nTotal usable ZONE points (excl. test flight): {total_kept}")

    print("\n" + "=" * 70)
    print("PER-ZONE DESCRIPTIVE STATS")
    print("=" * 70)
    pollutants = ["pm25", "pm10", "co2", "ch2o", "tvoc", "temp", "hum"]
    for z in ["Zone1", "Zone2", "Zone3"]:
        c = zones[z]
        report["descriptive"][z] = {"label": ZONE_LABEL[z],
                                    "time": f"{c[0]['date']} {c[0]['time']}-{c[-1]['time']}"}
        print(f"\n--- {z} ({ZONE_LABEL[z]}) | {report['descriptive'][z]['time']} | n={len(c)} ---")
        for p in pollutants:
            s = descstats([r[p] for r in c])
            report["descriptive"][z][p] = s
            if p in ("pm25", "pm10", "co2"):
                print(f"  {p:5s} mean={s['mean']:7.1f}  range[{s['min']:.0f},{s['max']:.0f}]"
                      f"  CV={s['cv']:.2f}  skew={s['skew']:+.2f}  uniq={s['n_unique']}")

    print("\n" + "=" * 70)
    print("SPATIAL HETEROGENEITY  (PM2.5)")
    print("=" * 70)
    for z in ["Zone1", "Zone2", "Zone3"]:
        c = zones[z]
        lat = np.array([r["lat"] for r in c]); lon = np.array([r["lon"] for r in c])
        pm = np.array([r["pm25"] for r in c])
        x, y = to_meters(lat, lon)
        extent_x = x.max() - x.min(); extent_y = y.max() - y.min()
        W = knn_weights(x, y, k=8)
        I, zsc, pval = morans_i(pm, W)
        hot, cold, zmax = getis_ord_gistar(pm, x, y, k=8)
        cv = idw_loocv(pm, x, y)
        report["spatial"][z] = dict(
            extent_m=[float(extent_x), float(extent_y)],
            morans_I=I, morans_z=zsc, morans_p=pval,
            gi_hot=hot, gi_cold=cold, gi_zmax=zmax,
            idw_rmse=cv["rmse"], idw_mae=cv["mae"], idw_r2=cv["r2"])
        print(f"\n--- {z} | spatial extent ~{extent_x:.0f} x {extent_y:.0f} m ---")
        print(f"  Moran's I = {I:+.3f}  (z={zsc:+.2f}, p={pval:.3f})  "
              f"-> {'significant spatial clustering' if pval<0.05 else 'no sig. clustering'}")
        print(f"  Getis-Ord Gi*: {hot} hot-spot pts, {cold} cold-spot pts (|z|>1.96), max|z|={zmax:.2f}")
        print(f"  IDW map LOOCV: RMSE={cv['rmse']:.2f} ug/m3  MAE={cv['mae']:.2f}  R2={cv['r2']:.2f}")

    print("\n" + "=" * 70)
    print("NON-IID QUANTIFICATION  (cross-zone PM2.5)")
    print("=" * 70)
    pm = {z: np.array([r["pm25"] for r in zones[z]]) for z in ["Zone1", "Zone2", "Zone3"]}
    allpm = np.concatenate(list(pm.values()))
    rng = (allpm.min(), allpm.max())
    pairs = [("Zone1", "Zone2"), ("Zone1", "Zone3"), ("Zone2", "Zone3")]
    for a, b in pairs:
        ks, ksp = stats.ks_2samp(pm[a], pm[b])
        js = js_divergence(pm[a], pm[b], bins=20, rng=rng)
        report["noniid"][f"{a}_vs_{b}"] = dict(ks=float(ks), ks_p=float(ksp), js_div=js)
        print(f"  {a} vs {b}: KS={ks:.3f} (p={ksp:.1e})  JS-divergence={js:.3f} bits")
    # global Kruskal-Wallis (do the 3 zones share a distribution?)
    H, kp = stats.kruskal(pm["Zone1"], pm["Zone2"], pm["Zone3"])
    report["noniid"]["kruskal_wallis"] = dict(H=float(H), p=float(kp))
    print(f"\n  Kruskal-Wallis across 3 zones: H={H:.1f}, p={kp:.1e} "
          f"-> {'distributions differ (non-IID confirmed)' if kp<0.05 else 'no diff'}")

    out = os.path.join(HERE, "results_spatial.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\nSaved -> {out}")
    return report


if __name__ == "__main__":
    main()
