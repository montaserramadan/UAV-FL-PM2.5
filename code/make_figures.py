"""Generate publication figures from the verified analysis (real data only)."""
import csv, os, math, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

np.random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

RAW = {
    "Zone1": "data/uav/zone1_vehicle_maintenance.csv",
    "Zone2": "data/uav/zone2_galvanization.csv",
    "Zone3": "data/uav/zone3_recycling.csv",
}
LABEL = {"Zone1": "Zone 1\nVehicle maintenance",
         "Zone2": "Zone 2\nGalvanization",
         "Zone3": "Zone 3\nRecycling"}
COL = {"Zone1": "#d1495b", "Zone2": "#edae49", "Zone3": "#00798c"}
LAT = (40.80, 41.30); LON = (28.50, 29.40)

def load(path):
    rows = list(csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8-sig")))
    out = []
    for r in rows:
        try:
            lat=float(r["Latitude"]); lon=float(r["Longitude"])
            alt=float(r["Altitude_m"]); spd=float(r["Speed_mps"]); pm=float(r["PM2.5"])
        except: continue
        if not(LAT[0]<lat<LAT[1] and LON[0]<lon<LON[1]): continue
        if alt>2000 or spd>20: continue
        out.append((lat,lon,pm))
    return np.array(out)

data = {z: load(p) for z,p in RAW.items()}
rep = json.load(open(os.path.join(HERE,"results_spatial.json")))

plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":0.3,
                     "figure.dpi":150,"savefig.bbox":"tight"})

# ---- Fig A: cross-zone PM2.5 distributions (the non-IID picture) ----
fig, ax = plt.subplots(figsize=(6,4))
pmlists = [data[z][:,2] for z in ["Zone1","Zone2","Zone3"]]
parts = ax.violinplot(pmlists, showmeans=True, showextrema=True, widths=0.8)
for i,pc in enumerate(parts['bodies']):
    pc.set_facecolor(list(COL.values())[i]); pc.set_alpha(0.6)
for key in ('cbars','cmins','cmaxes','cmeans'):
    parts[key].set_color('0.3')
ax.set_xticks([1,2,3]); ax.set_xticklabels([LABEL[z] for z in ["Zone1","Zone2","Zone3"]])
ax.set_ylabel("PM$_{2.5}$ (µg m$^{-3}$)")
ax.set_title("Cross-zone PM$_{2.5}$ distributions (non-IID)")
H=rep["noniid"]["kruskal_wallis"]["H"]; p=rep["noniid"]["kruskal_wallis"]["p"]
ax.text(0.97,0.95,f"Kruskal-Wallis H={H:.0f}, p<1e-50",transform=ax.transAxes,
        ha="right",va="top",fontsize=8,style="italic")
fig.savefig(os.path.join(FIG,"figA_zone_distributions.png")); plt.close(fig)

# ---- Fig B: spatial scatter maps colored by PM2.5 (single midday snapshot) ----
fig = plt.figure(figsize=(12,3.8))
gs = gridspec.GridSpec(1,3,wspace=0.28)
for j,z in enumerate(["Zone1","Zone2","Zone3"]):
    ax = fig.add_subplot(gs[j])
    d = data[z]; lat=d[:,1]; lon=d[:,0]; pm=d[:,2]  # note: col0 lat,col1 lon
    lat=d[:,0]; lon=d[:,1]
    sc = ax.scatter(lon,lat,c=pm,cmap="turbo",s=22,edgecolor="k",linewidth=0.2,
                    vmin=0,vmax=max(80,pm.max()))
    ax.set_title(LABEL[z].replace("\n"," — "),fontsize=9)
    ax.set_xlabel("Longitude");
    if j==0: ax.set_ylabel("Latitude")
    sp=rep["spatial"][z]
    ax.text(0.03,0.97,f"Moran's I={sp['morans_I']:.2f}\n(p={sp['morans_p']:.3f})",
            transform=ax.transAxes,va="top",fontsize=8,
            bbox=dict(boxstyle="round",fc="white",alpha=0.8))
    cb=fig.colorbar(sc,ax=ax,fraction=0.046,pad=0.04); cb.set_label("PM$_{2.5}$")
fig.suptitle("Spatial PM$_{2.5}$ along UAV flight paths (single midday snapshot per zone)",y=1.04)
fig.savefig(os.path.join(FIG,"figB_spatial_scatter.png")); plt.close(fig)

# ---- Fig C: non-IID divergence matrix ----
zs=["Zone1","Zone2","Zone3"]
M=np.zeros((3,3))
pairmap={("Zone1","Zone2"):"Zone1_vs_Zone2",("Zone1","Zone3"):"Zone1_vs_Zone3",
         ("Zone2","Zone3"):"Zone2_vs_Zone3"}
for a in range(3):
    for b in range(3):
        if a==b: M[a,b]=0
        else:
            key=pairmap.get((zs[min(a,b)],zs[max(a,b)]))
            M[a,b]=rep["noniid"][key]["js_div"]
fig,ax=plt.subplots(figsize=(4.6,4))
im=ax.imshow(M,cmap="magma_r")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(["Z1","Z2","Z3"]); ax.set_yticklabels(["Z1","Z2","Z3"])
for a in range(3):
    for b in range(3):
        ax.text(b,a,f"{M[a,b]:.2f}",ha="center",va="center",
                color="white" if M[a,b]>0.25 else "black")
ax.set_title("Pairwise PM$_{2.5}$ Jensen–Shannon\ndivergence (bits)")
fig.colorbar(im,fraction=0.046,pad=0.04)
fig.savefig(os.path.join(FIG,"figC_noniid_matrix.png")); plt.close(fig)

# ---- Fig D: IDW leave-one-out map validation ----
fig,axes=plt.subplots(1,3,figsize=(12,3.8))
for j,z in enumerate(["Zone1","Zone2","Zone3"]):
    ax=axes[j]; d=data[z]; pm=d[:,2]
    # recompute preds quickly via stored r2/rmse text only (scatter from fresh idw)
    from spatial_analysis import idw_loocv, to_meters
    x,y=to_meters(d[:,0],d[:,1])
    cv=idw_loocv(pm,x,y); preds=np.array(cv["preds"])
    ax.scatter(pm,preds,s=18,alpha=0.6,color=COL[z],edgecolor="k",linewidth=0.2)
    lim=[0,max(pm.max(),preds.max())*1.05]
    ax.plot(lim,lim,"k--",lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Observed PM$_{2.5}$");
    if j==0: ax.set_ylabel("IDW LOOCV predicted")
    ax.set_title(f"{z}: R$^2$={cv['r2']:.2f}, RMSE={cv['rmse']:.1f}",fontsize=9)
fig.suptitle("Heatmap interpolation accuracy (inverse-distance weighting, leave-one-out CV)",y=1.03)
fig.savefig(os.path.join(FIG,"figD_idw_validation.png")); plt.close(fig)

print("Figures written to", FIG)
for f in sorted(os.listdir(FIG)): print("  ",f)
