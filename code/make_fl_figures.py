"""FL figures: convergence curves + FL vs T-FL comparison (from real results)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
r = json.load(open(os.path.join(HERE, "results_fl.json")))
pers = r["persistence"]["rmse"]

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# ---- Convergence ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, m in zip(axes, ["TinyLSTM", "1D-CNN"]):
    fl = r["models"][m]["FL"]["convergence_rmse"]
    tf = r["models"][m]["T-FL"]["convergence_rmse"]
    x = np.arange(1, len(fl) + 1)
    ax.plot(x, fl, "-o", ms=3, color="#5b5f97", label="FL (random init)")
    ax.plot(x, tf, "-s", ms=3, color="#d1495b", label="T-FL (pretrained init)")
    ax.axhline(pers, ls="--", color="0.5", lw=1, label=f"Persistence ({pers:.1f})")
    ax.set_title(m); ax.set_xlabel("Federated round")
    ax.set_ylabel("Global test RMSE (µg m$^{-3}$)"); ax.legend(fontsize=8)
fig.suptitle("Federated convergence: transfer initialization converges faster and lower", y=1.02)
fig.savefig(os.path.join(FIG, "figE_convergence.png")); plt.close(fig)

# ---- Comparison bars (Global RMSE and R2) ----
models = ["TinyLSTM", "1D-CNN"]; settings = ["FL", "T-FL"]
rmse = np.array([[r["models"][m][s]["Global"]["rmse"] for s in settings] for m in models])
r2 = np.array([[r["models"][m][s]["Global"]["r2"] for s in settings] for m in models])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
w = 0.35; x = np.arange(len(models))
for k, s in enumerate(settings):
    a1.bar(x + (k - 0.5) * w, rmse[:, k], w, label=s,
           color=["#5b5f97", "#d1495b"][k])
a1.axhline(pers, ls="--", color="0.5", lw=1, label=f"Persistence ({pers:.1f})")
a1.set_xticks(x); a1.set_xticklabels(models); a1.set_ylabel("Global RMSE (µg m$^{-3}$)")
a1.set_title("Prediction error"); a1.legend(fontsize=8)
for k, s in enumerate(settings):
    a2.bar(x + (k - 0.5) * w, r2[:, k], w, label=s, color=["#5b5f97", "#d1495b"][k])
a2.set_xticks(x); a2.set_xticklabels(models); a2.set_ylabel("Global R$^2$")
a2.set_ylim(0, 1); a2.set_title("Goodness of fit"); a2.legend(fontsize=8)
fig.suptitle("Global PM$_{2.5}$ prediction: FL vs Transfer-FL", y=1.02)
fig.savefig(os.path.join(FIG, "figF_fl_comparison.png")); plt.close(fig)

print("Wrote figE_convergence.png, figF_fl_comparison.png")
