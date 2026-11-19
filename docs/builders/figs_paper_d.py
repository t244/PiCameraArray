"""Ablation figure: floor against views per band, raw and warped."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
M, OUT = sys.argv[1], sys.argv[2]
d = json.load(open(f"{M}/ablation_views.json"))
SHIZ, AKA, MUTED, GRID = "#11709C", "#C3592C", "#666666", "#DDDDDD"
plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=220)
ks = [1, 2, 3, 4]
for ax, key, title in ((axes[0], "sigma", "Per burst"), (axes[1], "sigma2h", "2-h average")):
    for r, c, name in (("20260728_073439", SHIZ, "Capsule A"), ("20260729_053037", AKA, "Capsule B")):
        for src, ls, lab in (("warp", "-", "warped"), ("raw", ":", "raw")):
            y = [d[r][f"{src}_{k}"][key] * 100 for k in ks]
            ax.plot(ks, y, ls, marker="o", color=c, lw=2.8, ms=7, label=f"{name}, {lab}")
            if key == "sigma" and src == "warp":
                lo = [d[r][f"warp_{k}"]["sigma_min"] * 100 for k in ks]
                hi = [d[r][f"warp_{k}"]["sigma_max"] * 100 for k in ks]
                ax.fill_between(ks, lo, hi, color=c, alpha=0.12, lw=0)
    ax.set_xticks(ks); ax.set_xticklabels(["1\n(4)", "2\n(8)", "3\n(12)", "4\n(16)"])
    ax.set_xlabel("Views per band (cameras in total)"); ax.set_title(title, fontsize=17)
    ax.grid(color=GRID); ax.set_ylim(0, None)
axes[0].set_ylabel("Measurement noise (%)")
axes[1].legend(frameon=False, fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/en_ablation_views.png"); plt.close(fig)
print("d done")
