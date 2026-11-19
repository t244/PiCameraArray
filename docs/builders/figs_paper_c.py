"""Two more English figures: reference moisture curves, nested long-pass bands."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = sys.argv[2]
fin = json.load(open("/workspace/code/output/final.json"))
SHIZ, AKA, MUTED, GRID = "#11709C", "#C3592C", "#666666", "#DDDDDD"
plt.rcParams.update({"font.size": 17, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=220)
for r, c, name in (("20260728_073439", SHIZ, "Capsule A"), ("20260729_053037", AKA, "Capsule B")):
    ref = fin["reference"][r]; ds = fin["dataset"][r]
    ax.plot(ref["hours"], ref["moisture"], "-o", color=c, lw=3, ms=8, mec="white", label=name)
    ax.axvline(ds["cloth"], color=c, ls="--", lw=1.5)
    first = r.startswith("20260728")
    ax.text(ds["cloth"] + (-0.2 if first else 0.2), 23.5 if first else 21.8, "oil cloth\nin", color=c, fontsize=13,
            ha="right" if first else "left")
ax.set_xlabel("Elapsed drying time (h)"); ax.set_ylabel("Shell moisture, LOD (%)")
ax.grid(color=GRID); ax.legend(frameon=False, fontsize=14, loc="lower left")
ax.set_xlim(0, 23); fig.tight_layout(); fig.savefig(f"{OUT}/en_moisture.png"); plt.close(fig)

mu = fin["mumodel"]
cut = [670, 760, 810, 860]; names = ["NIR-70N", "NIR-80N", "NIR-85N", "NIR-90N"]
cols = ["#D9A23A", "#C3592C", "#B0407E", "#5B4BB0"]
fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=220)
for k, (c0, n, col) in enumerate(zip(cut, names, cols)):
    ax.barh(3 - k, 1100 - c0, left=c0, height=0.6, color=col, alpha=0.85)
    ax.text(c0 - 8, 3 - k, n, ha="right", va="center", fontsize=14, color=col, fontweight="bold")
    ax.text(c0 + 10, 3 - k, f"> {c0} nm,  μ_eff {mu['mu'][str(750 + 50 * k)]:.3f}",
            va="center", fontsize=12, color="white")
ax.axvspan(940, 1000, color="#7FC8DC", alpha=0.35, zorder=0)
ax.text(970, 3.55, "water band\n(~970 nm)", ha="center", fontsize=12, color="#2E7D93")
ax.set_xlim(560, 1100); ax.set_ylim(-0.6, 4.1); ax.set_yticks([])
ax.set_xlabel("Wavelength (nm)  (sensor response ends near 1000–1100 nm)")
for sp in ("left",): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/en_filters.png"); plt.close(fig)
print("c done")
