"""Paper figures, round 3: Fig. 2 without the water band; an illustrated Fig. 3(b)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse, Circle, FancyArrowPatch, Rectangle, PathPatch, Arc
from matplotlib.path import Path

OUT = sys.argv[1]
fin = json.load(open("/workspace/code/output/final.json"))
INK, MUTED, GRID, DEEP = "#222222", "#666666", "#DDDDDD", "#632E62"
BAND_C = ["#D9A23A", "#C3592C", "#B0407E", "#5B4BB0"]
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "xtick.color": MUTED, "ytick.color": MUTED})

# ---- Fig. 2 --------------------------------------------------------------------------
wl = np.arange(400, 1001, 2.0)
qe_pts = np.array([[400, .17], [450, .43], [500, .68], [550, .85], [600, .96], [650, 1.0], [700, .97],
                   [750, .89], [800, .72], [850, .51], [900, .35], [950, .21], [1000, .09]])
qe = np.interp(wl, qe_pts[:, 0], qe_pts[:, 1])
fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=220)
for c0, name, col in zip(fin["mumodel"]["cuton"].values(), fin["mumodel"]["grade"].values(), BAND_C):
    ax.plot(wl, 0.9 / (1 + np.exp(-(wl - c0) / 6.0)), color=col, lw=2.6, label=f"{name} (cut-on {c0:.0f} nm)")
ax.plot(wl, qe, color=INK, lw=2.2, ls="--", label="Sensor response (IMX296, mono)")
ax.set_xlim(400, 1000); ax.set_ylim(0, 1.08)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Transmittance / relative response")
ax.grid(color=GRID); ax.legend(frameon=False, fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/en_spectral.png"); plt.close(fig)

# ---- Fig. 3(b) ------------------------------------------------------------------------
rng = np.random.default_rng(11)
STEEL, STEEL_D, CAP1, CAP2, BED = "#8E99A4", "#5C6670", "#E9A23B", "#C8572C", "#3A2A20"
fig = plt.figure(figsize=(14.0, 5.0), dpi=220)

# (i) illustration: drum with its axis into the page, near end plate cut away ------------------
ax = fig.add_axes([0.0, 0.0, 0.44, 1.0]); ax.set_xlim(0, 10.2); ax.set_ylim(0, 6.6); ax.axis("off"); ax.set_aspect("equal")
cx, cy, R, off = 6.9, 3.1, 2.5, (0.75, 0.45)
bx_, by_ = cx + off[0], cy + off[1]
# back end plate and the cylinder silhouette give the depth
ax.add_patch(Circle((bx_, by_), R, fc="#AEB9C3", ec=STEEL_D, lw=1.2, zorder=1))
for sgn in (1, -1):
    ang = np.arctan2(off[1], off[0]) + sgn * np.pi / 2
    ax.plot([cx + R * np.cos(ang), bx_ + R * np.cos(ang)], [cy + R * np.sin(ang), by_ + R * np.sin(ang)],
            color=STEEL_D, lw=1.4, zorder=1)
ax.add_patch(Circle((cx, cy), R, fc="#E3E8ED", ec="none", zorder=1))
# capsule bed, lifted towards the rising side by the rotation
for _ in range(95):
    a = rng.uniform(np.pi * 0.95, np.pi * 1.62); rr = R * np.sqrt(rng.uniform(0.35, 0.93))
    x, y = cx + rr * np.cos(a), cy + rr * np.sin(a)
    ax.add_patch(Ellipse((x, y), 0.36, 0.21, angle=rng.uniform(0, 180), fc=rng.choice([CAP1, CAP2]),
                         ec="#7A4A1A", lw=0.5, zorder=2))
# the cylindrical mesh wall, drawn as a ring of wires
th = np.linspace(0, 2 * np.pi, 400)
ax.plot(cx + R * np.cos(th), cy + R * np.sin(th), color=STEEL_D, lw=3.2, zorder=3)
ax.plot(cx + (R - 0.12) * np.cos(th), cy + (R - 0.12) * np.sin(th), color=STEEL, lw=1.2, zorder=3)
for a in np.linspace(0, 2 * np.pi, 72, endpoint=False):
    ax.plot([cx + (R - 0.14) * np.cos(a), cx + (R + 0.04) * np.cos(a)],
            [cy + (R - 0.14) * np.sin(a), cy + (R + 0.04) * np.sin(a)], color=STEEL_D, lw=1.0, zorder=3)
ax.add_patch(FancyArrowPatch((cx + 1.2, cy + 1.55), (cx - 1.2, cy + 1.55), connectionstyle="arc3,rad=0.45",
                             arrowstyle="-|>", mutation_scale=20, color=DEEP, lw=2.4, zorder=6))
ax.text(cx + 0.35, cy + R + 0.75, "rotating tumble dryer\n(near end plate cut away)", ha="center", fontsize=12, color=INK)
ax.text(cx + 0.9, cy - R - 0.5, "steel mesh wall", ha="center", fontsize=11.5, color=STEEL_D)
# camera array: a box whose lens face points at the drum; seen obliquely, the face is a parallelogram
P0 = np.array([2.3, 1.4]); U = np.array([0.95, 0.45]); V = np.array([0.0, 2.8]); D = np.array([-1.6, 0.22])
face = [P0, P0 + U, P0 + U + V, P0 + V]
ax.add_patch(Polygon([P0 + V, P0 + U + V, P0 + U + V + D, P0 + V + D], fc="#A9B6C3", ec="#46525E", lw=1.2, zorder=6))  # top
ax.add_patch(Polygon([P0, P0 + V, P0 + V + D, P0 + D], fc="#7D8C9B", ec="#46525E", lw=1.2, zorder=6))                  # side
ax.add_patch(Polygon(face, fc="#C4CFD9", ec="#46525E", lw=1.4, zorder=6))                                              # lens face
lens = {}
ang = np.degrees(np.arctan2(U[1], U[0]))
for r in range(4):
    for c in range(4):
        uu, vv = 0.14 + c * 0.24, 0.88 - r * 0.25
        L = P0 + uu * U + vv * V
        ax.add_patch(Ellipse(L, 0.26, 0.42, angle=ang - 25, fc="#1E242A", ec="#0B0E11", lw=1, zorder=7))
        ax.add_patch(Ellipse(L + np.array([0.02, 0.0]), 0.16, 0.26, angle=ang - 25, fc="#3E5A78", ec="none", zorder=8))
        ax.add_patch(Ellipse(L + np.array([0.05, 0.06]), 0.05, 0.07, angle=ang - 25, fc="white", ec="none", alpha=0.8, zorder=9))
        lens[(r, c)] = tuple(L)
ax.text(1.9, 0.95, "4 × 4 camera array", ha="center", fontsize=12.5, color=INK)
# two rays to the same capsule; each crosses the mesh wall at its own point
target = (cx - 1.55, cy - 0.85)
ax.add_patch(Ellipse(target, 0.46, 0.27, fc=CAP1, ec="black", lw=1.3, zorder=10))
for (r, c), col in (((0, 2), "#11709C"), ((3, 1), "#C3592C")):
    x0, y0 = lens[(r, c)]
    ax.plot([x0, target[0]], [y0, target[1]], color=col, lw=2.2, zorder=9)
    d = np.array(target) - np.array([x0, y0]); p0 = np.array([x0, y0]) - np.array([cx, cy])
    A, Bq, C = d @ d, 2 * p0 @ d, p0 @ p0 - R * R
    t = (-Bq - np.sqrt(Bq * Bq - 4 * A * C)) / (2 * A)          # first crossing of the wall
    ax.add_patch(Circle((x0 + t * d[0], y0 + t * d[1]), 0.1, fc=col, ec="white", lw=1.2, zorder=11))
ax.annotate("same capsule, seen through\ntwo different points of the mesh", xy=(cx - R + 0.05, cy - 0.35),
            xytext=(3.6, 0.35), fontsize=11, color=INK, ha="center",
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
ax.text(0.15, 6.25, "(i)", fontsize=14, color=DEEP, fontweight="bold")

# (ii) and (iii): coloured synthetic views ------------------------------------------------
H = W = 120
yy, xx = np.mgrid[0:H, 0:W]
bed = np.zeros((H, W, 3)) + np.array([0.23, 0.16, 0.12])
for _ in range(30):
    c0, c1, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(5.5, 8.5)
    col = np.array([0.91, 0.64, 0.23]) if rng.random() < 0.55 else np.array([0.78, 0.34, 0.17])
    g = np.exp(-(((xx - c0) ** 2 + ((yy - c1) * 1.4) ** 2) / (2 * r ** 2)) ** 1.6)[..., None]
    spec = np.exp(-(((xx - c0 + 2) ** 2 + (yy - c1 + 2) ** 2) / 3.0))[..., None]
    bed = bed * (1 - g) + col * g + 0.5 * spec
bed = np.clip(bed, 0, 1)
def meshed(dx, dy, p=15, w=4):
    img = bed.copy()
    wire = np.array([0.55, 0.60, 0.65])
    mx = ((xx + dx) % p) < w; my = ((yy + dy) % p) < w
    for m, shade in ((mx, 1.0), (my, 0.85)):
        img[m] = wire * shade
    return img
shifts = [(0, 0), (5, 2), (10, 6), (3, 10)]
x0f, y0f, wf, gap = 0.465, 0.06, 0.122, 0.012
for k, (dx, dy) in enumerate(shifts):
    a = fig.add_axes([x0f + (k % 2) * (wf + gap), y0f + (1 - k // 2) * (0.38 + 0.03), wf, 0.38])
    a.imshow(meshed(dx, dy)); a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_color("#46525E")
    a.text(4, 12, f"view {k + 1}", color="white", fontsize=10.5, fontweight="bold")
fig.text(x0f, 0.93, "(ii)", fontsize=14, color=DEEP, fontweight="bold")
fig.text(x0f + 0.028, 0.93, "views registered on the capsules:\ncapsules align, the mesh does not", fontsize=12, color=INK, va="center")
fig.patches.append(FancyArrowPatch((0.735, 0.47), (0.785, 0.47), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=26, color=DEEP, lw=3))
a = fig.add_axes([0.795, 0.2, 0.17, 0.58])
a.imshow(bed); a.set_xticks([]); a.set_yticks([])
for sp in a.spines.values(): sp.set_color("#46525E")
fig.text(0.795, 0.93, "(iii)", fontsize=14, color=DEEP, fontweight="bold")
fig.text(0.83, 0.93, "mesh removed", fontsize=12, color=INK, va="center")
fig.text(0.88, 0.07, "mesh transmission estimated across\nviews and over time, divided out",
         ha="center", fontsize=10.5, color=MUTED)
fig.savefig(f"{OUT}/en_principle.png"); plt.close(fig)
print("f done")
