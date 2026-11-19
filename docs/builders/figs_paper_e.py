"""Paper figures, round 2: spectral response (Fig. 2), mesh-removal principle (Fig. 3b),
model behaviour as one spectral-contrast plot (Fig. 6)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, FancyArrowPatch
sys.path.insert(0, "/workspace/code/analyze")
import moisture_model as MM

OUT = sys.argv[1]
fin = json.load(open("/workspace/code/output/final.json"))
INK, MUTED, GRID, DEEP = "#222222", "#666666", "#DDDDDD", "#632E62"
BAND_C = ["#D9A23A", "#C3592C", "#B0407E", "#5B4BB0"]
plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#999999", "xtick.color": MUTED, "ytick.color": MUTED})

# ---- Fig. 2: spectral response -----------------------------------------------------
wl = np.arange(400, 1101, 2.0)
# IMX296 monochrome relative response, read off the manufacturer's chart (approximate)
qe_pts = np.array([[400, .17], [450, .43], [500, .68], [550, .85], [600, .96], [650, 1.0], [700, .97],
                   [750, .89], [800, .72], [850, .51], [900, .35], [950, .21], [1000, .09], [1050, .04], [1100, .01]])
qe = np.interp(wl, qe_pts[:, 0], qe_pts[:, 1])
fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=220)
ax.axvspan(940, 1000, color="#7FC8DC", alpha=0.3, lw=0)
ax.text(970, 1.03, "water\n~970 nm", ha="center", va="bottom", fontsize=12, color="#2E7D93")
for c0, name, col in zip(fin["mumodel"]["cuton"].values(), fin["mumodel"]["grade"].values(), BAND_C):
    T = 0.9 / (1 + np.exp(-(wl - c0) / 6.0))          # nominal long-pass: 50 % at the stated cut-on
    ax.plot(wl, T, color=col, lw=2.6, label=f"{name} (cut-on {c0:.0f} nm)")
ax.plot(wl, qe, color=INK, lw=2.2, ls="--", label="Sensor response (IMX296, mono)")
ax.set_xlim(400, 1100); ax.set_ylim(0, 1.15)
ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Transmittance / relative response")
ax.grid(color=GRID); ax.legend(frameon=False, fontsize=11.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.tight_layout(); fig.savefig(f"{OUT}/en_spectral.png"); plt.close(fig)

# ---- Fig. 3(b): principle of synthetic-aperture mesh removal --------------------------
rng = np.random.default_rng(7)
fig = plt.figure(figsize=(13.5, 5.0), dpi=220)
# (i) geometry, side view
ax = fig.add_axes([0.02, 0.08, 0.36, 0.80]); ax.set_xlim(-30, 600); ax.set_ylim(-110, 110); ax.axis("off")
cams = [-60, -20, 20, 60]
for k, y in enumerate(cams):
    ax.add_patch(Rectangle((-28, y - 8), 22, 16, color="#4F6D8A"))
for y in np.arange(-105, 106, 7):                                  # mesh wires, exaggerated pitch
    ax.add_patch(Rectangle((98, y), 4, 3.2, color="#888888"))
ax.text(100, 102, "mesh", ha="center", fontsize=12, color=MUTED)
for (cy, cs) in ((-70, 12), (-35, 14), (0, 13), (35, 12), (70, 14)):
    ax.add_patch(Ellipse((540, cy), 22, cs * 2, color="#E4B66A", ec="#B07D2A"))
ax.text(540, 95, "capsules", ha="center", fontsize=12, color=MUTED)
for y, col in ((-60, "#11709C"), (60, "#C3592C")):
    ax.plot([-6, 540], [y, 0], color=col, lw=1.8)
    yc = y + (0 - y) * (100 + 6) / (540 + 6)
    ax.plot(100, yc, "o", color=col, ms=6)
ax.annotate("same capsule,\ndifferent mesh points", xy=(100, -48), xytext=(170, -95), fontsize=11.5, color=INK,
            arrowprops=dict(arrowstyle="->", color=MUTED))
ax.text(-17, -95, "camera\narray", ha="center", fontsize=12, color=MUTED)
ax.set_title("(i) Side view: the mesh is close,\nthe capsules are far", fontsize=13.5, color=DEEP)

# synthetic tiles for (ii) and (iii)
H = W = 120
yy, xx = np.mgrid[0:H, 0:W]
scene = np.zeros((H, W)) + 0.15
for _ in range(28):
    cx, cy, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(5, 8)
    scene += 0.85 * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)) ** 1.5)
scene = np.clip(scene, 0, 1)
def mesh(dx, dy, p=15, w=4):
    m = np.ones((H, W))
    m[((xx + dx) % p) < w] = 0.15
    m[((yy + dy) % p) < w] = 0.15
    return m
shifts = [(0, 0), (5, 2), (10, 6), (3, 10)]
for k, (dx, dy) in enumerate(shifts):
    a = fig.add_axes([0.41 + (k % 2) * 0.115, 0.47 - (k // 2) * 0.39, 0.105, 0.36])
    a.imshow(scene * mesh(dx, dy), cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
    a.set_xlabel(f"view {k + 1}", fontsize=11, color=MUTED, labelpad=2)
fig.text(0.525, 0.90, "(ii) Views registered on the capsule plane:\ncapsules align, the mesh does not",
         ha="center", fontsize=13.5, color=DEEP)
a = fig.add_axes([0.80, 0.22, 0.18, 0.52])
a.imshow(scene, cmap="gray", vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
fig.text(0.885, 0.90, "(iii) Mesh transmission\nestimated and divided out", ha="center", fontsize=12.5, color=DEEP)
fig.text(0.885, 0.10, "across views (bright envelope)\nand over time (the bed moves,\nthe mesh does not)",
         ha="center", va="top", fontsize=11.5, color=INK)
fig.patches.append(FancyArrowPatch((0.655, 0.48), (0.785, 0.48), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=28, color=DEEP, lw=2.5))
fig.savefig(f"{OUT}/en_principle.png"); plt.close(fig)

# ---- Fig. 6: model behaviour as one spectral-contrast curve per case --------------------
op = MM.Optics(fin)
m_hi = 25.0; mm = np.linspace(25, 8, 200); ln = np.log
cases = [("(a) water absorption only", [0.0, 0.0, ln(3), ln(1e-6), ln(1e-6), ln(1e-6), ln(1e-6), 0.0, -40, ln(40)], "#888888", "--"),
         ("(b) + wetting lengthens light paths", [0.0, ln(2.0), ln(3), ln(1e-6), ln(1e-6), ln(1e-6), ln(1e-6), 0.0, 14, ln(1.0)], "#11709C", "-"),
         ("(c) + shell dye changes with drying", [0.0, ln(2.0), ln(3), ln(0.25), ln(0.08), ln(0.02), ln(0.005), 0.10, 14, ln(1.0)], "#C3592C", "-")]
fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=220)
for name, th, col, ls in cases:
    P = op.channels(mm, np.array(th), m_hi)
    contrast = (P[3] - P[0]) - (P[3, -1] - P[0, -1])           # ln(NIR-90N / NIR-70N), relative to dry
    ax.plot(mm, contrast * 100, ls, color=col, lw=3, label=name)
ax.axvline(14, color=MUTED, ls=":", lw=1.5)
ax.text(13.6, -30, "m*: surface\nbecomes dry", fontsize=11.5, color=MUTED, va="center")
ax.invert_xaxis(); ax.grid(color=GRID)
ax.set_xlabel("Moisture content (%)   ← drying")
ax.set_ylabel("Change in ln( NIR-90N / NIR-70N ) (%)")
ax.legend(frameon=False, fontsize=12, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/en_model_contrast.png"); plt.close(fig)
print("e done")
