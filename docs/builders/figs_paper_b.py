"""Figures and equation images for the model appendix of the 11/12 deck."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/code/analyze")
import moisture_model as MM

M, OUT = sys.argv[1], sys.argv[2]
SHIZ, AKA, INK, MUTED, GRID, DEEP, VIO = "#11709C", "#C3592C", "#222222", "#666666", "#DDDDDD", "#632E62", "#92278F"
BAND_C = ["#D9A23A", "#C3592C", "#B0407E", "#5B4BB0"]          # 750 800 850 900, as the 9/4 filter figure
A, B = "20260728_073439", "20260729_053037"
plt.rcParams.update({"font.size": 18, "axes.edgecolor": "#999999", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "mathtext.fontset": "cm"})


def eq_image(lines, name, size=30, width=11.0, gap=0.95):
    fig = plt.figure(figsize=(width, gap * len(lines) + 0.2), dpi=220)
    for k, (tex, color) in enumerate(lines):
        fig.text(0.01, 1 - (k + 0.6) / len(lines), tex, fontsize=size, color=color, va="center")
    fig.savefig(f"{OUT}/{name}", transparent=True)
    plt.close(fig)


# ---- equations -------------------------------------------------------------------
eq_image([(r"$I(\lambda)\;=\;I_0\,\sum_{n}\;\tau_n(g_w)\;a(\lambda)^{\,n}\;b(\lambda)^{\,\gamma n}$", INK),
          (r"$a(\lambda)=e^{-u(\lambda)\,r_0},\qquad b(\lambda)=e^{-v(\lambda)\,r_0},\qquad g_w=g_d^{\,1-\gamma}\,g_s^{\,\gamma}$", MUTED)],
         "en_eq_shimano.png", size=30, width=10.6, gap=1.1)
eq_image([(r"$\mathrm{(1)}\quad I_j\;=\;\exp\!\left[-\,\bar n(\gamma)\,\left(1-e^{-r_0\,\left(u_j(m)+\gamma\,v_j\right)}\right)\right]$", INK),
          (r"$\mathrm{(2)}\quad \bar n(\gamma)\;=\;\bar n_d\,\left(\bar n_s/\bar n_d\right)^{\gamma}$", INK),
          (r"$\mathrm{(3)}\quad u_j(m)\;=\;u_j^{0}\,\left(1+\kappa\,(m_{\mathrm{hi}}-m)\right)$", INK),
          (r"$\mathrm{(4)}\quad \gamma(m)\;=\;\mathrm{softplus}\!\left(\frac{m-m^{*}}{w}\right)\;/\;\mathrm{softplus}\!\left(\frac{m_{\mathrm{hi}}-m^{*}}{w}\right)$", INK),
          (r"$\mathrm{(5)}\quad C_k\;=\;\sum_{j\geq k}\,s_{kj}\,I_j\qquad(k=1..4:\ 750,\,800,\,850,\,900)$", INK)],
         "en_eq_model.png", size=26, width=10.4, gap=1.05)

# ---- schematic: light paths in a dry and a wet surface -------------------------------
rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=220)
for ax, wet, title in ((axes[0], False, "Dry surface"), (axes[1], True, "Wet surface")):
    ax.set_xlim(0, 10); ax.set_ylim(-7.4, 2.2); ax.axis("off")
    ax.fill_between([0, 10], -6.2, 0, color="#BFE3EE" if wet else "#F1ECE4", zorder=0)
    ax.plot([0, 10], [0, 0], color="#999999", lw=1.5)
    px, py = rng.uniform(0.3, 9.7, 60), rng.uniform(-6.0, -0.3, 60)
    ax.scatter(px, py, s=60, color="#D8CFC2" if not wet else "#A9C9D3", edgecolor="#8C8073" if not wet else "#5E8C99", lw=0.8, zorder=1)
    # one illustrative path: forward-biased steps when wet, short and turning when dry
    p = np.array([3.0, 0.0]); path = [p.copy()]
    ang = -np.pi / 2
    steps = 9 if wet else 5
    for k in range(steps):
        ang += rng.normal(0, 0.45 if wet else 1.1)
        if k == steps - 3:
            ang = np.pi / 2 - 0.3 if wet else np.pi / 2
        L = 1.2 if wet else 0.9
        p = p + L * np.array([np.cos(ang), np.sin(ang)])
        p[1] = min(max(p[1], -5.6), -0.2) if k < steps - 1 else 0.0
        path.append(p.copy())
    path = np.array(path)
    ax.plot([2.0, 3.0], [2.0, 0.0], color="#E3A21A", lw=3)
    ax.annotate("", xy=(3.0, 0.05), xytext=(2.0, 2.0), arrowprops=dict(arrowstyle="-|>", color="#E3A21A", lw=3))
    ax.plot(path[:, 0], path[:, 1], "-", color=VIO, lw=2.5, zorder=3)
    ax.scatter(path[1:-1, 0], path[1:-1, 1], s=55, color=VIO, zorder=4)
    ax.annotate("", xy=(path[-1, 0] + 0.9, 1.9), xytext=(path[-1, 0], 0.0),
                arrowprops=dict(arrowstyle="-|>", color=VIO, lw=3, alpha=0.4 if wet else 0.9))
    ax.text(5, -7.3, f"{steps - 1} scattering events, shallow" if not wet else f"{steps - 1} scattering events, deep", ha="center", va="bottom",
            fontsize=17, color=INK)
    ax.set_title(title, fontsize=21, color=DEEP, fontweight="bold")
pass
fig.tight_layout(); fig.savefig(f"{OUT}/en_paths.png"); plt.close(fig)

# ---- model behaviour with illustrative coefficients: what each term does ------------
fin = json.load(open("/workspace/code/output/final.json"))
op = MM.Optics(fin)
ev = json.load(open(f"{M}/model_eval.json"))
names = ["750 nm", "800 nm", "850 nm", "900 nm"]
m_hi = 25.0
mm = np.linspace(25, 8, 200)
ln = np.log
#            ln nbar_d, ln ratio, ln r0, ln u0 (4 sub-bands)                    kappa, m*,  ln w
cases = [("(a) Water absorption only",            [0.0, 0.0,     ln(3), ln(1e-6), ln(1e-6), ln(1e-6), ln(1e-6), 0.0,  -40, ln(40)]),
         ("(b) + wetting lengthens paths",  [0.0, ln(2.0), ln(3), ln(1e-6), ln(1e-6), ln(1e-6), ln(1e-6), 0.0,  14,  ln(1.0)]),
         ("(c) + dye concentrates on drying",  [0.0, ln(2.0), ln(3), ln(0.25), ln(0.08), ln(0.02), ln(0.005), 0.10, 14,  ln(1.0)])]
fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2), dpi=200)
prev = None
for ax, (title, th) in zip(axes, cases):
    P = op.channels(mm, np.array(th), m_hi)
    P = P - P[:, -1:]                              # relative to the driest point
    if title.startswith("③") and prev is not None:
        for k in range(4):
            ax.plot(mm, prev[k] * 100, lw=2, ls="--", color=BAND_C[k], alpha=0.45)
        ax.text(0.97, 0.97, "dashed: (b), no dye change", transform=ax.transAxes, ha="right", va="top", fontsize=14, color=MUTED)
    for k in range(4):
        ax.plot(mm, P[k] * 100, lw=3.2, color=BAND_C[k], label=names[k])
    prev = P
    ax.invert_xaxis(); ax.grid(color=GRID)
    ax.set_title(title, fontsize=20, color=DEEP, fontweight="bold")
    ax.set_xlabel("Moisture content (%)  ← drying")
    if th[8] > 0:
        ax.axvline(th[8], color=MUTED, ls="--", lw=1.5)
        ax.text(th[8] - 0.3, ax.get_ylim()[1] if False else 0, "", fontsize=1)
axes[0].set_ylabel("Change in log intensity (%), rel. to dry")
for ax in axes: ax.set_ylabel(ax.get_ylabel())
axes[0].legend(fontsize=14, frameon=False, loc="lower left", ncol=2)
for ax in axes[1:]:
    ax.text(13.6, 0.02, "m*", transform=ax.get_xaxis_transform(), fontsize=16, color=MUTED)
fig.tight_layout(); fig.savefig(f"{OUT}/en_model_concept.png"); plt.close(fig)

print("appendix figures written")
