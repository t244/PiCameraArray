"""Two new figures for the 11/12 deck, in the 9/4 deck's figure style."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401  (IPAexGothic)

M, OUT = sys.argv[1], sys.argv[2]          # /scratch/.../moisture, output dir
SHIZ, AKA, INK, MUTED, GRID = "#11709C", "#C3592C", "#222222", "#666666", "#DDDDDD"
A, B = "20260728_073439", "20260729_053037"
plt.rcParams.update({"font.size": 20, "axes.edgecolor": "#999999", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---- 1: floor against frames averaged per burst --------------------------------
fr = {r: json.load(open(f"{M}/frames90_{r}.json")) for r in (A, B)}
ns = [1, 2, 3, 5, 9, 15, 30, 45, 90]
fig, ax = plt.subplots(figsize=(10.8, 5.6), dpi=200)
for r, c, name, dy in ((A, SHIZ, "しずく", -0.7), (B, AKA, "赤", 0.7)):
    y = [fr[r][str(n)]["spread"] * 100 for n in ns]
    ax.plot(ns, y, "-o", color=c, lw=3.5, ms=9, mec="white", mew=1.5)
    ax.text(100, y[-1] + dy, f"{name}  {y[-1]:.1f}%", color=c, va="center", fontsize=20, fontweight="bold")
ax.axvline(6, color=MUTED, ls="--", lw=1.5)
ax.text(6.4, 13.2, "9月の解析（6コマ）", color=MUTED, fontsize=18)
ax.axvspan(30, 90, color="#EAE5EB", zorder=0)
ax.text(31, 13.2, "30コマ以上でほぼ頭打ち", color="#632E62", fontsize=18)
ax.set_xscale("log"); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns])
ax.set_xlim(0.9, 260); ax.set_ylim(0, 14.5)
ax.set_xlabel("1回の撮影で平均するコマ数"); ax.set_ylabel("1回の撮影のばらつき（%）")
ax.grid(axis="y", color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_frames.png"); plt.close(fig)

# ---- 2: inverted moisture against the reference, with leave-one-out points ------
ev = json.load(open(f"{M}/model_eval.json"))
fin = json.load(open("/workspace/code/output/final.json"))
fig, axes = plt.subplots(1, 2, figsize=(10.4, 6.2), dpi=200, sharey=True)
for ax, (r, c, name) in zip(axes, ((A, SHIZ, "しずく"), (B, AKA, "赤"))):
    e, ds, ref = ev[r], fin["dataset"][r], fin["reference"][r]
    H = np.array([h for h in ref["hours"] if h <= ds["cut"]]); Mr = np.array(ref["moisture"][:len(H)])
    pred = Mr + np.array(e["loo_errors"]["model"])
    ax.plot(H, Mr, "-", color="#BBBBBB", lw=2, zorder=1)
    ax.plot(H, Mr, "o", color=INK, ms=9, zorder=3, label="御社の実測")
    ax.plot(H, pred, "o", mfc="white", mec=c, mew=3, ms=14, zorder=2, label="画像による予測\n（その点を伏せて推定）")
    for h, a_, b_ in zip(H, Mr, pred):
        ax.plot([h, h], [a_, b_], color=c, lw=1.5, alpha=0.6, zorder=1)
    ax.set_title(f"{name}　誤差 {e['loo']['model']:.2f} %ポイント", color=c, fontsize=21, fontweight="bold")
    ax.set_xlabel("経過時間（時間）"); ax.grid(color=GRID)
    ax.set_xlim(0, 19)
axes[0].set_ylabel("含水率（%）"); axes[0].set_ylim(6, 28)
axes[0].legend(loc="lower left", fontsize=15, frameon=False)
axes[1].set_xlim(0, 15)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_estimate.png"); plt.close(fig)
print("figures written")
