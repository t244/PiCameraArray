"""The spectral direction each run actually moved in, per moisture point."""
import json, sys, numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
fin = json.load(open("/workspace/code/output/final.json"))
B = F.BANDS
mu = np.array([fin["mumodel"]["mu"][b] for b in B]); c = mu - mu.mean()
pl = np.array([0.0] + [fin["lamp"]["planck"][b] for b in B[1:]]); pl = pl - pl.mean()
unit = lambda v: v / np.linalg.norm(v)
for run, path in (("20260728_073439", sys.argv[1]), ("20260729_053037", sys.argv[2])):
    ds, ref = fin["dataset"][run], fin["reference"][run]
    rows = [r for r in F.load(path) if (int(r["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
    t, L = F.levels(rows, "warp", "p95")
    X = L - L.mean(1, keepdims=True); X = X - X.mean(0, keepdims=True)
    m = F.moisture_at(t, ref)
    d = np.array([np.polyfit(m, x, 1)[0] for x in X])          # per band, per moisture point
    resid = X - np.outer(d, m - m.mean())
    print(f"{run} ({fin['dataset'][run]['product']}): d ln I / d moisture = " +
          "  ".join(f"{b} {v*100:+.3f}%" for b, v in zip(B, d)))
    print(f"   |d| {np.linalg.norm(d)*100:.3f}%/pt   cos(d, water) {unit(d) @ unit(-c):+.3f}   "
          f"cos(d, lamp) {unit(d) @ unit(pl):+.3f}   cos(water, lamp) {unit(-c) @ unit(pl):+.3f}")
    # what water path change would explain d, and how much is left over
    a = (d @ -c) / (c @ c)
    print(f"   water fit: {a:.4f} per pt, unexplained {np.linalg.norm(d + a * c) / np.linalg.norm(d) * 100:.0f}% of |d|")
