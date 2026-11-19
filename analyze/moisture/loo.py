"""Leave-one-out moisture prediction at the reference points, within each run."""
import json, sys, numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
fin = json.load(open("/workspace/code/output/final.json"))
B = F.BANDS
mu = np.array([fin["mumodel"]["mu"][b] for b in B]); c = mu - mu.mean(); u = c / np.linalg.norm(c)
files = {"20260728_073439": "levels_20260728_073439_f6.jsonl", "20260729_053037": "levels_20260729_053037_f6.jsonl"}
def loo(x, y):
    err = []
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        a = np.polyfit(x[k], y[k], 1); err.append(np.polyval(a, x[i]) - y[i])
    return float(np.sqrt(np.mean(np.square(err))))
for run, f in files.items():
    ds, ref = fin["dataset"][run], fin["reference"][run]
    rows = [r for r in F.load("/workspace/out/moisture/" + f) if (int(r["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
    t, L = F.levels(rows, "warp", "p95")
    X = L - L.mean(1, keepdims=True); X = X - X.mean(0, keepdims=True)
    w = u @ X
    H = np.array(ref["hours"]); M = np.array(ref["moisture"]); keep = H <= ds["cut"]
    H, M = H[keep], M[keep]
    for half in (0.25, 0.5):
        Wv = np.array([w[(t >= h - half) & (t <= h + half)].mean() for h in H])
        print(f"{run} ({ds['product']}), {len(H)} reference points, index averaged over +-{half*60:.0f} min:"
              f"  water {loo(Wv, M):.2f}   clock {loo(H, M):.2f}   mean-only {loo(np.zeros_like(H) + np.arange(len(H))*0, M) if False else float(np.sqrt(np.mean([(M[np.arange(len(M))!=i].mean()-M[i])**2 for i in range(len(M))]))):.2f}"
              f"   (moisture points RMSE)   r(water, moisture) {np.corrcoef(Wv, M)[0,1]:+.3f}")
