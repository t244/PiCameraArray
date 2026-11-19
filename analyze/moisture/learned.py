"""Process model: learn the spectral direction that tracks the reference moisture
(oil, dye and water together), leak-safe leave-one-out at the reference points."""
import json, sys, numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
fin = json.load(open("/workspace/code/output/final.json"))
B = F.BANDS
mu = np.array([fin["mumodel"]["mu"][b] for b in B]); c = mu - mu.mean(); u_phys = c / np.linalg.norm(c)
files = {"20260728_073439": "levels_20260728_073439_f6.jsonl", "20260729_053037": "levels_20260729_053037_f6.jsonl"}
data = {}
for run, f in files.items():
    ds, ref = fin["dataset"][run], fin["reference"][run]
    rows = [r for r in F.load("/workspace/out/moisture/" + f) if (int(r["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
    t, L = F.levels(rows, "warp", "p95")
    X = L - L.mean(1, keepdims=True); X = X - X.mean(0, keepdims=True)
    H = np.array(ref["hours"]); M = np.array(ref["moisture"]); k = H <= ds["cut"]
    data[run] = dict(t=t, X=X, m=F.moisture_at(t, ref), H=H[k], M=M[k], product=ds["product"])

def learn_dir(t, X, m, exclude=()):
    keep = np.ones(len(t), bool)
    for h in exclude:
        keep &= np.abs(t - h) > 1.0
    Xk, mk = X[:, keep], m[keep] - m[keep].mean()
    Xk = Xk - Xk.mean(1, keepdims=True)
    d = Xk @ mk / (mk @ mk)                       # spectral change per moisture point
    return d / np.linalg.norm(d)

def index_at(run, u, H, half=0.5):
    D = data[run]; w = u @ D["X"]
    return np.array([w[(D["t"] >= h - half) & (D["t"] <= h + half)].mean() for h in H])

def loo(run, dir_fn):
    D = data[run]; H, M = D["H"], D["M"]; err = []
    for i in range(len(H)):
        u = dir_fn(i)
        x = index_at(run, u, H)
        k = np.arange(len(H)) != i
        a = np.polyfit(x[k], M[k], 1); err.append(np.polyval(a, x[i]) - M[i])
    return float(np.sqrt(np.mean(np.square(err))))

def clock(run):
    D = data[run]; H, M = D["H"], D["M"]
    return float(np.sqrt(np.mean([(np.polyval(np.polyfit(np.delete(H, i), np.delete(M, i), 1), H[i]) - M[i]) ** 2 for i in range(len(H))])))

runs = list(files)
full = {r: learn_dir(data[r]["t"], data[r]["X"], data[r]["m"]) for r in runs}
print("learned directions (unit, 750 800 850 900):")
for r in runs:
    print(f"  {data[r]['product']:8s} " + " ".join(f"{x:+.3f}" for x in full[r]) + f"   cos with physical water {full[r] @ u_phys:+.3f}")
print(f"  cos between the two learned directions {full[runs[0]] @ full[runs[1]]:+.3f}")
both = learn_dir(np.r_[data[runs[0]]["t"], data[runs[1]]["t"] + 100],
                 np.c_[data[runs[0]]["X"], data[runs[1]]["X"]], np.r_[data[runs[0]]["m"], data[runs[1]]["m"]])
print("\nleave-one-out RMSE at the reference points (moisture points); gain/offset refit per run each time:")
for r in runs:
    other = runs[1] if r == runs[0] else runs[0]
    D = data[r]
    print(f"  {D['product']:8s} ({len(D['H'])} pts)"
          f"  physical water {loo(r, lambda i: u_phys):.2f}"
          f" | own learned (leak-safe) {loo(r, lambda i: learn_dir(D['t'], D['X'], D['m'], exclude=[D['H'][i]])):.2f}"
          f" | other run's direction {loo(r, lambda i: full[other]):.2f}"
          f" | pooled-runs direction {loo(r, lambda i: both):.2f}"
          f" | clock {clock(r):.2f}")
