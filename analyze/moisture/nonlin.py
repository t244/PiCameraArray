"""How curved is the spectral response against moisture? 30-min bins."""
import json, sys, numpy as np
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
exec(open("/workspace/re/learned.py").read().split("def index_at")[0])   # data, learn_dir
def bins(t, y, m):
    e = np.arange(0, t.max() + .5, .5); i = np.digitize(t, e); ks = [k for k in np.unique(i) if (i == k).sum() > 3]
    return np.array([m[i == k].mean() for k in ks]), np.array([y[i == k].mean() for k in ks]), np.array([y[i == k].std() / np.sqrt((i == k).sum()) for k in ks])
def fits(m, y):
    out = {}
    for deg in (1, 2):
        r = y - np.polyval(np.polyfit(m, y, deg), m); out[f"poly{deg}"] = (r @ r, deg + 1)
    best = None
    for knot in np.linspace(np.quantile(m, .15), np.quantile(m, .85), 40):
        A = np.c_[np.ones_like(m), m, np.maximum(m - knot, 0)]
        r = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
        if best is None or r @ r < best[0]: best = (r @ r, knot)
    out["broken"] = (best[0], 4); out["knot"] = best[1]
    return out
for run, D in data.items():
    u = learn_dir(D["t"], D["X"], D["m"])
    L_bright = None
    for name, y in (("learned index", u @ D["X"]),):
        m, yb, se = bins(D["t"], y, D["m"]); n = len(m)
        f = fits(m, yb)
        aic = {k: n * np.log(v[0] / n) + 2 * v[1] for k, v in f.items() if k != "knot"}
        print(f"{D['product']:8s} {name}: n={n}  rss lin {f['poly1'][0]:.2e}  quad {f['poly2'][0]:.2e}  broken {f['broken'][0]:.2e} (knot {f['knot']:.1f}%)"
              f"  AIC lin {aic['poly1']:.1f} quad {aic['poly2']:.1f} broken {aic['broken']:.1f}   typical bin s.e. {np.median(se):.4f}")
# brightness (common mode) against moisture, raw p95 of the warped 4-view means
files = {"20260728_073439": "levels_20260728_073439_f6.jsonl", "20260729_053037": "levels_20260729_053037_f6.jsonl"}
for run, f in files.items():
    ds, ref = fin["dataset"][run], fin["reference"][run]
    rows = [r for r in F.load("/workspace/out/moisture/" + f) if (int(r["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
    t, L = F.levels(rows, "warp", "p95"); m = F.moisture_at(t, ref)
    mb, yb, _ = bins(t, L.mean(0), m); f_ = fits(mb, yb); n = len(mb)
    aic = {k: n * np.log(v[0] / n) + 2 * v[1] for k, v in f_.items() if k != "knot"}
    print(f"{ds['product']:8s} brightness: AIC lin {aic['poly1']:.1f} quad {aic['poly2']:.1f} broken {aic['broken']:.1f} (knot {f_['knot']:.1f}%)")
