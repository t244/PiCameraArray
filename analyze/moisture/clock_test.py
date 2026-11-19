"""Does the water index follow the moisture CURVE, or just the clock?"""
import json, sys, numpy as np
for path in sys.argv[1:]:
    rep = json.load(open(path))
    for run, r in rep["runs"].items():
        s = r["series"]; t = np.array(s["hours"]); w = np.array(s["water"]); m = np.array(s["moisture"])
        z = np.array(s["null"]); br = np.array(s["bright"])
        def c(a, b): a, b = a - a.mean(), b - b.mean(); return (a*b).sum() / np.sqrt((a*a).sum()*(b*b).sum())
        def resid(y, *xs):
            X = np.column_stack([np.ones_like(y), *xs]); return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        # 30-min means, so burst scatter does not hide the shape
        e = np.arange(0, t.max() + .5, .5); i = np.digitize(t, e)
        g = lambda x: np.array([x[i == k].mean() for k in np.unique(i)])
        T, W, M, Z0, Z1, B = g(t), g(w), g(m), g(z[0]), g(z[1]), g(br)
        print(f"{run}: 30-min bins {len(T)}")
        print(f"  r(water, moisture) {c(W, M):+.3f}   r(water, time) {c(W, T):+.3f}   r(moisture, time) {c(M, T):+.3f}")
        print(f"  water vs moisture with time held fixed (partial) {c(resid(W, T), resid(M, T)):+.3f}")
        print(f"  null0/null1 vs moisture {c(Z0, M):+.3f} {c(Z1, M):+.3f}   brightness vs moisture {c(B, M):+.3f}")
        for k in ("water", "null0", "null1"):
            y = {"water": W, "null0": Z0, "null1": Z1}[k]
            a = np.polyfit(M, y, 1)[0]; print(f"  slope {k} per moisture pt {a:+.5f}")
