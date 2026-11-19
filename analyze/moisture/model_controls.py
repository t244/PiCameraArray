"""Controls for moisture_model.py: does the gain come from the spectra or from the
monotone prior? And a fairer clock (exponential drying curve)."""
import json, sys, numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, "/workspace/code/analyze")
import moisture_model as MM
fin = json.load(open("/workspace/code/output/final.json"))
op = MM.Optics(fin)
files = {"20260728_073439": "levels_20260728_073439_f6.jsonl", "20260729_053037": "levels_20260729_053037_f6.jsonl"}
def exp_clock(H, M, i):
    Hk, Mk = np.delete(H, i), np.delete(M, i)
    f = lambda t, a, b, tau: a + b * np.exp(-t / tau)
    try:
        p, _ = curve_fit(f, Hk, Mk, p0=[Mk.min(), Mk.max() - Mk.min(), 6], maxfev=20000)
        return f(H[i], *p) - M[i]
    except RuntimeError:
        return np.nan
for s, fn in files.items():
    R = MM.load_run("/workspace/out/moisture/" + fn, fin, s)
    H, M = R["H"], R["M"]
    e = [exp_clock(H, M, i) for i in range(len(H))]
    print(f"{R['product']:8s} exponential clock LOO {np.sqrt(np.nanmean(np.square(e))):.2f}")
    rng = np.random.default_rng(1)
    vals = []
    for rep in range(3):
        P = dict(R); perm = rng.permutation(R["X"].shape[1]); P["X"] = R["X"][:, perm]
        rmse, _ = MM.evaluate(op, P, starts=8)
        vals.append(rmse["model"])
        print(f"{R['product']:8s} time-shuffled spectra, rep {rep}: model LOO {rmse['model']:.2f}")
    print(f"{R['product']:8s} shuffled mean {np.mean(vals):.2f}")
