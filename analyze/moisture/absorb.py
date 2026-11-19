"""Can a per-product material-absorption term (Shimano et al. 2017, eq. 7,
linearised) absorb the difference between the two runs?"""
import json, sys, numpy as np
from scipy.optimize import lsq_linear
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
fin = json.load(open("/workspace/code/output/final.json"))
B = F.BANDS
sub = fin["mumodel"]["bands"]; names = list(sub)
v = np.array([sub[n]["mu"] for n in names]); sh = np.array([sub[n]["share"] for n in names])
W = np.zeros((4, 4))                       # channel k integrates sub-bands j >= k
for k in range(4):
    W[k, k:] = sh[k:] / sh[k:].sum()
mu = np.array([fin["mumodel"]["mu"][b] for b in B])
assert np.allclose(W @ v, mu, rtol=1e-3), (W @ v, mu)
P = np.eye(4) - np.ones((4, 4)) / 4        # removes the common (illumination) term
lamp = np.array([0.0] + [fin["lamp"]["planck"][b] for b in B[1:]])
rep = json.load(open("/workspace/out/moisture/report_AB_f6.json"))
files = {"20260728_073439": "levels_20260728_073439_f6.jsonl", "20260729_053037": "levels_20260729_053037_f6.jsonl"}

def binned(run):
    ds, ref = fin["dataset"][run], fin["reference"][run]
    rows = [r for r in F.load("/workspace/out/moisture/" + files[run]) if (int(r["burst"][5:]) - 1) * 120 / 3600 <= ds["cut"]]
    t, L = F.levels(rows, "warp", "p95")
    X = P @ (L - L.mean(1, keepdims=True))
    m = F.moisture_at(t, ref)
    e = np.arange(0, t.max() + .5, .5); i = np.digitize(t, e); ks = np.unique(i)
    return (np.array([t[i == k].mean() for k in ks]), np.array([m[i == k].mean() for k in ks]),
            np.array([X[:, i == k].mean(1) for k in ks]).T)

out = {}
for run in files:
    t, m, X = binned(run)                   # X: 4 x nbins, common removed
    mc = m - m.mean()
    d = X @ mc / (mc @ mc)                  # d ln I / d moisture, per band
    res = X - np.outer(d, mc)
    se = np.sqrt((res ** 2).sum(1) / (len(m) - 2) / (mc @ mc))   # per-band slope s.e.
    floor30 = rep["runs"][run]["configs"]["warped, 4-view mean"]["sigma@30min"]
    print(f"\n== {run} ({fin['dataset'][run]['product']}), {len(m)} half-hour bins, moisture {m.max():.1f} -> {m.min():.1f}")
    print("  observed d ln I/d m (%/pt):", " ".join(f"{x*100:+.3f}" for x in d), "  s.e.", " ".join(f"{x*100:.3f}" for x in se))
    # columns: wetter -> darker, so the model slope is -(...)
    Cw = -(P @ W @ v)[:, None]             # water, scale alpha >= 0
    Cu = -(P @ W)                           # material absorption per sub-band, beta_j >= 0
    Cl = (P @ lamp)[:, None]                # lamp colour drift, free sign
    # Material absorption that FALLS with wavelength (a dye edge tailing into the
    # NIR): beta_1 >= beta_2 >= beta_3 >= beta_4 >= 0, written as cumulative steps
    # c_j >= 0. A uniform absorption is invisible (it is common-mode), so only the
    # steps at 760 / 810 / 860 nm matter.
    S = np.triu(np.ones((4, 4))).T          # beta = S @ c, column j = step below sub-band j
    S = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]], float).T[:, ::-1]
    S = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], float)[::-1]  # rows sub-band 670.., cols steps
    Cd = Cu @ S                              # wetter -> darker where the dye absorbs
    models = {
        "1 water only":                                  (Cw, [0], [np.inf]),
        "2 water + dye, wet darkens (Shimano)":          (np.c_[Cw, Cd], [0] * 5, [np.inf] * 5),
        "3 water + dye, dry darkens (dye concentrates)": (np.c_[Cw, -Cd], [0] * 5, [np.inf] * 5),
        "4 water + dye either way":                      (np.c_[Cw, Cd, -Cd], [0] * 9, [np.inf] * 9),
        "5 = 4 + lamp drift":                            (np.c_[Cw, Cd, -Cd, Cl], [0] * 9 + [-np.inf], [np.inf] * 10),
        "6 water + lamp only":                           (np.c_[Cw, Cl], [0, -np.inf], [np.inf] * 2),
        "x unconstrained material (any spectrum)":       (np.c_[Cw, Cu], [0] + [-np.inf] * 4, [np.inf] * 5),
    }
    out[run] = {}
    for name, (C, lo, hi) in models.items():
        r = lsq_linear(C / se[:, None], d / se, bounds=(lo, hi))
        fit = C @ r.x
        chi = float((((d - fit) / se) ** 2).sum())
        frac = np.linalg.norm(d - fit) / np.linalg.norm(d)
        print(f"  {name:38s} misfit {frac*100:5.1f}% of |d|   chi2 {chi:7.1f} (3 dof data)   params "
              + " ".join(f"{x:.4f}" for x in r.x))
        out[run][name] = {"x": r.x.tolist(), "frac": frac, "chi2": chi}
    # does one moisture-driven direction explain the whole series?
    print(f"  series residual after d*(m-mean): rms {np.sqrt((res**2).sum(0).mean()/2)*100:.2f}% per null-dim-equivalent;"
          f" 30-min floor {floor30*100:.2f}%")
    x = out[run]["4 water + dye either way"]["x"]
    beta = S @ (np.array(x[1:5]) - np.array(x[5:9]))
    print("  model 4 water scale %.4f (water-only fit %.4f); dye absorption by sub-band, x water's own:" % (x[0], out[run]["1 water only"]["x"][0]),
          " ".join(f"{n.split()[0]} {b / v[j] / max(x[0], 1e-9):+.2f}" for j, (n, b) in enumerate(zip(names, beta))))
json.dump(out, open("/workspace/out/moisture/absorb_fit.json", "w"))
