"""Oil (C-H overtones), the cloth event, and a Kubelka-Munk bulk model."""
import json, sys, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
exec(open("/workspace/re/absorb.py").read().split("out = {}")[0])   # W, v, P, lamp, binned(), files
unit = lambda x: x / np.linalg.norm(x)
wat = unit(-(P @ W @ v))                         # wetter -> this direction
e = np.eye(4)
oil3 = unit(-(P @ W @ e[3]))                     # absorber only above 860 nm (C-H 3rd overtone ~910-930)
oil4 = unit(-(P @ W @ e[0]))                     # absorber only in 670-760 (C-H 4th overtone ~740-760, weak)
lam = unit(P @ lamp)
print("direction cosines in the 4 channels (common removed):")
print(f"  water vs oil(>860 nm) {wat @ oil3:+.3f}   water vs oil(670-760) {wat @ oil4:+.3f}   water vs lamp {wat @ lam:+.3f}")

# --- cloth event: 1 h before vs 1 h after, common removed ---------------------
fin2 = fin
for run, f in files.items():
    ds = fin2["dataset"][run]; tc = ds["cloth"]
    rows = F.load("/workspace/out/moisture/" + f)
    t, L = F.levels(rows, "warp", "p95")
    X = P @ (L - L.mean(1, keepdims=True))
    pre = X[:, (t > tc - 1.0) & (t < tc - 0.1)].mean(1); post = X[:, (t > tc + 0.1) & (t < tc + 1.0)].mean(1)
    dX = post - pre
    print(f"{run} ({ds['product']}) cloth at {tc:.2f} h: after-before " + " ".join(f"{x*100:+.2f}%" for x in dX)
          + f" | cos: water {unit(dX) @ wat:+.2f}  oil>860 {unit(dX) @ oil3:+.2f}  lamp {unit(dX) @ lam:+.2f}"
          + f"  brightness {(L[:, (t > tc + 0.1) & (t < tc + 1)].mean() - L[:, (t > tc - 1) & (t < tc - .1)].mean())*100:+.1f}%")

# --- Kubelka-Munk: R = 1 + q - sqrt(q^2 + 2q), q = K/S per sub-band -------------
# K_j = k0 + w*v_j (baseline material/oil absorption k0, water w); channel = share-weighted sum of sub-band R.
# Response per moisture point: dlnI/dm = a_w * d/dw + a_s * d/dlnS, at a baseline (k0/S, w/S).
def chan(q):
    R = 1 + q - np.sqrt(q * q + 2 * q)
    return np.log(W @ R)
def responses(k0, w0):
    q = (k0 + w0 * v)
    h = 1e-4
    dw = (chan(q + h * v) - chan(q - h * v)) / (2 * h)            # more water
    ds = (chan(q * np.exp(-h)) - chan(q * np.exp(h))) / (2 * h)   # less scattering (gaps filled)
    return P @ dw, P @ ds
D = {}
for run in files:
    t, m, X = binned(run); mc = m - m.mean()
    d = X @ mc / (mc @ mc); res = X - np.outer(d, mc)
    D[run] = (d, np.sqrt((res ** 2).sum(1) / (len(m) - 2) / (mc @ mc)) * 1.2)
runs = list(files)
def resid(p):
    lk0, lw0, aA, sA, aB, sB = p
    dw, ds = responses(np.exp(lk0), np.exp(lw0))
    out = []
    for run, a, s_ in ((runs[0], aA, sA), (runs[1], aB, sB)):
        d, se = D[run]
        out.append((d - (a * dw + s_ * ds)) / se)
    return np.concatenate(out)
best = None
for lk0 in np.linspace(-6, 1, 8):
    for lw0 in np.linspace(-4, 3, 8):
        r = least_squares(resid, [lk0, lw0, 0.001, 0.001, 0.001, 0.001])
        if best is None or r.cost < best.cost:
            best = r
c2 = 2 * best.cost
print(f"KM bulk, shared baseline (k0/S={np.exp(best.x[0]):.3g}, w/S={np.exp(best.x[1]):.3g}), per-run water + scattering amplitudes:"
      f" chi2 {c2:.2f} on {8 - 6} dof (8 numbers: 4 bands x 2 runs, common mode costs 1 each)")
print("   per-run [water, scattering] amplitudes per moisture pt:", np.round(best.x[2:], 5))
# KM without the scattering term (water only, nonlinear)
def resid_w(p):
    lk0, lw0, aA, aB = p
    dw, _ = responses(np.exp(lk0), np.exp(lw0))
    return np.concatenate([(D[runs[0]][0] - aA * dw) / D[runs[0]][1], (D[runs[1]][0] - aB * dw) / D[runs[1]][1]])
bw = min((least_squares(resid_w, [a, b, .001, .001]) for a in np.linspace(-6, 1, 8) for b in np.linspace(-4, 3, 8)), key=lambda r: r.cost)
print(f"KM bulk, water only (nonlinear): chi2 {2*bw.cost:.2f} on 2 dof")
for run, a, s_ in ((runs[0], best.x[2], best.x[3]), (runs[1], best.x[4], best.x[5])):
    dw, ds = responses(np.exp(best.x[0]), np.exp(best.x[1]))
    print(f"   {run}: observed " + " ".join(f"{x*100:+.3f}" for x in D[run][0]) + "   KM fit " + " ".join(f"{x*100:+.3f}" for x in a * dw + s_ * ds))
