"""Few-parameter joint fits over both runs: does a per-product dye term with a
shared shape absorb the difference? 6 numbers of data (3 per run)."""
import json, sys, numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import chi2 as CHI2
sys.path.insert(0, "/workspace/code/analyze")
import moisture_floor as F
exec(open("/workspace/re/absorb.py").read().split("out = {}")[0])   # W, v, P, lamp, binned()
runs = list(files)
D, SE = {}, {}
for run in runs:
    t, m, X = binned(run); mc = m - m.mean()
    d = X @ mc / (mc @ mc); res = X - np.outer(d, mc)
    # residuals are autocorrelated over half-hours; inflate the s.e. by the lag-1 factor
    r1 = np.mean([np.corrcoef(rr[:-1], rr[1:])[0, 1] for rr in res])
    infl = np.sqrt((1 + r1) / (1 - r1)) if r1 > 0 else 1.0
    D[run] = d; SE[run] = np.sqrt((res ** 2).sum(1) / (len(m) - 2) / (mc @ mc)) * infl
    print(f"{run}: lag-1 residual corr {r1:.2f}, s.e. inflated x{infl:.2f}")
# a 3-d orthonormal basis of the common-free subspace, so 4 correlated band values -> 3 numbers
Q = np.linalg.qr(np.c_[np.ones(4), np.eye(4)])[0][:, 1:4]
y = np.concatenate([Q.T @ D[r] for r in runs])
s = np.concatenate([np.sqrt(Q.T ** 2 @ SE[r] ** 2) for r in runs])
wat = -(P @ W @ v); lam = P @ lamp
def step(j):            # dye edge: absorbs in sub-bands below boundary j (0: 670-760 only, 1: < 810, 2: < 860)
    b = np.zeros(4); b[: j + 1] = 1; return -(P @ W @ b)
Z = np.zeros(4)
def fit(cols, lo, hi, name):
    A = np.array(cols).T
    r = lsq_linear(A / s[:, None], y / s, bounds=(lo, hi))
    c2 = float((((y - A @ r.x) / s) ** 2).sum()); dof = len(y) - A.shape[1]
    p = 1 - CHI2.cdf(c2, dof) if dof > 0 else float("nan")
    print(f"  {name:58s} k={A.shape[1]} dof={dof}  chi2 {c2:7.2f}  p {p:.3f}   " + " ".join(f"{x:+.4f}" for x in r.x))
    return r.x, c2, dof
blk = lambda a, b: np.concatenate([Q.T @ a, Q.T @ b])
print("\nwater sensitivity shared by both products")
fit([blk(wat, wat)], [0], [np.inf], "water only")
fit([blk(wat, wat), blk(lam, Z), blk(Z, lam)], [0, -np.inf, -np.inf], [np.inf] * 3, "water + lamp per run")
for j, edge in enumerate(("760", "810", "860")):
    fit([blk(wat, wat), blk(step(j), Z), blk(Z, step(j))], [0, -np.inf, -np.inf], [np.inf] * 3,
        f"water + dye edge {edge} nm, amplitude per product")
print("\nwater sensitivity per product (penetration differs)")
fit([blk(wat, Z), blk(Z, wat)], [0, 0], [np.inf] * 2, "water only")
for j, edge in enumerate(("760", "810", "860")):
    fit([blk(wat, Z), blk(Z, wat), blk(step(j), Z), blk(Z, step(j))], [0, 0, -np.inf, -np.inf], [np.inf] * 4,
        f"water + dye edge {edge} nm, amplitude per product")
fit([blk(wat, Z), blk(Z, wat), blk(lam, Z), blk(Z, lam)], [0, 0, -np.inf, -np.inf], [np.inf] * 4,
    "water + lamp, per product")
