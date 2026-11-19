#!/usr/bin/env python3
"""
moisture_model.py - a soft-capsule moisture model: physics-shaped forward model
of what the four bands see, per product, inverted under "moisture only falls".

The target is the manufacturer's moisture content, not a photometric amount of
water. Whatever the process does alongside the water - the lubricating oil, the
dye concentrating as the shell shrinks, the surface drying out before the bulk -
belongs in the model, as long as the output is moisture.

FORWARD MODEL, per sub-band j of the nested long-pass channels
    Shimano et al., CVPR 2017, eq. 7: I = sum_n tau_n a^n b^(gamma n). With the
    number of scattering events n taken as Poisson with mean nbar, the sum closes:
        I_j = exp( -nbar(gamma) * (1 - exp(-r0 * (u_j(m) + gamma * v_j))) )
        nbar(gamma) = nbar_d * (nbar_s / nbar_d) ** gamma         (their eq. 5)
    v_j        water absorption of the sub-band (known, output/final.json)
    u_j(m)     material / dye absorption, u0_j * (1 + kappa * (m_hi - m)):
               kappa > 0 is a dye that concentrates as the shell dries
    gamma(m)   surface wetness: a soft knee at m_star, 0 well below it, 1 at m_hi
    Channel k is the share-weighted sum of the sub-bands it integrates; the
    model is compared in logs with the common (illumination) term and a
    per-band constant removed, exactly as the data are.
    Oil absorbing above 860 nm looks exactly like water in these channels
    (cos 0.998), so it is not a separate term: it rides on the water path.

PRODUCT PARAMETERS theta_p (10)
    ln nbar_d, ln(nbar_s/nbar_d), ln r0, ln u0_1..4, kappa, m_star, ln width

INVERSION
    Dynamic programming over a moisture grid with m(t) non-increasing; the
    per-band constants are re-estimated from the inverted series a few times.

EVALUATION (--evaluate)
    Leave one reference point out: theta is fitted without the bins within 1 h of
    it and with the moisture curve interpolated without it, the run is inverted,
    and the held-out point is predicted. Compared with a monotone broken-line
    model on the learned spectral direction, and with elapsed time alone.

USAGE
    python analyze/moisture_model.py --levels A.jsonl B.jsonl --final output/final.json \\
        --out model.json [--evaluate]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import moisture_floor as MF

BIN_H = 0.25
GRID = np.arange(7.0, 27.0, 0.05)


# ------------------------------------------------------------------ data ---
def load_run(path, fin, session):
    ds, ref = fin["dataset"][session], fin["reference"][session]
    rows = [r for r in MF.load(path) if (int(r["burst"][5:]) - 1) * MF.BURST_S / 3600 <= ds["cut"]]
    t, L = MF.levels(rows, "warp", "p95")
    X = L - L.mean(0, keepdims=True)                     # common term out
    e = np.arange(0, t.max() + BIN_H, BIN_H)
    idx = np.digitize(t, e)
    ks = [k for k in np.unique(idx) if (idx == k).sum() >= 3]
    tb = np.array([t[idx == k].mean() for k in ks])
    Xb = np.array([X[:, idx == k].mean(1) for k in ks]).T
    sb = np.array([X[:, idx == k].std(1) / np.sqrt((idx == k).sum()) for k in ks]).T
    H, M = np.array(ref["hours"]), np.array(ref["moisture"])
    keep = H <= ds["cut"]
    return dict(session=session, product=ds["product"], t=tb, X=Xb, s=np.median(sb, 1),
                H=H[keep], M=M[keep], all_H=H, all_M=M)


def moisture_curve(t, H, M):
    return np.interp(t, H, M, left=M[0])


# --------------------------------------------------------------- optics ---
class Optics:
    def __init__(self, fin):
        sub = fin["mumodel"]["bands"]
        names = list(sub)
        self.v = np.array([sub[n]["mu"] for n in names])
        sh = np.array([sub[n]["share"] for n in names])
        self.W = np.zeros((4, 4))
        for k in range(4):
            self.W[k, k:] = sh[k:] / sh[k:].sum()

    def channels(self, m, th, m_hi):
        """log channel levels, common term removed, for moisture values m (T,)."""
        lnd, lnr, lnr0, lu1, lu2, lu3, lu4, kappa, mstar, lnw = th
        u0 = np.exp([lu1, lu2, lu3, lu4])
        wdt = np.exp(lnw)
        sp = lambda z: np.logaddexp(0, z)
        gamma = sp((m - mstar) / wdt) / sp((m_hi - mstar) / wdt)
        nbar = np.exp(lnd + lnr * gamma)
        u = u0[None, :] * (1 + kappa * (m_hi - m))[:, None]
        tau = np.exp(lnr0) * (np.clip(u, 0, None) + gamma[:, None] * self.v[None, :])
        I = np.exp(-nbar[:, None] * (1 - np.exp(-tau)))      # T x 4 sub-bands
        C = np.log(I @ self.W.T + 1e-300)                    # T x 4 channels
        return (C - C.mean(1, keepdims=True)).T              # 4 x T


def centred(A):
    return A - A.mean(1, keepdims=True)


# ------------------------------------------------------------------ fit ---
def fit_theta(op, t, X, s, m, m_hi, starts=12, seed=0):
    rng = np.random.default_rng(seed)
    Xc = centred(X)

    def resid(th):
        P = centred(op.channels(m, th, m_hi))
        return ((Xc - P) / s[:, None]).ravel()

    lo = [-3, 0, -6, -8, -8, -8, -8, -0.2, 7, np.log(0.2)]
    hi = [4, 4, 2, 2, 2, 2, 2, 0.2, 27, np.log(8)]
    best = None
    for _ in range(starts):
        x0 = [rng.uniform(a, b) for a, b in zip(lo, hi)]
        x0[8] = rng.uniform(m.min(), m.max())
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=4000)
        except ValueError:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x, 2 * best.cost / max(1, Xc.size - len(best.x))


def invert(op, th, X, s, m_hi, iters=4, m_init=None):
    """Monotone (non-increasing) moisture series that best explains X."""
    P = op.channels(GRID, th, m_hi)                          # 4 x G
    T = X.shape[1]
    m = np.full(T, GRID.mean()) if m_init is None else m_init.copy()
    for _ in range(iters):
        K = (X - op.channels(m, th, m_hi)).mean(1, keepdims=True)   # per-band constant
        cost = (((X - K)[:, :, None] - P[:, None, :]) / s[:, None, None]) ** 2
        cost = cost.sum(0)                                   # T x G
        acc = np.zeros_like(cost); back = np.zeros(cost.shape, int)
        acc[0] = cost[0]
        for i in range(1, T):
            # non-increasing in time: from any state at >= this moisture
            run_min = np.minimum.accumulate(acc[i - 1][::-1])[::-1]
            back_i = np.empty(len(GRID), int)
            best_j = len(GRID) - 1
            for g in range(len(GRID) - 1, -1, -1):
                if acc[i - 1][g] <= acc[i - 1][best_j]:
                    best_j = g
                back_i[g] = best_j
            back[i] = back_i
            acc[i] = cost[i] + run_min
        path = np.empty(T, int); path[-1] = int(np.argmin(acc[-1]))
        for i in range(T - 1, 0, -1):
            path[i - 1] = back[i][path[i]]
        m = GRID[path]
    return m


# ------------------------------------------------------------- baseline ---
def learned_dir(t, X, m, exclude=()):
    keep = np.ones(len(t), bool)
    for h in exclude:
        keep &= np.abs(t - h) > 1.0
    Xk, mk = centred(X[:, keep]), m[keep] - m[keep].mean()
    d = Xk @ mk / (mk @ mk)
    return d / np.linalg.norm(d)


def broken_line(x, y):
    best = None
    for knot in np.quantile(x, np.linspace(.2, .8, 13)):
        A = np.c_[np.ones_like(x), x, np.maximum(x - knot, 0)]
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ c
        if best is None or r @ r < best[0]:
            best = (r @ r, knot, c)
    _, knot, c = best
    return lambda z: c[0] + c[1] * z + c[2] * np.maximum(z - knot, 0)


def at(t, series, h, half=0.5):
    k = np.abs(t - h) <= half
    return float(series[k].mean())


# ------------------------------------------------------------- evaluate ---
def evaluate(op, R, starts):
    t, X, s, H, M = R["t"], R["X"], R["s"], R["H"], R["M"]
    m_hi = float(R["all_M"].max())
    out = {"model": [], "baseline": [], "clock": []}
    for i in range(len(H)):
        Hk, Mk = np.delete(H, i), np.delete(M, i)
        mt = moisture_curve(t, Hk, Mk)
        keep = np.abs(t - H[i]) > 1.0
        th, _ = fit_theta(op, t[keep], X[:, keep], s, mt[keep], m_hi, starts=starts, seed=i)
        m_inv = invert(op, th, X, s, m_hi, m_init=mt)
        # map the inverted series onto the reference scale with the other points
        # (the absolute level is weakly determined by the shape alone)
        xi = np.array([at(t, m_inv, h) for h in Hk])
        a = np.polyfit(xi, Mk, 1)
        out["model"].append(np.polyval(a, at(t, m_inv, H[i])) - M[i])

        u = learned_dir(t, X, mt, exclude=[H[i]])
        w = u @ X
        f = broken_line(np.array([at(t, w, h) for h in Hk]), Mk)
        out["baseline"].append(float(f(at(t, w, H[i])) - M[i]))

        out["clock"].append(float(np.polyval(np.polyfit(Hk, Mk, 1), H[i]) - M[i]))
    return {k: float(np.sqrt(np.mean(np.square(v)))) for k, v in out.items()}, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", type=Path, required=True)
    ap.add_argument("--final", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--starts", type=int, default=12, help="random restarts per fit")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    fin = json.loads(args.final.read_text())
    op = Optics(fin)
    names = ["ln nbar_d", "ln nbar_s/nbar_d", "ln r0", "ln u0 670-760", "ln u0 760-810",
             "ln u0 810-860", "ln u0 860+", "kappa", "m_star", "ln width"]
    report = {}
    for path in args.levels:
        session = next(s for s in fin["dataset"] if s in path.name)
        R = load_run(path, fin, session)
        m_hi = float(R["all_M"].max())
        mt = moisture_curve(R["t"], R["H"], R["M"])
        th, chi = fit_theta(op, R["t"], R["X"], R["s"], mt, m_hi, starts=args.starts)
        m_inv = invert(op, th, R["X"], R["s"], m_hi, m_init=mt)
        print(f"\n== {session} ({R['product']}): {R['X'].shape[1]} bins of {BIN_H*60:.0f} min")
        print("  theta: " + "  ".join(f"{n} {v:+.3f}" for n, v in zip(names, th)))
        print(f"  reduced chi2 of the forward fit {chi:.2f}")
        res = {"theta": dict(zip(names, th.tolist())), "chi2_red": chi,
               "t": R["t"].tolist(), "m_inverted": m_inv.tolist(), "m_reference": mt.tolist()}
        if args.evaluate:
            rmse, errs = evaluate(op, R, args.starts)
            print("  leave-one-out RMSE (moisture points): " +
                  "  ".join(f"{k} {v:.2f}" for k, v in rmse.items()))
            res["loo"] = rmse; res["loo_errors"] = errs
        report[session] = res
    args.out.write_text(json.dumps(report))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
