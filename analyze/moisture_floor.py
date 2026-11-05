#!/usr/bin/env python3
"""
moisture_floor.py - how small a change in water the four bands can resolve,
and whether it reproduces between runs, from analyze/band_levels.py's output.

THE MODEL (Beer-Lambert, in logs)
    ln I_b(t) = ln E(t) + K_b - mu_b * w(t) + noise
    K_b    per band, constant           -> removed by centring each band over the run
    E(t)   per instant, all bands alike  -> removed by centring each instant over bands
    w(t)   the water, along mu - mean(mu)
What is left after those three has no physical source in the model: it is the
NULL plane (2 dimensions out of 4), and its spread is the measurement floor.
Dividing by |mu - mean(mu)| turns it into the water change it would mask.

FAST AND SLOW
    The null plane is split with a running median over --slow-min minutes:
    fast  burst-to-burst scatter - averaging N bursts divides it by sqrt(N)
    slow  what drifts over tens of minutes - more data does nothing for it
    floor@window is the null spread after averaging over a window, which is
    what a moisture curve sampled every 2 h actually sees.

REPRODUCIBILITY
    The water index is correlated with the reference moisture in each run; the
    null index too. A real water signal has the same sign in both runs and
    beats the null.

USAGE
    python analyze/moisture_floor.py --levels A.jsonl B.jsonl --final output/final.json \\
        --out report.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

BANDS = ("750", "800", "850", "900")
STATS = ("mean", "p50", "p90", "p95", "p99")
BAND_TILE = {(0, 0): "800", (0, 1): "750", (1, 0): "900", (1, 1): "850"}
CAMS = {b: [f"e{i:02d}" for i in range(16) if BAND_TILE[((i // 4) % 2, (i % 4) % 2)] == b]
        for b in BANDS}
ROW = {f"e{i:02d}": i // 4 for i in range(16)}
BURST_S = 120.0


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
    rows = [r for r in rows if "frames" in r]
    rows.sort(key=lambda r: int(r["burst"][5:]))
    return rows


def levels(rows, source, stat, frames=None, cams=None, every=1):
    """-> (hours, 4 x T log levels). One value per burst: mean log over frames."""
    k = STATS.index(stat)
    t, L = [], []
    for r in rows[::every]:
        fr = sorted(r["frames"], key=int)
        if frames is not None:
            fr = fr[:frames] if isinstance(frames, int) else [f for f in fr if int(f) in frames]
        per_band = []
        for b in BANDS:
            vals = []
            for f in fr:
                rec = r["frames"][f]
                if source == "band":
                    v = [rec["band"][b][k]]
                else:
                    sel = cams[b] if cams else CAMS[b]
                    v = [rec[source][c][k] for c in sel]
                v = [x for x in v if x and x > 0]
                if v:
                    vals.append(np.mean(np.log(v)))
            per_band.append(np.mean(vals) if vals else np.nan)
        if np.all(np.isfinite(per_band)):
            t.append((int(r["burst"][5:]) - 1) * BURST_S / 3600)
            L.append(per_band)
    return np.array(t), np.array(L).T


def directions(mu):
    c = mu - mu.mean()
    sens = np.linalg.norm(c)
    q, _ = np.linalg.qr(np.c_[np.ones(4), c, np.eye(4)])
    return c / sens, sens, q[:, 2:4]


def running_median(t, x, minutes):
    h = minutes / 60 / 2
    return np.array([np.median(x[(t >= ti - h) & (t <= ti + h)]) for ti in t])


def window_mean(t, x, minutes):
    if minutes <= 0:
        return x
    edges = np.arange(t.min(), t.max() + minutes / 60, minutes / 60)
    idx = np.digitize(t, edges)
    return np.array([x[idx == i].mean() for i in np.unique(idx) if (idx == i).sum() >= 2])


def floor(t, L, mu, slow_min=30.0, windows=(0, 30, 120)):
    u, sens, N = directions(mu)
    X = L - L.mean(1, keepdims=True)
    X = X - X.mean(0, keepdims=True)
    z = N.T @ X                                     # 2 x T
    sm = np.array([running_median(t, zi, slow_min) for zi in z])
    fast = z - sm
    out = {"n": int(len(t)), "sens": float(sens),
           "sigma": float(np.sqrt(z.var(1).mean())),
           "fast": float(np.sqrt(fast.var(1).mean())),
           "slow": float(np.sqrt(sm.var(1).mean()))}
    for w in windows:
        if w:
            out[f"sigma@{w}min"] = float(np.sqrt(np.mean([window_mean(t, zi, w).var() for zi in z])))
    return out, u @ X, z


def moisture_at(t, ref):
    return np.interp(t, ref["hours"], ref["moisture"], left=ref["moisture"][0])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", nargs="+", type=Path, required=True,
                    help="band_levels.py json lines, one per run (session name in the file name)")
    ap.add_argument("--final", type=Path, required=True,
                    help="output/final.json: reference moisture, cut times and mu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slow-min", type=float, default=30.0)
    args = ap.parse_args()

    fin = json.loads(args.final.read_text())
    mu = np.array([fin["mumodel"]["mu"][b] for b in BANDS])
    report = {"mu": mu.tolist(), "slow_min": args.slow_min, "runs": {}}

    near = {b: [c for c in CAMS[b] if ROW[c] in (0, 1)] for b in BANDS}   # 94 / 102 mm rows
    far = {b: [c for c in CAMS[b] if ROW[c] in (2, 3)] for b in BANDS}    # 115 / 120 mm rows
    single = {b: [CAMS[b][-1]] for b in BANDS}
    configs = {
        "raw, 1 camera": dict(source="raw", cams=single),
        "raw, 4-view mean": dict(source="raw"),
        "warped, 4-view mean": dict(source="warp"),
        "warped, pixel-averaged": dict(source="band"),
        "warped, near rows only": dict(source="warp", cams=near),
        "warped, far rows only": dict(source="warp", cams=far),
    }

    for path in args.levels:
        session = next(s for s in fin["dataset"] if s in path.name)
        ds, ref = fin["dataset"][session], fin["reference"][session]
        rows = [r for r in load(path) if (int(r["burst"][5:]) - 1) * BURST_S / 3600 <= ds["cut"]]
        nfr = max(len(r["frames"]) for r in rows)
        res = {"bursts": len(rows), "frames_per_burst": nfr, "cut_h": ds["cut"], "configs": {},
               "stats": {}, "sampling": {}, "frames": {}}

        for name, cfg in configs.items():
            t, L = levels(rows, cfg["source"], "p95", cams=cfg.get("cams"))
            fl, w, z = floor(t, L, mu, args.slow_min)
            m = moisture_at(t, ref)
            fl.update(r_water=corr(w, m), r_null=max(abs(corr(zi, m)) for zi in z))
            res["configs"][name] = fl

        for st in STATS:
            t, L = levels(rows, "warp", st)
            res["stats"][st] = floor(t, L, mu, args.slow_min)[0]

        for every, nf in ((2, 6), (2, None), (1, 6), (1, None)):
            t, L = levels(rows, "warp", "p95", frames=nf, every=every)
            res["sampling"][f"every {every}, {nf or nfr} frames"] = floor(t, L, mu, args.slow_min)[0]

        for nf in sorted({1, 2, 4, 8, nfr}):
            if nf <= nfr:
                t, L = levels(rows, "warp", "p95", frames=nf)
                res["frames"][str(nf)] = floor(t, L, mu, args.slow_min)[0]

        t, L = levels(rows, "warp", "p95")
        _, w, z = floor(t, L, mu, args.slow_min)
        res["series"] = {"hours": t.round(4).tolist(), "water": w.round(5).tolist(),
                         "null": z.round(5).tolist(), "moisture": moisture_at(t, ref).round(3).tolist(),
                         "bright": L.mean(0).round(5).tolist()}
        report["runs"][session] = res

        print(f"\n== {session}: {len(rows)} bursts up to {ds['cut']} h, {nfr} frames each")
        for name, fl in res["configs"].items():
            print(f"  {name:26s} sigma {fl['sigma']*100:5.2f}%  fast {fl['fast']*100:5.2f}%  "
                  f"slow {fl['slow']*100:5.2f}%  @2h {fl['sigma@120min']*100:5.2f}%  "
                  f"r_water {fl['r_water']:+.2f}  |r_null| {fl['r_null']:.2f}")
        print("  statistic: " + "  ".join(f"{k} {v['sigma']*100:.2f}%" for k, v in res["stats"].items()))
        print("  sampling : " + "  ".join(f"[{k}] {v['sigma']*100:.2f}%/{v['fast']*100:.2f}%"
                                          for k, v in res["sampling"].items()))
        print("  frames   : " + "  ".join(f"{k}: {v['fast']*100:.2f}%" for k, v in res["frames"].items()))

    args.out.write_text(json.dumps(report))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
