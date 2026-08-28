#!/usr/bin/env python3
"""
temporal_align.py - Estimate the integer frame offset between cameras.

Why this is needed
------------------
`camera_agent.py` stamps each frame with `datetime.now()` on the capturing Pi,
so the timestamps carry that Pi's clock error.  In session 20260728_072115 the
burst0001 start times span 130 s:

    e00                          07:21:29.6   (-77 s)
    e01,02,03,05,06,07,08,09,10,11,13,15
                                 07:22:46-49  (reference cluster)
    e14                          07:22:58.8   (+12 s)
    e12                          07:23:34.4   (+48 s)
    e04                          07:23:39.5   (+53 s)

yet every camera's burst0001 -> burst0002 interval is 200.01 s, i.e. they are
all locked to the same Arduino trigger schedule.  Either the Pi clocks are
skewed (all 16 bursts are the same physical event, and only a small integer
frame offset separates them) or the agents really did capture different
bursts (in which case only 12 cameras overlap and the 900 nm band collapses
to e06 alone).  Timestamps cannot tell these apart; image content can.

Method
------
Per camera, per frame, a small viewpoint- and band-robust signature is built:

    s[k] = [ sharpness(k), d_sharpness(k), |global motion|(k), dx(k), dy(k) ]

Sharpness (mean |Laplacian| of the locally normalised frame) dips whenever the
tumbler motion smears a frame; the global motion vector comes from phase
correlation against the previous frame.  Both are driven by the drum's motion,
which is a property of the SCENE, so all cameras observing the same trigger
burst see the same time series - regardless of viewpoint, filter band or
exposure.  Each signature channel is z-scored (killing per-band gain), and the
offset is the lag maximising the cross-correlation against the reference.

A sharp, high peak means "same physical burst, shift by n frames".  A flat
correlogram means the bursts are genuinely different events.

Usage
-----
  python analyze/temporal_align.py collected_data/20260728_072115
  python analyze/temporal_align.py collected_data/20260728_072115 --burst 1 --max-lag 30
  python analyze/temporal_align.py collected_data/20260728_072115 --cams 0 4 12 14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from raw10 import unpack_raw10  # noqa: E402

N_CAM = 16
ROW_STEP = 4          # decimate rows before unpacking: 4x cheaper, signature-safe
SIG_W = 364           # signature working width


def find_burst(session: Path, cam: int, burst: int) -> Path:
    subs = sorted(session.glob(f"e{cam:02d}_*"))
    if not subs:
        raise FileNotFoundError(f"no directory for e{cam:02d}")
    c = sorted(subs[0].glob(f"e{cam:02d}_burst{burst:04d}_*.npz"))
    if not c:
        raise FileNotFoundError(f"no burst{burst:04d} for e{cam:02d}")
    return c[0]


THUMB = (32, 24)      # (w, h) coarse thumbnail used for sequence matching


def frame_signature(frames: np.ndarray):
    """(N, h, w) float32 -> (sig (N,3), thumbs (N, 24, 32)).

    Two complementary descriptors, both independent of radiometry:

    sig     scalar channels driven by drum motion - sharpness (dips whenever
            the tumbler smears a frame), its derivative, and residual contrast.
            NOTE phase-correlation motion was tried and discarded: the mesh is
            a strong periodic texture, so phase correlation locks onto the
            wrong harmonic and returns |dx| of 60-165 px on frames where the
            true motion is ~2 px.

    thumbs  heavily blurred, locally normalised 32x24 thumbnails.  At this
            scale inter-camera parallax is a fraction of a pixel while drum
            rotation still moves content, so matching whole thumbnail
            SEQUENCES across cameras is far more discriminative than any
            scalar - and local normalisation makes it band-agnostic.
    """
    N = frames.shape[0]
    sig = np.zeros((N, 3), np.float64)
    thumbs = np.zeros((N, THUMB[1], THUMB[0]), np.float32)
    for k in range(N):
        f = frames[k]
        mu = cv2.blur(f, (31, 31))
        g = f - mu
        s = float(np.sqrt((g * g).mean()) + 1e-9)
        g = g / s
        sig[k, 0] = float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())
        sig[k, 2] = s
        t = cv2.resize(f, THUMB, interpolation=cv2.INTER_AREA)
        t = t - t.mean()
        thumbs[k] = t / (t.std() + 1e-9)
    sig[1:, 1] = np.diff(sig[:, 0])
    return sig, thumbs


def load_signature(session: Path, cam: int, burst: int, cache: Path,
                   n_frames: int | None = None):
    cp = cache / f"e{cam:02d}_b{burst:04d}_sig.npz"
    if cp.exists():
        d = np.load(cp)
        return d["sig"], d["thumbs"]
    p = find_burst(session, cam, burst)
    d = np.load(p)
    packed = d["frames_raw"]
    W = int(np.asarray(d["raw_shape"]).ravel()[1])
    n = packed.shape[0] if n_frames is None else min(n_frames, packed.shape[0])
    out = []
    CH = 10
    for s in range(0, n, CH):
        blk = unpack_raw10(packed[s:s + CH, ::ROW_STEP, :], W).astype(np.float32)
        h = int(blk.shape[1] * SIG_W / blk.shape[2])
        blk = np.stack([cv2.resize(x, (SIG_W, h), interpolation=cv2.INTER_AREA)
                        for x in blk])
        out.append(blk)
    frames = np.concatenate(out, 0)
    del out, packed, d
    sig, thumbs = frame_signature(frames)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cp, sig=sig, thumbs=thumbs)
    return sig, thumbs


def zscore(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    return x / np.maximum(s, 1e-9)


def _pairs(N, L):
    """Index pairs (i in ref, j in other) for lag L: ref[k+L] <-> other[k]."""
    if L >= 0:
        return np.arange(L, N), np.arange(0, N - L)
    return np.arange(0, N + L), np.arange(-L, N)


def best_lag(ref, other, max_lag: int, w_thumb: float = 1.0,
             w_sig: float = 0.5, channels=(0, 1)):
    """Integer-lag alignment from thumbnail-sequence + scalar correlation.

    Returns (lag, peak, peak_to_sidelobe, lags, score).  A high peak-to-
    sidelobe ratio means the two bursts really are the same physical event;
    a flat correlogram means there is no temporal overlap at all.
    """
    sig_r, th_r = ref
    sig_o, th_o = other
    a = zscore(sig_r[:, list(channels)])
    b = zscore(sig_o[:, list(channels)])
    # Remove each camera's temporal mean thumbnail.  What is left is purely the
    # time-varying part of the scene (drum rotation, tumbling capsules); the
    # static part - vignetting, filter transmission, the mean mesh pattern -
    # carries no timing information and would otherwise dominate the ZNCC and
    # flatten the correlogram.
    Tr = th_r.reshape(len(th_r), -1)
    To = th_o.reshape(len(th_o), -1)
    Tr = Tr - Tr.mean(0, keepdims=True)
    To = To - To.mean(0, keepdims=True)
    N = min(len(a), len(b))

    lags = np.arange(-max_lag, max_lag + 1)
    score = np.full(len(lags), -np.inf)
    for j, L in enumerate(lags):
        ii, jj = _pairs(N, L)
        if len(ii) < 20:
            continue
        s_sig = float((zscore(a[ii]) * zscore(b[jj])).mean())
        # per-frame ZNCC of thumbnails, averaged over the overlap
        X, Y = Tr[ii], To[jj]
        X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
        Y = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-9)
        s_th = float((X * Y).mean())
        score[j] = w_thumb * s_th + w_sig * s_sig

    i = int(np.argmax(score))
    peak = score[i]
    side = np.delete(score, slice(max(0, i - 2), i + 3))
    side = side[np.isfinite(side)]
    psr = float((peak - side.mean()) / max(side.std(), 1e-9)) if side.size else 0.0
    return int(lags[i]), float(peak), psr, lags, score


def grid_edges(axis: str = "h", span: int = 2):
    """Edges of the 4x4 array, restricted to one baseline direction.

    PARALLAX ALIASES INTO TIME.  The tumbler content drifts vertically at
    ~2.4 px/frame, so displacing a camera vertically shifts the observed
    content in exactly the direction the scene is already moving - a
    thumbnail matcher cannot tell "one row down" from "9 frames later".
    Measured on session 20260728_072115:

        horizontal pairs   40 mm  ->  lag  0
        horizontal pairs   80 mm  ->  lag  0, -3, +1, -2, 0   (band-independent)
        vertical   pairs   40 mm  ->  lag -9   (all 8 pairs)
        vertical   pairs   80 mm  ->  lag -17  (also for SAME-band pairs)

    The vertical lag is strictly proportional to vertical baseline and
    independent of the filter band, i.e. it is geometry, not timing.  Only
    baselines ORTHOGONAL to the dominant scene motion carry usable timing
    information, so by default only horizontal edges are used.
    """
    E = []
    for i in range(N_CAM):
        r, c = divmod(i, 4)
        for s in range(1, span + 1):
            if axis in ("h", "both") and c + s < 4:
                E.append((i, i + s))
            if axis in ("v", "both") and r + s < 4:
                E.append((i, i + 4 * s))
    return E


def motion_axis(sigs) -> str:
    """Dominant scene-motion axis -> the baseline direction to AVOID.

    Returns "h" if the scene moves vertically (use horizontal edges) and
    "v" if it moves horizontally.
    """
    dv = []
    for c, (sig, th) in sigs.items():
        t = th - th.mean(0, keepdims=True)
        d = np.diff(t, axis=0)
        dv.append((np.abs(np.diff(d, axis=1)).mean(),      # vertical structure
                   np.abs(np.diff(d, axis=2)).mean()))     # horizontal structure
    dv = np.asarray(dv).mean(0)
    return "h" if dv[0] >= dv[1] else "v"


def solve_offsets(sigs, max_lag, ref, min_psr=2.5, verbose=True, axis="h"):
    """Global frame offsets from pairwise NEAREST-NEIGHBOUR lags.

    Matching every camera directly against one reference fails here: the mesh
    sits so close to the array that the disparity between opposite corners
    exceeds the sensor width, so a 32x24 thumbnail from e12 shares almost no
    content with one from e01 and the correlogram flattens (measured PSR drops
    from 5.8 at a 40 mm baseline to 1.6 at 120 mm).

    Only short baselines orthogonal to the scene motion are therefore
    correlated, and the per-edge lags L_ij = o_i - o_j are combined by
    PSR-weighted least squares over the grid graph.  The redundancy (20 edges
    for 15 unknowns) both stabilises the solution and exposes inconsistency:
    a large residual on an edge means that pair genuinely does not overlap.
    """
    E = grid_edges(axis)
    rows, rhs, wts, info = [], [], [], []
    for (i, j) in E:
        if i not in sigs or j not in sigs:
            continue
        lag, peak, psr, _, _ = best_lag(sigs[i], sigs[j], max_lag)
        r = np.zeros(N_CAM)
        r[i], r[j] = 1.0, -1.0
        w = max(psr - min_psr, 0.0)
        rows.append(r)
        rhs.append(lag)
        wts.append(w)
        info.append((i, j, lag, peak, psr))
        if verbose:
            print(f"  e{i:02d}-e{j:02d}  lag={lag:+4d}  peak={peak:5.3f}  "
                  f"PSR={psr:5.2f}  w={w:4.2f}", flush=True)

    # gauge fix: offset of the reference camera is 0
    r = np.zeros(N_CAM)
    r[ref] = 1.0
    rows.append(r)
    rhs.append(0.0)
    wts.append(10.0)

    A = np.asarray(rows)
    b = np.asarray(rhs, float)
    w = np.sqrt(np.asarray(wts, float))
    sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    resid = A @ sol - b
    return sol, info, resid[:len(info)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--burst", type=int, default=1)
    ap.add_argument("--max-lag", type=int, default=30)
    ap.add_argument("--ref", type=int, default=1)
    ap.add_argument("--cams", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--direct", action="store_true",
                    help="also report naive direct-to-reference lags")
    ap.add_argument("--axis", choices=["h", "v", "both", "auto"], default="auto",
                    help="baseline direction to correlate along; must be "
                         "orthogonal to the dominant scene motion")
    a = ap.parse_args()

    out = a.out or (a.session.parent.parent / "sa_results" / a.session.name)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"

    cams = a.cams if a.cams else list(range(N_CAM))
    sigs = {}
    for c in cams + ([a.ref] if a.ref not in cams else []):
        sigs[c] = load_signature(a.session, c, a.burst, cache)
        print(f"  e{c:02d} signature ready ({len(sigs[c][0])} frames)", flush=True)

    print(f"\nreference = e{a.ref:02d}, burst{a.burst:04d}, "
          f"lag search +-{a.max_lag} frames (1 frame = 33.25 ms)")

    if a.direct:
        ref = sigs[a.ref]
        res = {}
        print(f"\n[direct-to-reference]\n{'cam':5s} {'lag':>5s} {'peak':>7s} "
              f"{'PSR':>7s}  verdict")
        for c in cams:
            lag, peak, psr, _, _ = best_lag(ref, sigs[c], a.max_lag)
            verdict = ("SAME burst" if psr >= 4.0 and peak > 0.25 else
                       "weak" if psr >= 2.5 else "DIFFERENT burst / no overlap")
            res[f"e{c:02d}"] = dict(lag=lag, peak=peak, psr=psr, verdict=verdict)
            print(f"e{c:02d}   {lag:5d} {peak:7.3f} {psr:7.2f}  {verdict}")
        (out / f"temporal_align_direct_b{a.burst:04d}.json").write_text(
            json.dumps(dict(ref=a.ref, burst=a.burst, results=res), indent=2))

    axis = a.axis if a.axis != "auto" else motion_axis(sigs)
    print(f"\n[grid graph, {axis}-baselines only "
          f"(scene motion is {'vertical' if axis == 'h' else 'horizontal'})]")
    sol, info, resid = solve_offsets(sigs, a.max_lag, a.ref, axis=axis)
    off = np.rint(sol).astype(int)
    # edges below the PSR floor carry zero weight and are excluded from the fit;
    # scoring them would report a failure the solution never depended on.
    keep = [k for k, (_, _, _, _, psr) in enumerate(info) if psr > 2.5]
    bad = [(info[k][0], info[k][1], resid[k]) for k in keep if abs(resid[k]) > 3]

    print(f"\n{'cam':5s} {'offset':>7s}   (frames relative to e{a.ref:02d})")
    for i in range(N_CAM):
        print(f"e{i:02d}   {off[i]:+7d}")
    rms = float(np.sqrt((resid[keep] ** 2).mean())) if keep else 0.0
    print(f"\nedge residual RMS = {rms:.2f} frames over {len(keep)} "
          f"confident edges ({len(info) - len(keep)} discarded)")
    if bad:
        print("inconsistent edges (|residual| > 3 frames):")
        for i, j, r in bad:
            print(f"  e{i:02d}-e{j:02d}  residual {r:+.1f}")
    print("\nA small RMS means every camera captured the SAME trigger burst and "
          "the timestamp spread is pure Pi clock skew.")

    (out / f"temporal_align_b{a.burst:04d}.json").write_text(json.dumps(dict(
        ref=a.ref, burst=a.burst,
        offsets={f"e{i:02d}": int(off[i]) for i in range(N_CAM)},
        edges=[dict(i=i, j=j, lag=l, peak=p, psr=s, residual=float(r))
               for (i, j, l, p, s), r in zip(info, resid)],
        residual_rms=rms), indent=2))
    print(f"wrote {out / f'temporal_align_b{a.burst:04d}.json'}")


if __name__ == "__main__":
    main()
